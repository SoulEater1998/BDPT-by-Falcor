/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Lights.LightHelpers;
import Rendering.Materials.Microfacet;
import Rendering.Materials.InteriorListHelpers;
import Rendering.Volumes.HomogeneousVolumeSampler;
import Rendering.Utils.PixelStats;
import Rendering.RTXDI.RTXDI;
import Utils.Color.ColorHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Debug.PixelDebug;
import Utils.Math.MathHelpers;
import LoadShadingData;
import ColorType;
import PathData;
import Node;
__exported import PathState;
__exported import BDPTParams;

/*
inline void CoordinateSystem(const float3 v1, out float3 v2, out float3 v3)
{
    if(abs(v1.x) > abs(v1.y))
    {
        v2 = float3(-v1.z, 0.f, v1.x) / sqrt(v1.x * v1.x + v1.z * v1.z);
    }
    else
    {
        v2 = float3(0.f, v1.z, -v1.y) / sqrt(v1.y * v1.y + v1.z * v1.z);
    }
    v3 = cross(v1, v2);
}
*/
static const uint tempBufferCount = 10;
static const uint quantLevels = 1 << 10;
//Vertex tempVertexBuffer[tempBufferCount];

/** Path tracer.

    This implements the high-level path tracing logic and is shared among
    different scheduling implementations.
*/


struct PathTracer
{
    /** Interface for querying visibility in the scene.
        This is used in `handleHit`.
    */
    interface IVisibilityQuery
    {
        /** Trace a visibility ray against the scene.
            \param[in] ray Ray.
            \return Returns true if the ray endpoints are mutually visible (i.e. the ray does NOT intersect the scene).
        */
        [mutating] bool traceVisibilityRay(const Ray ray);
    };

    PathTracerParams params; ///< Runtime parameters.
    float3 corner;
    uint leafNodeStart;
    uint counter;
    float3 dimension;
    //RWStructuredBuffer<Vertex> tempVertexBuffer;
    // Samplers
    EnvMapSampler envMapSampler;                    ///< Environment map sampler. Only valid when kUseEnvLight == true.
    EmissiveLightSampler emissiveSampler;           ///< Emissive light sampler. Only valid when kUseEmissiveLights == true.

    // Inputs
    Texture2D<PackedHitInfo> vbuffer;               ///< Fullscreen V-buffer for the primary hits.
    Texture2D<float3> viewDir;                      ///< Optional view direction. Only valid when kUseViewDir == true.
    Texture2D<uint> sampleCount;                    ///< Optional input sample count buffer. Only valid when kSamplesPerPixel == 0.
    Texture2D<uint> sampleOffset;                   ///< Output offset into per-sample buffers. Only valid when kSamplesPerPixel == 0.
    StructuredBuffer<Node> nodes;
    //Buffer<uint> LightPathsIndexBufferCount;
    
    //uint line_number;

    // Outputs
    RWStructuredBuffer<ColorType> sampleColor;            ///< Output per-sample color if kSamplesPerPixel != 1.

    RWTexture2D<float4> outputColor;                      ///< Output color buffer if kSamplesPerPixel == 1.

    RWStructuredBuffer<VertexInfo> LightPathsVertexsBuffer; ///< Output the Vertex of all the light paths.
    RWStructuredBuffer<uint2> LightPathsIndexBuffer;
    RWStructuredBuffer<float4> LightPathsVertexsPositionBuffer;
    //RWStructuredBuffer<float4> CameraPathsIndexBuffer;
    //RWStructuredBuffer<uint> MCounter;

    //RWStructuredBuffer<CameraVertex> CameraPathsVertexsReservoirBuffer;
   
    /*******************************************************************
                                Static members
    *******************************************************************/

    // Render settings that depend on the scene.
    // TODO: Move into scene defines.
    static const bool kUseEnvLight = USE_ENV_LIGHT;
    static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
    static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
    static const bool kUseCurves = USE_CURVES;
    static const bool kUseHairMaterial = USE_HAIR_MATERIAL;

    // Additional specialization.
    static const bool kOutputGuideData = OUTPUT_GUIDE_DATA;

    /** Types of samplable lights.
    */
    enum class LightType
    {
        EnvMap,
        Emissive,
        Analytic
    };

    /** Describes a light sample.
    */
    struct LightSample
    {
        float3  Li;         ///< Incident radiance at the shading point (unshadowed). This is already divided by the pdf.
        float   pdf;        ///< Pdf with respect to solid angle at the shading point.
        float3  origin;     ///< Ray origin for visibility evaluation (offseted to avoid self-intersection).
        float   distance;   ///< Ray distance for visibility evaluation (shortened to avoid self-intersection).
        float3  dir;        ///< Ray direction for visibility evaluation (normalized).
        uint    lightType;  ///< Light type this sample comes from (LightType casted to uint).

        Ray getVisibilityRay() { return Ray(origin, dir, 0.f, distance); }
    };

    /** Describes a path vertex.
    */
    struct PathVertex
    {
        uint index;         ///< Vertex index (0 = camera, 1 = primary hit, 2 = secondary hit, etc.).
        float3 pos;         ///< Vertex position.
        float3 normal;      ///< Shading normal at the vertex (zero if not on a surface).
        float3 faceNormal;  ///< Geometry normal at the vertex (zero if not on a surface).

        /** Initializes a path vertex.
            \param[in] index Vertex index.
            \param[in] pos Vertex position.
            \param[in] normal Shading normal.
            \param[in] faceNormal Geometry normal.
        */
        __init(uint index, float3 pos, float3 normal = float3(0.f), float3 faceNormal = float3(0.f))
        {
            this.index = index;
            this.pos = pos;
            this.normal = normal;
            this.faceNormal = faceNormal;
        }

        /** Get position with offset applied in direction of the geometry normal to avoid self-intersection
            for visibility rays.
            \param[in] rayDir Direction of the visibility ray (does not need to be normalized).
            \return Returns the offseted position.
        */
        float3 getRayOrigin(float3 rayDir)
        {
            return computeRayOrigin(pos, dot(faceNormal, rayDir) >= 0 ? faceNormal : -faceNormal);
        }
    };
    

    /*******************************************************************
                              Member functions
    *******************************************************************/

    /** Check if the path has finished all surface bounces and needs to be terminated.
        Note: This is expected to be called after generateScatterRay(), which increments the bounce counters.
        \param[in] path Path state.
        \return Returns true if path has processed all bounces.
    */
    bool hasFinishedSurfaceBounces(const PathState path)
    {
        const uint diffuseBounces = path.getBounces(BounceType::Diffuse);
        const uint specularBounces = path.getBounces(BounceType::Specular);
        const uint transmissionBounces = path.getBounces(BounceType::Transmission);
        const uint surfaceBounces = diffuseBounces + specularBounces + transmissionBounces;
        return
            (surfaceBounces > kMaxSurfaceBounces) ||
            (diffuseBounces > kMaxDiffuseBounces) ||
            (specularBounces > kMaxSpecularBounces) ||
            (transmissionBounces > kMaxTransmissionBounces);
    }

    /** Compute the total length of a terminated path.
        \param[in] path Path state.
        \return Returns the total number of bounces a path took.
    */
    uint getTerminatedPathLength(const PathState path)
    {
        // Account for the fact that we may have counted one bounce too many (scatter ray at the last path vertex).
        uint diffuseBounces = min(kMaxDiffuseBounces, path.getBounces(BounceType::Diffuse));
        uint specularBounces = min(kMaxSpecularBounces, path.getBounces(BounceType::Specular));
        uint transmissionBounces = min(kMaxTransmissionBounces, path.getBounces(BounceType::Transmission));
        uint surfaceBounces = min(kMaxSurfaceBounces, diffuseBounces + specularBounces + transmissionBounces);
        return surfaceBounces;
    }

    /** Generate the path state for a primary hit in screen space.
        This is treated equivalent to subsequent path vertices to reduce code divergence.
        \param[in] pathID Path ID which encodes pixel and sample index.
        \param[out] path Path state for the primary hit.
    */
    void generatePath(const uint pathID, out PathState path)
    {
        path = {};
        path.setActive();
        path.id = pathID;
        path.thp = float3(1.f);

        const uint2 pixel = path.getPixel();

        // Create primary ray.
        Ray cameraRay = gScene.camera.computeRayPinhole(pixel, params.frameDim);
        if (kUseViewDir) cameraRay.dir = -viewDir[pixel];
        path.origin = cameraRay.origin;
        path.dir = cameraRay.dir;

        // Create sample generator.
        const uint maxSpp = kSamplesPerPixel > 0 ? kSamplesPerPixel : kMaxSamplesPerPixel;
        path.sg = SampleGenerator(pixel, params.frameCount * 2 + 1);

        // Load the primary hit info from the V-buffer.
        const HitInfo hit = HitInfo(vbuffer[pixel]);

        // If invalid, the path is still active and treated as a miss.
        if (hit.isValid())
        {
            path.setHit(hit);
            path.setVertexIndex(1);
        }
    }

    /** Generate the first vertex of the light subpath
        \param[in] pathID Path ID which encodes pixel and sample index.
        \param[in] line Pixel per line.
        \param[out] path Path state for the primary hit.
    */
    void generateLightPath(const uint pathID, out PathState path, out float dl)
    {
        // intial path state
        path = {};
        path.setActive();
        path.id = pathID;

        Vertex lv;
        const uint2 pixel = path.getPixel();
        const uint maxSpp = kSamplesPerPixel > 0 ? kSamplesPerPixel : kMaxSamplesPerPixel;
        path.sg = SampleGenerator(pixel, params.frameCount * 2);

        //Sample a point on Lights
        float3 ray_dir = float3(0,0,0);
        float dirPdf;
        if (!generateLightSample_BDPT(path.sg, lv, ray_dir, dirPdf)) return;
        path.setVertexIndex(0);
        //uint offset = lightPathOffset(path);
        // ShadingData sd = {};
        lv.type = VertexType::Light;
        VertexInfo vInfo = VertexInfo::createLight(lv);
        vInfo.de = 1.0f / lv.pdfFwd;
        vInfo.beta *= dirPdf;
        dl = vInfo.de;
        
        //LightPathsIndexBuffer[LightPathsIndexBuffer.IncrementCounter()] = offset;
        //tempVertexBuffer[0] = lv;
        //LightPathsVertexsBuffer[(pixel.y * line_number + pixel.x) * kMaxSurfaceBounces] = lv;

        path.origin = lv.sd.posW;
        path.normal = lv.sd.N;
        path.dir = ray_dir;

        path.thp = lv.beta;
        path.pdf = dirPdf;

        uint vertexID = LightPathsIndexBuffer.IncrementCounter();
        uint mortonCode = GenMortonCode(path.origin);
        LightPathsIndexBuffer[vertexID] = uint2(vertexID, mortonCode);
        LightPathsVertexsBuffer[vertexID] = vInfo;
        LightPathsVertexsPositionBuffer[vertexID] = float4(path.origin, getIntensity(vInfo.beta));
    }

    /** Set up path for logging and debugging.
        \param[in] path Path state.
    */
    void setupPathLogging(const PathState path)
    {
        printSetPixel(path.getPixel());
        logSetPixel(path.getPixel());
    }

    /** Update the path throughouput.
        \param[in,out] path Path state.
        \param[in] weight Vertex throughput.
    */
    void updatePathThroughput(inout PathState path, const float3 weight)
    {
        path.thp *= weight;
    }

    /** Add radiance to the path contribution.
        \param[in,out] path Path state.
        \param[in] radiance Vertex radiance.
    */
    void addToPathContribution(inout PathState path, const float3 radiance)
    {
        path.L += path.thp * radiance;
    }

    /** Generates a new scatter ray using BSDF importance sampling.
        \param[in] sd Shading data.
        \param[in] bsdf BSDF at the shading point.
        \param[in,out] path The path state.
        \return True if a ray was generated, false otherwise.
    */
    bool generateScatterRay(const ShadingData sd, const IBSDF bsdf, inout PathState path)
    {
        BSDFSample result;
        bool valid = bsdf.sample(sd, path.sg, result, kUseBSDFSampling);
        if (valid) valid = generateScatterRay(result, sd, bsdf, path);

        return valid;
    }

    /** Generates a new scatter ray given a valid BSDF sample.
        \param[in] bs BSDF sample (assumed to be valid).
        \param[in] sd Shading data.
        \param[in] bsdf BSDF at the shading point.
        \param[in,out] path The path state.
        \return True if a ray was generated, false otherwise.
    */
    bool generateScatterRay(const BSDFSample bs, const ShadingData sd, const IBSDF bsdf, inout PathState path)
    {
        const bool isTriangleHit = path.hit.getType() == HitType::Triangle;
        const bool isCurveHit = kUseCurves && (path.hit.getType() == HitType::Curve);
        const bool isHairMaterial = kUseHairMaterial && (sd.mtl.getMaterialType() == MaterialType::Hair);
        // TODO: Decouple geometry from the material.
        const bool isCurvePolyTubeHit = isTriangleHit && isHairMaterial;

        path.dir = bs.wo;
        updatePathThroughput(path, bs.weight);
        path.pdf = bs.pdf;

        path.clearEventFlags();

        // Handle reflection events.
        if (bs.isLobe(LobeType::Reflection))
        {
            // We classify specular events as diffuse if the roughness is above some threshold.
            float roughness = bsdf.getProperties(sd).roughness;
            bool isDiffuse = bs.isLobe(LobeType::DiffuseReflection) || roughness > params.specularRoughnessThreshold;

            if (isDiffuse)
            {
                path.incrementBounces(BounceType::Diffuse);
            }
            else
            {
                path.incrementBounces(BounceType::Specular);
                path.setSpecular();
            }
        }

        // Handle delta events.
        if (bs.isLobe(LobeType::Delta))
        {
            path.setDelta();
        }

        // Handle transmission events.
        if (bs.isLobe(LobeType::Transmission))
        {
            path.incrementBounces(BounceType::Transmission);
            path.setTransmission();

            if (isCurveHit)
            {
                // No need to offset the origin in this case.
            }
            else if (isCurvePolyTubeHit)
            {
                // For curves tessellated into poly-tubes, we make sure that the origin is on the same side as a scatter ray direction
                // so there is no self-intersection.
                path.origin = sd.posW - sd.N * sd.curveRadius * 2.1f;
            }
            else
            {
                // Compute ray origin for next ray segment.
                path.origin = sd.computeNewRayOrigin(false);

                // Update interior list and inside volume flag.
                if (!sd.mtl.isThinSurface())
                {
                    uint nestedPriority = sd.mtl.getNestedPriority();
                    path.interiorList.handleIntersection(sd.materialID, nestedPriority, sd.frontFacing);
                    path.setInsideDielectricVolume(!path.interiorList.isEmpty());
                }
            }
        }

        // Save the shading normal. This is needed for MIS.
        path.normal = sd.N;

        // Mark the path as valid only if it has a non-zero throughput.
        bool valid = any(path.thp > 0.f);

        return valid;
    }

    /** Evaluates the currently configured heuristic for multiple importance sampling (MIS).
        \param[in] n0 Number of samples taken from the first sampling strategy.
        \param[in] p0 Pdf for the first sampling strategy.
        \param[in] n1 Number of samples taken from the second sampling strategy.
        \param[in] p1 Pdf for the second sampling strategy.
        \return Weight for the contribution from the first strategy (p0).
    */
    float evalMIS(float n0, float p0, float n1, float p1)
    {
        switch (MISHeuristic(kMISHeuristic))
        {
        case MISHeuristic::Balance:
        {
            // Balance heuristic
            float q0 = n0 * p0;
            float q1 = n1 * p1;
            return q0 / (q0 + q1);
        }
        case MISHeuristic::PowerTwo:
        {
            // Power two heuristic
            float q0 = (n0 * p0) * (n0 * p0);
            float q1 = (n1 * p1) * (n1 * p1);
            return q0 / (q0 + q1);
        }
        case MISHeuristic::PowerExp:
        {
            // Power exp heuristic
            float q0 = pow(n0 * p0, kMISPowerExponent);
            float q1 = pow(n1 * p1, kMISPowerExponent);
            return q0 / (q0 + q1);
        }
        default:
            return 0.f;
        }
    }

    /** Generates a light sample on the environment map.
        \param[in] vertex Path vertex.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateEnvMapSample(const PathVertex vertex, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {}; // Default initialization to avoid divergence at returns.

        if (!kUseEnvLight) return false;

        // Sample environment map.
        EnvMapSample lightSample;
        if (!envMapSampler.sample(sampleNext2D(sg), lightSample)) return false;

        // Setup returned sample.
        ls.Li = lightSample.pdf > 0.f ? lightSample.Le / lightSample.pdf : float3(0);
        ls.pdf = lightSample.pdf;
        ls.origin = vertex.getRayOrigin(lightSample.dir);
        ls.distance = kRayTMax;
        ls.dir = lightSample.dir;

        return any(ls.Li > 0.f);
    }

    /** Generates a light sample on the emissive geometry.
        \param[in] vertex Path vertex.
        \param[in] upperHemisphere True if only upper hemisphere (w.r.t. shading normal) should be considered.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateEmissiveSample(const PathVertex vertex, const bool upperHemisphere, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {}; // Default initialization to avoid divergence at returns.
        if (!kUseEmissiveLights) return false;

        TriangleLightSample tls;
        if (!emissiveSampler.sampleLight(vertex.pos, vertex.normal, upperHemisphere, sg, tls)) return false;

        // Setup returned sample.
        ls.Li = tls.pdf > 0.f ? tls.Le / tls.pdf : float3(0);
        ls.pdf = tls.pdf;
        // Offset shading and light position to avoid self-intersection.
        float3 lightPos = computeRayOrigin(tls.posW, tls.normalW);
        ls.origin = vertex.getRayOrigin(lightPos - vertex.pos);
        float3 toLight = lightPos - ls.origin;
        ls.distance = length(toLight);
        ls.dir = normalize(toLight);

        return any(ls.Li > 0.f);
    }

    /** Generates a light sample on the emissive geometry for BDPT.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateEmissiveSample_BDPT(inout SampleGenerator sg, out Vertex lv, out float3 ray_dir, out float dirPdf)
    {
        lv = Vertex::init(); // Default initialization to avoid divergence at returns.
        if (!kUseEmissiveLights) return false;

        //sample the lights uniformly.
        float uLight = sampleNext1D(sg);
        uint triangleCount = gScene.lightCollection.getActiveTriangleCount();
        uint idx = min((uint)(uLight * triangleCount), triangleCount - 1); // Safety precaution as the result of the multiplication may be rounded to triangleCount even if uLight < 1.0 when triangleCount is large.
        uint triangleIndex = gScene.lightCollection.activeTriangles[idx];
        float triangleSelectionPdf = 1.f / (float)triangleCount;

        //sample the triangle uniformly.
        float2 u = sampleNext2D(sg);
        const float3 barycentrics = sample_triangle(u);
        const EmissiveTriangle tri = gScene.lightCollection.getTriangle(triangleIndex);
        float3 pos = tri.getPosition(barycentrics);
        float3 normal = tri.normal;
        float2 uv = tri.getTexCoord(barycentrics);
        float3 Le = gScene.materials.evalEmissive(tri.materialID, uv);  

        float denom = max(FLT_MIN, tri.area);
        float posPdf = 1.f / denom;

        //sample a cosine-weighted outgoing direction w.  
        float2 v = sampleNext2D(sg);
        float3 w = sample_cosine_hemisphere_polar(v, dirPdf);
        float3 v2, v3;
        CoordinateSystem(normal, v2, v3);
        w = w.x * v2 + w.y * v3 + w.z * normal; //ray direction
        ray_dir = w;

        if(posPdf <= 0.f || dirPdf <= 0.f || all(Le) <= 0.f) return false;

        // Setup returned sample.
        lv.beta = Le /* * abs(dot(normal, w))*/ / (triangleSelectionPdf * posPdf * dirPdf); //the "beta" of the subpath
        lv.pdfFwd = triangleSelectionPdf * posPdf;
        lv.type = VertexType::Light;
        lv.sd.N = normal;
        lv.sd.posW = pos;

        return any(lv.beta > 0.f);
    }

    /** Generates a light sample on the analytic lights.
        \param[in] vertex Path vertex.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateAnalyticLightSample(const PathVertex vertex, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {}; // Default initialization to avoid divergence at returns.

        uint lightCount = gScene.getLightCount();
        if (!kUseAnalyticLights || lightCount == 0) return false;

        // Sample analytic light source selected uniformly from the light list.
        // TODO: Sample based on estimated contributions as pdf.
        uint lightIndex = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1);

        // Sample local light source.
        AnalyticLightSample lightSample;
        if (!sampleLight(vertex.pos, gScene.getLight(lightIndex), sg, lightSample)) return false;

        // Setup returned sample.
        ls.pdf = lightSample.pdf / lightCount;
        ls.Li = lightSample.Li * lightCount;
        // Offset shading position to avoid self-intersection.
        ls.origin = vertex.getRayOrigin(lightSample.dir);
        // Analytic lights do not currently have a geometric representation in the scene.
        // Do not worry about adjusting the ray length to avoid self-intersections at the light.
        ls.distance = lightSample.distance;
        ls.dir = lightSample.dir;

        return any(ls.Li > 0.f);
    }

    /** Generates a light sample on the analytic lights for BDPT.
        \param[in,out] sg Sample generator.
        \param[out] lv Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateAnalyticLightSample_BDPT(inout SampleGenerator sg, out Vertex lv, out float3 ray_dir, out float dirPdf)
    {
        lv = Vertex::init(); // Default initialization to avoid divergence at returns.

        uint lightCount = gScene.getLightCount();
        if (!kUseAnalyticLights || lightCount == 0) return false;

        // Sample analytic light source selected uniformly from the light list.
        // TODO: Sample based on estimated contributions as pdf.
        uint lightIndex = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1);
        float lightCountPdf = 1.f / (float)lightCount;

        // Sample local light source.
        AnalyticLightSample lightSample;
        if (!sampleLight_BDPT(gScene.getLight(lightIndex), sg, lightSample)) return false;

        lv.sd.posW = lightSample.posW;
        lv.sd.N = lightSample.normalW;
        ray_dir = lightSample.dir;
        lv.beta = lightSample.Li /* * dot(lv.sd.N, ray_dir)*/ / (lightCountPdf * lightSample.pdf * lightSample.distance);
        lv.pdfFwd = lightCountPdf * lightSample.pdf;
        lv.type = VertexType::Light;
        dirPdf = lightSample.distance;

        return any(lv.beta > 0.f);
    }

    /** Return the probabilities for selecting different light types.
        \param[out] p Probabilities.
    */
    void getLightTypeSelectionProbabilities(out float p[3])
    {
        // Set relative probabilities of the different sampling techniques.
        // TODO: These should use estimated irradiance from each light type. Using equal probabilities for now.
        p[0] = kUseEnvLight ? 1.f : 0.f;
        p[1] = kUseEmissiveLights ? 1.f : 0.f;
        p[2] = kUseAnalyticLights ? 1.f : 0.f;

        // Normalize probabilities. Early out if zero.
        float sum = p[0] + p[1] + p[2];
        if (sum == 0.f) return;

        float invSum = 1.f / sum;
        p[0] *= invSum;
        p[1] *= invSum;
        p[2] *= invSum;
    }

    float getEnvMapSelectionProbability()   { float p[3]; getLightTypeSelectionProbabilities(p); return p[0]; }
    float getEmissiveSelectionProbability() { float p[3]; getLightTypeSelectionProbabilities(p); return p[1]; }
    float getAnalyicSelectionProbability()  { float p[3]; getLightTypeSelectionProbabilities(p); return p[2]; }

    /** Select a light type for sampling.
        \param[out] lightType Selected light type.
        \param[out] pdf Probability for selected type.
        \param[in,out] sg Sample generator.
        \return Return true if selection is valid.
    */
    bool selectLightType(out uint lightType, out float pdf, inout SampleGenerator sg)
    {
        float p[3];
        getLightTypeSelectionProbabilities(p);

        float u = sampleNext1D(sg);

        [unroll]
        for (lightType = 0; lightType < 3; ++lightType)
        {
            if (u < p[lightType])
            {
                pdf = p[lightType];
                return true;
            }
            u -= p[lightType];
        }

        return false;
    }

    /** Samples a light source in the scene.
        This function first stochastically selects a type of light source to sample,
        and then calls that the sampling function for the chosen light type.
        The upper/lower hemisphere is defined as the union of the hemispheres w.r.t. to the shading and face normals.
        \param[in] vertex Path vertex.
        \param[in] sampleUpperHemisphere True if the upper hemisphere should be sampled.
        \param[in] sampleLowerHemisphere True if the lower hemisphere should be sampled.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateLightSample(const PathVertex vertex, const bool sampleUpperHemisphere, const bool sampleLowerHemisphere, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {};

        uint lightType;
        float selectionPdf;
        if (!selectLightType(lightType, selectionPdf, sg)) return false;

        bool valid = false;
        if (kUseEnvLight && lightType == (uint)LightType::EnvMap) valid = generateEnvMapSample(vertex, sg, ls);
        if (kUseEmissiveLights && lightType == (uint)LightType::Emissive)
        {
            // Emissive light samplers have an option to exclusively sample the upper hemisphere.
            bool upperHemisphere = sampleUpperHemisphere && !sampleLowerHemisphere;
            valid = generateEmissiveSample(vertex, upperHemisphere, sg, ls);
        }
        if (kUseAnalyticLights && lightType == (uint)LightType::Analytic)
        {
            valid = generateAnalyticLightSample(vertex, sg, ls);
        }
        if (!valid) return false;

        // Reject samples in non-requested hemispheres.
        float cosTheta = dot(vertex.normal, ls.dir);
        // Flip the face normal to point in the same hemisphere as the shading normal.
        float3 faceNormal = sign(dot(vertex.normal, vertex.faceNormal)) * vertex.faceNormal;
        float cosThetaFace = dot(faceNormal, ls.dir);
        if (!sampleUpperHemisphere && (max(cosTheta, cosThetaFace) >= -kMinCosTheta)) return false;
        if (!sampleLowerHemisphere && (min(cosTheta, cosThetaFace) <= kMinCosTheta)) return false;

        // Account for light type selection.
        ls.lightType = lightType;
        ls.pdf *= selectionPdf;
        ls.Li /= selectionPdf;

        return true;
    }

    /** Samples a light source in the scene for BDPT.
        This function first stochastically selects a type of light source to sample,
        and then calls that the sampling function for the chosen light type.
        \param[in,out] sg Sample generator.
        \param[out] lv Struct describing valid samples vertex.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateLightSample_BDPT(inout SampleGenerator sg, out Vertex lv, out float3 ray_dir, out float dirPdf)
    {
        //lv = {};

        uint lightType;
        float selectionPdf;
        if (!selectLightType(lightType, selectionPdf, sg)) return false;

        bool valid = false;
        if (kUseEnvLight && lightType == (uint)LightType::EnvMap) valid = false;
        if (kUseEmissiveLights && lightType == (uint)LightType::Emissive)
        {
            // Emissive light samplers have an option to exclusively sample the upper hemisphere.
            valid = generateEmissiveSample_BDPT(sg, lv, ray_dir, dirPdf);
        }
        if (kUseAnalyticLights && lightType == (uint)LightType::Analytic)
        {
            valid = generateAnalyticLightSample_BDPT(sg, lv, ray_dir, dirPdf);
        }
        if (!valid) return false;

        // Account for light type selection.
        lv.pdfFwd *= selectionPdf;
        lv.beta /= selectionPdf;

        return true;
    }

    /** Handle hits on dielectrics.
        \return True if this is an valid intersection, false if it is rejected.
    */
    bool handleNestedDielectrics(inout ShadingData sd, inout PathState path)
    {
        // Check for false intersections.
        uint nestedPriority = sd.mtl.getNestedPriority();
        if (!path.interiorList.isTrueIntersection(nestedPriority))
        {
            // If it is a false intersection, we reject the hit and continue the path
            // on the other side of the interface.
            // If the offset position is not quite large enough, due to self-intersections
            // it is possible we repeatedly hit the same surface and try to reject it.
            // This has happened in a few occasions with large transmissive triangles.
            // As a workaround, count number of rejected hits and terminate the path if too many.
            if (path.rejectedHits < kMaxRejectedHits)
            {
                path.rejectedHits++;
                path.interiorList.handleIntersection(sd.materialID, nestedPriority, sd.frontFacing);
                path.origin = sd.computeNewRayOrigin(false);
                path.decrementVertexIndex();
            }
            else
            {
                path.terminate();
            }
            return false;
        }

        // Compute index of refraction for medium on the outside.
        sd.IoR = computeOutsideIoR(path.interiorList, sd.materialID, sd.frontFacing);

        return true;
    }

    /** Apply russian roulette to terminate paths early.
        \param[in,out] path Path.
        \param[in] u Uniform random number in [0,1).
        \return Returns true if path was terminated.
    */
    bool terminatePathByRussianRoulette(inout PathState path, float u)
    {
        const float rrVal = luminance(path.thp);
        const float prob = max(0.f, 1.f - rrVal);
        if (u < prob)
        {
            path.terminate();
            return true;
        }
        path.thp /= 1.f - prob;
        return false;
    }

    /** Helper to create a texture sampler instance.
        The method for computing texture level-of-detail depends on the configuration.
        \param[in] path Path state.
        \param[in] isPrimaryTriangleHit True if primary hit on a triangle.
        \return Texture sampler instance.
    */
    ITextureSampler createTextureSampler(const PathState path, bool isPrimaryHit, bool isTriangleHit)
    {
        if (kPrimaryLodMode == TexLODMode::RayDiffs && isPrimaryHit && isTriangleHit)
        {
            // Filtered lookups at primary hit on triangle.
            float2 ddx, ddy;
            computeDerivativesAtPrimaryTriangleHit(path.hit.getTriangleHit(), path.getPixel(), params.frameDim, ddx, ddy);
            return ExplicitGradientTextureSampler(ddx, ddy);
        }
        else
        {
            float lod = isPrimaryHit ? 0.f : params.lodBias;
            return ExplicitLodTextureSampler(lod);
        }
    }

    /** Handle the case when a scatter ray hits a surface.
        After handling the hit, a new scatter ray is generated or the path is terminated.
        \param[in,out] path The path state.
        \param[in,out] vq Visibility query.
    */
    void handleHit<VisibilityQuery : IVisibilityQuery>(inout PathState path, inout VisibilityQuery vq)
    {
        // Upon hit:
        // - Load vertex/material data
        // - Compute MIS weight if path.getVertexIndex() > 1 and emissive hit
        // - Add emitted radiance
        // - Sample light(s) using shadow rays
        // - Sample scatter ray or terminate

        const bool isPrimaryHit = path.getVertexIndex() == 1;
        const HitType hitType = path.hit.getType();
        const bool isTriangleHit = hitType == HitType::Triangle;
        const bool isCurveHit = kUseCurves && (hitType == HitType::Curve);

        let lod = createTextureSampler(path, isPrimaryHit, isTriangleHit);

        // Load shading data. This is a long latency operation.
        ShadingData sd = loadShadingData(path.hit, path.origin, path.dir, isPrimaryHit, lod);

        const bool isHairMaterial = kUseHairMaterial && (sd.mtl.getMaterialType() == MaterialType::Hair);
        // TODO: Decouple geometry from the material.
        const bool isCurvePolyTubeHit = isTriangleHit && isHairMaterial;

        // Account for volume absorption.
        if (!path.interiorList.isEmpty())
        {
            const uint materialID = path.interiorList.getTopMaterialID();
            const HomogeneousVolumeData hvd = gScene.materials.getHomogeneousVolumeData(materialID);
            updatePathThroughput(path, HomogeneousVolumeSampler::evalTransmittance(hvd, length(sd.posW - path.origin)));
        }

        // Reject false hits in nested dielectrics.
        if (!handleNestedDielectrics(sd, path)) return;

        logPathVertex();

        // Create BSDF instance and query its properties.
        const IBSDF bsdf = gScene.materials.getBSDF(sd, lod);
        BSDFProperties bsdfProperties = bsdf.getProperties(sd);

        // Disable specular lobes if caustics are disabled and path already contains a diffuse vertex.
        bool isSpecular = bsdfProperties.roughness <= params.specularRoughnessThreshold;
        if (kDisableCaustics && path.getBounces(BounceType::Diffuse) > 0 && isSpecular)
        {
            sd.mtl.setActiveLobes((uint)LobeType::Diffuse);
        }

        // Optionally disable emission inside volumes.
        if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume())
        {
            bsdfProperties.emission = float3(0.f);
        }

        // Check if the scatter event is samplable by the light sampling technique.
        const bool isLightSamplable = path.isLightSamplable();

        // Add emitted radiance.
        // The primary hit is always included, secondary hits only if emissive lights are enabled and the full light contribution hasn't been sampled elsewhere.
        bool computeEmissive = isPrimaryHit || kUseEmissiveLights && (!kUseNEE || kUseMIS || !path.isLightSampled() || !isLightSamplable);

        // With RTXDI enabled, we sample the full direct illumination contribution on the primary hit.
        // Skip any additional contribution on the secondary hit unless it comes from a scatter event
        // that RTXDI cannot handle, such as transmission or delta scattering events.
        if (kUseRTXDI && path.getVertexIndex() == 2 && !path.isTransmission() && !path.isDelta()) computeEmissive = false;

        float3 attenuatedEmission = 0.f;

        if (computeEmissive && any(bsdfProperties.emission > 0.f))
        {
            float misWeight = 1.f;
            if (kUseEmissiveLights && kUseNEE && kUseMIS && isTriangleHit && !isPrimaryHit && path.isLightSampled() && isLightSamplable)
            {
                // If NEE and MIS are enabled, and we've already sampled emissive lights,
                // then we need to evaluate the MIS weight here to account for the remaining contribution.
                // Note that MIS is only applied for hits on emissive triangles (other emissive geometry is not supported).

                // Prepare hit point struct with data needed for emissive light PDF evaluation.
                TriangleHit triangleHit = path.hit.getTriangleHit();
                TriangleLightHit hit;
                hit.triangleIndex = gScene.lightCollection.getTriangleIndex(triangleHit.instanceID, triangleHit.primitiveIndex);
                hit.posW = sd.posW;
                hit.normalW = sd.frontFacing ? sd.faceN : -sd.faceN;

                // Evaluate PDF at the hit, had it been generated with light sampling.
                // Emissive light samplers have an option to exclusively sample the upper hemisphere.
                bool upperHemisphere = path.isLightSampledUpper() && !path.isLightSampledLower();
                float lightPdf = getEmissiveSelectionProbability() * emissiveSampler.evalPdf(path.origin, path.normal, upperHemisphere, hit);

                // Compute MIS weight by combining this with BSDF sampling.
                // Note we can assume path.pdf > 0.f since we shouldn't have got here otherwise.
                misWeight = evalMIS(1, path.pdf, 1, lightPdf);
            }

            // Accumulate emitted radiance weighted by path throughput and MIS weight.
            addToPathContribution(path, misWeight * bsdfProperties.emission);

            attenuatedEmission = path.thp * misWeight * bsdfProperties.emission;
        }

        // Terminate after scatter ray on last vertex has been processed.
        if (hasFinishedSurfaceBounces(path))
        {
            path.terminate();
            return;
        }

        // Compute origin for rays traced from this path vertex.
        if (isCurveHit)
        {
            // For curves, we set the new origin at the sphere center.
            path.origin = sd.posW - sd.N * sd.curveRadius;
        }
        else if (isCurvePolyTubeHit)
        {
            // For curves tessellated into poly-tubes, we offset the new origin away from the curve center.
            path.origin = sd.posW + sd.N * sd.curveRadius * 0.1f;
        }
        else
        {
            path.origin = sd.computeNewRayOrigin();
        }

        // Determine if BSDF has non-delta lobes.
        const uint lobes = bsdf.getLobes(sd);
        const bool hasNonDeltaLobes = (lobes & (uint)LobeType::NonDelta) != 0;

        // Check if we should apply NEE.
        const bool applyNEE = kUseNEE && hasNonDeltaLobes;

        // Check if sample from RTXDI should be applied instead of NEE.
        const bool applyRTXDI = kUseRTXDI && isPrimaryHit && hasNonDeltaLobes;

        // TODO: Support multiple shadow rays.
        path.setLightSampled(false, false);
        if (applyNEE || applyRTXDI)
        {
            LightSample ls = {};
            bool validSample = false;

            if (applyRTXDI)
            {
                // Query final sample from RTXDI.
                validSample = gRTXDI.getFinalSample(path.getPixel(), ls.dir, ls.distance, ls.Li);
                ls.origin = path.origin;
            }
            else
            {
                // Setup path vertex.
                PathVertex vertex = PathVertex(path.getVertexIndex(), sd.posW, sd.N, sd.faceN);

                // Determine if upper/lower hemispheres need to be sampled.
                bool sampleUpperHemisphere = isCurveHit || isCurvePolyTubeHit || ((lobes & (uint)LobeType::NonDeltaReflection) != 0);
                if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume()) sampleUpperHemisphere = false;
                bool sampleLowerHemisphere = isCurveHit || isCurvePolyTubeHit || ((lobes & (uint)LobeType::NonDeltaTransmission) != 0);

                // Sample a light.
                validSample = generateLightSample(vertex, sampleUpperHemisphere, sampleLowerHemisphere, path.sg, ls);
                path.setLightSampled(sampleUpperHemisphere, sampleLowerHemisphere);
            }

            if (validSample)
            {
                // Apply MIS weight.
                if (kUseMIS && !applyRTXDI && ls.lightType != (uint)LightType::Analytic)
                {
                    float scatterPdf = bsdf.evalPdf(sd, ls.dir, kUseBSDFSampling);
                    ls.Li *= evalMIS(1, ls.pdf, 1, scatterPdf);
                }

                float3 weight = bsdf.eval(sd, ls.dir, path.sg);
                float3 Lr = weight * ls.Li;
                if (any(Lr > 0.f))
                {
                    Ray ray = ls.getVisibilityRay();

                    if (isCurvePolyTubeHit)
                    {
                        // For curves tessellated into poly-tubes, we make sure that the origin is on the same side as light
                        // so there is no self-shadowing (transmission lobe of hair BSDF takes care of that).
                        if (dot(sd.N, ray.dir) < 0.f)
                        {
                            ray.origin = ray.origin - sd.N * sd.curveRadius * 2.1f;
                        }
                    }

                    logTraceRay(PixelStatsRayType::Visibility);
                    bool visible = vq.traceVisibilityRay(ray);
                    if (visible) addToPathContribution(path, Lr);
                }
            }
        }

        // Russian roulette to terminate paths early.
        if (kUseRussianRoulette)
        {
            if (terminatePathByRussianRoulette(path, sampleNext1D(path.sg))) return;
        }

        const bool wasDeltaOnlyPathBeforeScattering = path.isDeltaOnlyPath();

        // Generate the next path segment or terminate.
        bool valid = generateScatterRay(sd, bsdf, path);

        // Check if this is the last path vertex.
        const bool isLastVertex = hasFinishedSurfaceBounces(path);

        // Terminate if this is the last path vertex and light sampling already completely sampled incident radiance.
        if (kUseNEE && !kUseMIS && isLastVertex && path.isLightSamplable()) valid = false;

        // Terminate caustics paths.
        if (kDisableCaustics && path.getBounces(BounceType::Diffuse) > 0 && path.isSpecular()) valid = false;

        if (!valid)
        {
            path.terminate();
        }
    }

    inline uint3 BitExpansion(uint3 x)
    {
        // https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
        x = (x | x << 16) & 0x30000ff;
        x = (x | x << 8) & 0x300f00f;
        x = (x | x << 4) & 0x30c30c3;
        x = (x | x << 2) & 0x9249249;
        return x;
    }

    uint GenMortonCode(float3 pos) {
        
        float3 normPos = (pos - corner) / dimension;
        uint3 quantPos = min(max(0, uint3(normPos * quantLevels)), quantLevels - 1);
        quantPos = BitExpansion(quantPos);
        return quantPos.x * 4 + quantPos.y * 2 + quantPos.z;
    }

    void lightPathHandleHit<VisibilityQuery : IVisibilityQuery>(inout PathState path, inout VisibilityQuery vq, inout float dl) {
        const bool isPrimaryHit = false;
        const HitType hitType = path.hit.getType();
        const bool isTriangleHit = hitType == HitType::Triangle;
        const bool isCurveHit = kUseCurves && (hitType == HitType::Curve);

        let lod = createTextureSampler(path, false, isTriangleHit);

        // Load shading data. This is a long latency operation.
        ShadingData sd = loadShadingData(path.hit, path.origin, path.dir, isPrimaryHit, lod);

        const bool isHairMaterial = kUseHairMaterial && (sd.mtl.getMaterialType() == MaterialType::Hair);
        // TODO: Decouple geometry from the material.
        const bool isCurvePolyTubeHit = isTriangleHit && isHairMaterial;

        if (hasFinishedSurfaceBounces(path) || path.getVertexIndex() >= kMaxSurfaceBounces) {
            path.terminate();
            return;
        }

        // Account for volume absorption.
        if (!path.interiorList.isEmpty())
        {
            const uint materialID = path.interiorList.getTopMaterialID();
            const HomogeneousVolumeData hvd = gScene.materials.getHomogeneousVolumeData(materialID);
            updatePathThroughput(path, HomogeneousVolumeSampler::evalTransmittance(hvd, length(sd.posW - path.origin)));

        }

        // Reject false hits in nested dielectrics.
        if (!handleNestedDielectrics(sd, path)) return;

        // TODO: move this to TraceLightPath
        //uint offset = lightPathOffset(path);
        //Vertex prev = LightPathsVertexsBuffer[offset - 1].unpack(false);

        Vertex vertex = Vertex::create(sd, path.thp, path.pdf, 0, VertexType::SurfaceOrMedium);
        vertex.pdfFwd = vertex.ConvertDensity(path);

        const IBSDF bsdf = gScene.materials.getBSDF(sd, lod);
        BSDFProperties bsdfProperties = bsdf.getProperties(sd);

        const uint lobes = bsdf.getLobes(sd);

        // Disable specular lobes if caustics are disabled and path already contains a diffuse vertex.
        bool isSpecular = bsdfProperties.roughness <= params.specularRoughnessThreshold;
        if (kDisableCaustics && path.getBounces(BounceType::Diffuse) > 0 && isSpecular)
        {
            sd.mtl.setActiveLobes((uint)LobeType::Diffuse);
        }

        // Optionally disable emission inside volumes.
        if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume())
        {
            bsdfProperties.emission = float3(0.f);
        }

        BSDFSample bs;
        bool valid = bsdf.sample(sd, path.sg, bs, kUseBSDFSampling);

        float3 wi = sd.V;
        ShadingData sdRev = sd;
        sdRev.V = bs.wo;
        const IBSDF bsdfRev = gScene.materials.getBSDF(sdRev, lod);
        float pdfRev = bsdfRev.evalPdf(sdRev, wi);
        bool hasDelta = ((uint)LobeType::Delta & lobes) != 0;
        float N = hasDelta ? 0 : 2.0;
        pdfRev = vertex.ConvertRevDensity(path, pdfRev);

        VertexInfo vInfo = VertexInfo::create(path, vertex);
        dl = (N + pdfRev * dl) / remap0(vertex.pdfFwd);
        vInfo.de = dl;
        //LightPathsVertexsBuffer[offset] = vInfo;

        float3 albedo =   bsdfProperties.diffuseReflectionAlbedo 
                        + bsdfProperties.diffuseTransmissionAlbedo 
                        + bsdfProperties.specularReflectionAlbedo 
                        + bsdfProperties.specularTransmissionAlbedo;

        if (!hasDelta) {
            uint vertexID = LightPathsIndexBuffer.IncrementCounter();
            uint mortonCode = GenMortonCode(vertex.sd.posW);
            LightPathsVertexsBuffer[vertexID] = vInfo;
            LightPathsIndexBuffer[vertexID] = uint2(vertexID, mortonCode);
            LightPathsVertexsPositionBuffer[vertexID] = float4(vertex.sd.posW, (!vertex.delta) ? getIntensity(vertex.beta) : 0);
        }

        if (isCurveHit)
        {
            // For curves, we set the new origin at the sphere center.
            path.origin = sd.posW - sd.N * sd.curveRadius;
        }
        else if (isCurvePolyTubeHit)
        {
            // For curves tessellated into poly-tubes, we offset the new origin away from the curve center.
            path.origin = sd.posW + sd.N * sd.curveRadius * 0.1f;
        }
        else
        {
            path.origin = sd.computeNewRayOrigin();
        }
        //InterlockedAdd(MCounter[path.getVertexIndex()], 1);

        if (valid) valid = generateScatterRay(bs, sd, bsdf, path);
/*
        if (((lobes & (uint)LobeType::DeltaReflection) != 0) || ((lobes & (uint)LobeType::DeltaTransmission) != 0)) {
            vertex.delta = true;
            LightPathsVertexsBuffer[offset].setDelta(true);
            pdfRev = 0;
            //path.pdf = 0;
        }
*/
        //LightPathsVertexsBuffer[offset - 1].pdfRev = vertex.ConvertDensity(pdfRev, prev);
        

        const bool isLastVertex = hasFinishedSurfaceBounces(path) || path.getVertexIndex() + 1 >= kMaxSurfaceBounces;

        if (isLastVertex || !valid) {
            path.terminate();
        }
        
    }

    void CameraPathHandleHit<VisibilityQuery : IVisibilityQuery>(inout PathState path, inout VisibilityQuery vq, inout float de)
    {
        const bool isPrimaryHit = path.getVertexIndex() == 1;
        const uint index = path.getVertexIndex() - 1;
        const HitType hitType = path.hit.getType();
        const bool isTriangleHit = hitType == HitType::Triangle;
        const bool isCurveHit = kUseCurves && (hitType == HitType::Curve);

        let lod = createTextureSampler(path, isPrimaryHit, isTriangleHit);

        // Load shading data. This is a long latency operation.
        ShadingData sd = loadShadingData(path.hit, path.origin, path.dir, isPrimaryHit, lod);

        const bool isHairMaterial = kUseHairMaterial && (sd.mtl.getMaterialType() == MaterialType::Hair);
        // TODO: Decouple geometry from the material.
        const bool isCurvePolyTubeHit = isTriangleHit && isHairMaterial;

        if (hasFinishedSurfaceBounces(path) || path.getVertexIndex() - 1 >= kMaxSurfaceBounces) {
            path.terminate();
            return;
        }

        // Account for volume absorption.
        if (!path.interiorList.isEmpty())
        {
            const uint materialID = path.interiorList.getTopMaterialID();
            const HomogeneousVolumeData hvd = gScene.materials.getHomogeneousVolumeData(materialID);
            updatePathThroughput(path, HomogeneousVolumeSampler::evalTransmittance(hvd, length(sd.posW - path.origin)));
        }

        // Reject false hits in nested dielectrics.
        if (!handleNestedDielectrics(sd, path)) return;

        // uint bounce = path.getVertexIndex();
        // Vertex vertex = Vertex::init();
        // Vertex *vertex = &LightPathsVertexsBuffer[lightPathOffset(path)]; // = &tempVertexBuffer[bounce];
    
        Vertex vertex = Vertex::create(sd, path.thp, path.pdf, 0, VertexType::SurfaceOrMedium);
        vertex.pdfFwd = vertex.ConvertDensity(path);
        
        const IBSDF bsdf = gScene.materials.getBSDF(sd, lod);
        BSDFProperties bsdfProperties = bsdf.getProperties(sd);

        const uint lobes = bsdf.getLobes(sd);
        const bool hasNonDeltaLobes = (lobes & (uint)LobeType::NonDelta) != 0;

        // Disable specular lobes if caustics are disabled and path already contains a diffuse vertex.
        bool isSpecular = bsdfProperties.roughness <= params.specularRoughnessThreshold;
        if (kDisableCaustics && path.getBounces(BounceType::Diffuse) > 0 && isSpecular)
        {
            sd.mtl.setActiveLobes((uint)LobeType::Diffuse);
        }

        // Optionally disable emission inside volumes.
        if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume())
        {
            bsdfProperties.emission = float3(0.f);
        }

        BSDFSample bs;
        bool valid = bsdf.sample(sd, path.sg, bs, kUseBSDFSampling);

        float3 wi = sd.V;
        ShadingData sdRev = sd;
        sdRev.V = bs.wo;
        const IBSDF bsdfRev = gScene.materials.getBSDF(sdRev, lod);
        float pdfRev = bsdfRev.evalPdf(sdRev, wi);

        bool hasDelta = (lobes & (uint)LobeType::Delta) != 0;
        vertex.delta = hasDelta;
        float N = (isPrimaryHit ? 2.f : 1.f);

        de = (N + de * vertex.ConvertRevDensity(path, pdfRev)) / remap0(vertex.pdfFwd);

        bool sampleUpperHemisphere = isCurveHit || isCurvePolyTubeHit || ((lobes & (uint)LobeType::NonDeltaReflection) != 0);
        if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume()) sampleUpperHemisphere = false;
        bool sampleLowerHemisphere = isCurveHit || isCurvePolyTubeHit || ((lobes & (uint)LobeType::NonDeltaTransmission) != 0);

        //s >= 1
        ConnectBDPT(path, vertex, vq, de, sampleUpperHemisphere, sampleLowerHemisphere);

        // s == 1
        // TODO:trans
        if (isPrimaryHit) {
            bool validSample = false;

            // Sample a point on a light and connect it to the camera subpath
            PathVertex pvertex = PathVertex(path.getVertexIndex(), sd.posW, sd.N, sd.faceN);

            Vertex lightVertex = {};
            float3 ray_dir;
            float dirPdf;
            validSample = generateLightSample_BDPT(path.sg, lightVertex, ray_dir, dirPdf);

            ray_dir = normalize(lightVertex.sd.posW - pvertex.pos);

            float cosTheta = dot(pvertex.normal, ray_dir);
            // Flip the face normal to point in the same hemisphere as the shading normal.
            float3 faceNormal = sign(dot(pvertex.normal, pvertex.faceNormal)) * pvertex.faceNormal;
            float cosThetaFace = dot(faceNormal, ray_dir);
            if (!sampleUpperHemisphere && (max(cosTheta, cosThetaFace) >= -kMinCosTheta)) validSample = false;
            if (!sampleLowerHemisphere && (min(cosTheta, cosThetaFace) <= kMinCosTheta)) validSample = false;

            if (validSample)
            {
                lightVertex.beta = lightVertex.beta * dirPdf; // / abs(dot(ray_dir, lightVertex.sd.N));
                float3 wi = lightVertex.sd.posW - vertex.sd.posW;
                lightVertex.beta = lightVertex.beta * abs(dot(normalize(wi), lightVertex.sd.N)) / dot(wi, wi);
                float3 lightPos = computeRayOrigin(lightVertex.sd.posW, lightVertex.sd.N);
                float3 origin = pvertex.getRayOrigin(lightPos - sd.posW);
                float3 toLight = lightPos - origin;
                float distSqr = max(dot(toLight, toLight), kMinLightDistSqr);
                float distance = sqrt(distSqr);
                float3 dir = toLight / distance;

                Ray ray = Ray(origin, dir, 0, distance);
                float3 weight = bsdf.eval(sd, dir, path.sg);
                float3 Lr = vertex.beta * weight * lightVertex.beta;

                if (params.hasFlag(BDPTFlags::s1) && any(Lr > 0.f))
                {
                    // Ray ray = ls.getVisibilityRay();
                    logTraceRay(PixelStatsRayType::Visibility);
                    bool visible = vq.traceVisibilityRay(ray);
                    // visible = true;
                    if (visible) path.L += Lr * BDPTMIS(path, lightVertex, vertex, de);
                }
            }
        }
       
        // s == 0

        bool computeEmissive = kUseEmissiveLights;
        TriangleHit triangleHit = path.hit.getTriangleHit();
        TriangleLightHit hit;
        hit.triangleIndex = gScene.lightCollection.getTriangleIndex(triangleHit.instanceID, triangleHit.primitiveIndex);
        hit.posW = sd.posW;
        hit.normalW = sd.frontFacing ? sd.faceN : -sd.faceN;

        //bool sampleLowerHemisphere = isCurveHit || isCurvePolyTubeHit || ((lobes & (uint)LobeType::NonDeltaTransmission) != 0);
        if (params.hasFlag(BDPTFlags::s0) && computeEmissive && any(bsdfProperties.emission > 0.f)) {
            path.L += vertex.beta * bsdfProperties.emission * BDPTMIS(path, hit, de);
        }

        if (isCurveHit)
        {
            // For curves, we set the new origin at the sphere center.
            path.origin = sd.posW - sd.N * sd.curveRadius;
        }
        else if (isCurvePolyTubeHit)
        {
            // For curves tessellated into poly-tubes, we offset the new origin away from the curve center.
            path.origin = sd.posW + sd.N * sd.curveRadius * 0.1f;
        }
        else
        {
            path.origin = sd.computeNewRayOrigin();
        }

        if (valid) valid = generateScatterRay(bs, sd, bsdf, path);

        const bool isLastVertex = hasFinishedSurfaceBounces(path) || path.getVertexIndex() >= kMaxSurfaceBounces;

        if (isLastVertex || !valid) {
            path.terminate();
        }
    }

/*
    void pushInBuffer(PathState path, uint line) {
        uint2 pixel = path.getPixel();
        uint base = (pixel.y * line + pixel.x) * kMaxSurfaceBounces;
        uint bounces = path.getVertexIndex();
        for (uint i = 0; i < bounces; ++i) {
            LightPathsVertexsBuffer[base + i] = tempVertexBuffer[i];
        }
    }
*/
    uint lightPathOffset(PathState path) {
        uint2 pixel = path.getPixel();
        uint base = (pixel.y * kLightPassWidth + pixel.x) * kMaxSurfaceBounces;
        return base + path.getVertexIndex();
    }

    uint cameraPathOffset(PathState path, int offset = 0) {
        uint2 pixel = path.getPixel();
        uint base = (pixel.y * params.frameDim.x + pixel.x) * kMaxSurfaceBounces;
        return base + path.getVertexIndex() - 1 + offset;
    }

    float3 RISWeight(Vertex cameraPathEnd, Vertex sample, inout PathState path, const float3 beta, bool fr0 = true, bool g = true, bool fr1 = true) {
        // RISBDPT
        const bool isPrimaryHit = path.getVertexIndex() == 1;
        const HitType hitType = path.hit.getType();
        const bool isTriangleHit = hitType == HitType::Triangle;

        let lod = createTextureSampler(path, false, isTriangleHit);

        float3 result = beta;
        //f = result;

        result *= g ? G(cameraPathEnd, sample) : 1;
        //f *= G(cameraPathEnd, sample);
        const IBSDF bsdf0 = gScene.materials.getBSDF(cameraPathEnd.sd, lod);
        const IBSDF bsdf1 = gScene.materials.getBSDF(sample.sd, lod);
        float3 toSample = normalize(sample.sd.posW - cameraPathEnd.sd.posW);

        const uint lobes0 = bsdf0.getLobes(cameraPathEnd.sd);
        const uint lobes1 = bsdf1.getLobes(sample.sd);
        /*
                bool sampleUpperHemisphere_cameraSide = ((lobes0 & (uint)LobeType::NonDeltaReflection) != 0);
                if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume()) sampleUpperHemisphere_cameraSide = false;
                bool sampleLowerHemisphere_cameraSide = ((lobes0 & (uint)LobeType::NonDeltaTransmission) != 0);
                float cosTheta_cameraSide = dot(cameraPathEnd.sd.N, toSample);
                // Flip the face normal to point in the same hemisphere as the shading normal.
                float3 faceNormal_cameraSide = sign(dot(cameraPathEnd.sd.N, cameraPathEnd.sd.faceN)) * cameraPathEnd.sd.faceN;
                float cosThetaFace_cameraSide = dot(faceNormal_cameraSide, toSample);
                if (!sampleUpperHemisphere_cameraSide && (max(cosTheta_cameraSide, cosThetaFace_cameraSide) >= -kMinCosTheta)) return 0;
                if (!sampleLowerHemisphere_cameraSide && (min(cosTheta_cameraSide, cosThetaFace_cameraSide) <= kMinCosTheta)) return 0;

                bool sampleUpperHemisphere_lightSide = ((lobes1 & (uint)LobeType::NonDeltaReflection) != 0);
                bool sampleLowerHemisphere_lightSide = ((lobes1 & (uint)LobeType::NonDeltaTransmission) != 0);
                float cosTheta_lightSide = dot(sample.sd.N, -toSample);
                // Flip the face normal to point in the same hemisphere as the shading normal.
                float3 faceNormal_lightSide = sign(dot(sample.sd.N, sample.sd.faceN)) * sample.sd.faceN;
                float cosThetaFace_lightSide = dot(faceNormal_lightSide, -toSample);
                if (!sampleUpperHemisphere_lightSide && (max(cosTheta_lightSide, cosThetaFace_lightSide) >= -kMinCosTheta)) return 0;
                if (!sampleLowerHemisphere_lightSide && (min(cosTheta_lightSide, cosThetaFace_lightSide) <= kMinCosTheta)) return 0;

                bool cameraSideTrans = sampleLowerHemisphere_cameraSide && (min(cosTheta_cameraSide, cosThetaFace_cameraSide) <= kMinCosTheta);
                bool lightSideTrans = sampleLowerHemisphere_lightSide && (min(cosTheta_lightSide, cosThetaFace_lightSide) <= kMinCosTheta);
                bool sameMaterial = cameraPathEnd.sd.materialID == sample.sd.materialID;

                if (cameraSideTrans) {
                    const uint materialID = cameraPathEnd.sd.materialID;
                    const HomogeneousVolumeData hvd = gScene.materials.getHomogeneousVolumeData(materialID);
                    result *= HomogeneousVolumeSampler::evalTransmittance(hvd, length(cameraPathEnd.sd.posW - sample.sd.posW));
                }
                else if (lightSideTrans) {
                    const uint materialID = sample.sd.materialID;
                    const HomogeneousVolumeData hvd = gScene.materials.getHomogeneousVolumeData(materialID);
                    result *= HomogeneousVolumeSampler::evalTransmittance(hvd, length(cameraPathEnd.sd.posW - sample.sd.posW));
                }
                if (sampleLowerHemisphere_lightSide &&
                    (min(cosTheta_lightSide, cosThetaFace_lightSide) <= kMinCosTheta) &&
                    sampleLowerHemisphere_cameraSide &&
                    (min(cosTheta_cameraSide, cosThetaFace_cameraSide) <= kMinCosTheta) &&
                    cameraPathEnd.sd.materialID == sample.sd.materialID)
                {
                    const uint materialID = cameraPathEnd.sd.materialID;
                    const HomogeneousVolumeData hvd = gScene.materials.getHomogeneousVolumeData(materialID);
                    result *= HomogeneousVolumeSampler::evalTransmittance(hvd, length(cameraPathEnd.sd.posW - sample.sd.posW));
                }
        */
        result *= fr0 ? bsdf0.eval(cameraPathEnd.sd, toSample, path.sg) : 1;
        //f *= bsdf0.eval(cameraPathEnd.sd, toSample, path.sg);
        result *= fr1 ? bsdf1.eval(sample.sd, -toSample, path.sg) : 1;
        //f *= bsdf1.eval(sample.sd, -toSample, path.sg);
        //result *= abs(dot(sample.sd.N, wo));
        
        return result;
    }

    float3 G(Vertex v0, Vertex v1) {
        float3 d = v0.sd.posW - v1.sd.posW;
        float g = 1 / dot(d, d);
        //d *= sqrt(g);
        //g *= v0.type == VertexType::SurfaceOrMedium ? abs(dot(v0.sd.N, d)) : 1;
        //g *= v1.type == VertexType::SurfaceOrMedium ? abs(dot(v1.sd.N, d)) : 1;
        return g;
    }

    Ray getVisibiliyTestRay(Vertex v, Vertex sample) {
        PathVertex pVertexCameraSide = PathVertex(2, v.sd.posW, v.sd.N, v.sd.faceN);
        PathVertex pVertexLigthSide = PathVertex(2, sample.sd.posW, sample.sd.N, sample.sd.faceN);

        float3 cameraSideOrigin = pVertexCameraSide.getRayOrigin(sample.sd.posW - v.sd.posW);
        float3 lightSideOrigin = pVertexLigthSide.getRayOrigin(v.sd.posW - sample.sd.posW);

        float3 toLight = lightSideOrigin - cameraSideOrigin;
        float distSqr = max(dot(toLight, toLight), kMinLightDistSqr);
        float distance = sqrt(distSqr);
        float3 dir = toLight / distance;

        Ray ray = Ray(cameraSideOrigin, dir, 0, distance);
        return ray;
    }

    Ray getVisibiliyTestRay(Vertex v, VertexInfo lightInfo) {
        PathVertex pVertexCameraSide = PathVertex(2, v.sd.posW, v.sd.N, v.sd.faceN);
        PathVertex pVertexLigthSide = PathVertex(2, lightInfo.origin, lightInfo.dir, lightInfo.dir);

        float3 cameraSideOrigin = pVertexCameraSide.getRayOrigin(lightInfo.origin - v.sd.posW);
        float3 lightSideOrigin = pVertexLigthSide.getRayOrigin(v.sd.posW - lightInfo.origin);

        float3 toLight = lightSideOrigin - cameraSideOrigin;
        float distSqr = max(dot(toLight, toLight), kMinLightDistSqr);
        float distance = sqrt(distSqr);
        float3 dir = toLight / distance;

        Ray ray = Ray(cameraSideOrigin, dir, 0, distance);
        return ray;
    }

    inline bool firstChildWeight(Vertex v, inout float prob0, inout PathState path, int child0, int child1, IBSDF bsdf, bool upper, bool lower)
    {
        Node c0 = nodes[child0];
        Node c1 = nodes[child1];
        // TODO: avoid "if"
        float c0_intensity = c0.betaSum;
        float c1_intensity = c1.betaSum;

        if (c0_intensity == 0)
        {
            if (c1_intensity == 0)
                return false;
            prob0 = 0;
            return true;
        }
        else if (c1_intensity == 0)
        {
            prob0 = 1;
            return true;
        }

        float3 c0_boundMin = c0.boundMin;
        float3 c0_boundMax = c0.boundMax;
        float3 c1_boundMin = c1.boundMin;
        float3 c1_boundMax = c1.boundMax;

        float3 p = v.sd.posW;
/*
        float3 center0 = (c0_boundMin + c0_boundMax) / 2;
        float3 center1 = (c1_boundMin + c1_boundMax) / 2;
        float3 wo0 = center0 - p;
        float3 wo1 = center1 - p;
        float dist0 = length(wo0);
        float r0 = length(c0_boundMin - c0_boundMax) / 2;
        float dist1 = length(wo1);
        float r1 = length(c1_boundMin - c1_boundMax) / 2;
        // Compute the weights
        bool isTheVertexInTheBox = (all(p >= c0_boundMin) && all(p <= c0_boundMax)) || (all(p >= c1_boundMin) && all(p <= c1_boundMax));
        bool isSpreadAngleSmallEnough = (r0 < 0.8 * dist0) && (r1 < 0.8 * dist1);

        if ((!isTheVertexInTheBox) && isSpreadAngleSmallEnough && (upper || lower)) {
            c0_intensity *= getIntensity(bsdf.eval(v.sd, wo0, path.sg));
            c1_intensity *= getIntensity(bsdf.eval(v.sd, wo1, path.sg));
        }
*/

        float intensGeom0 = c0_intensity;
        float intensGeom1 = c1_intensity;

        //prob0 = c0_intensity / (c1_intensity + c0_intensity);

        float l2_min0;
        float l2_min1;
        l2_min0 = SquaredDistanceToClosestPoint(p, c0_boundMin, c0_boundMax);
        l2_min1 = SquaredDistanceToClosestPoint(p, c1_boundMin, c1_boundMax);

        float l2_max0 = SquaredDistanceToFarthestPoint(p, c0_boundMin, c0_boundMax);
        float l2_max1 = SquaredDistanceToFarthestPoint(p, c1_boundMin, c1_boundMax);
        float w_max0 = l2_min0 == 0 && l2_min1 == 0 ? intensGeom0 / (intensGeom0 + intensGeom1) : normalizedWeights(l2_min0, l2_min1, intensGeom0, intensGeom1);
        float w_min0 = normalizedWeights(l2_max0, l2_max1, intensGeom0, intensGeom1);
        prob0 = 0.5 * (w_max0 + w_min0);

        return true;
    }

    inline bool selectVertex(inout uint nid, inout float r, inout double nprob, inout PathState path, Vertex v, IBSDF bsdf, bool upper, bool lower)
    {
        bool deadBranch = false;
        while (nid < leafNodeStart)
        {
            uint c0_id = nid << 1;
            uint c1_id = c0_id + 1;
            float prob0;
            bool x = firstChildWeight(v, prob0, path, c0_id, c1_id, bsdf, upper, lower);
            if (x)
            {
                nid = (r < prob0) ? c0_id : c1_id;
                nprob *= (r < prob0) ? prob0 : (1 - prob0);
                r = (r < prob0) ? r / prob0 : (r - prob0) / (1 - prob0);
            }
            else
            {
                deadBranch = true;
                break;
            }
        }

        return deadBranch;
    }

    void ConnectBDPT<VisibilityQuery : IVisibilityQuery>(inout PathState path, const Vertex v, inout VisibilityQuery vq, in float de, in bool sampleUpper, in bool sampleLower){
        // select several valid vertex from the light path

        const bool isPrimaryHit = path.getVertexIndex() == 1;
        const HitType hitType = path.hit.getType();
        const bool isTriangleHit = hitType == HitType::Triangle;

        let lod = createTextureSampler(path, false, isTriangleHit);

        const IBSDF bsdf = gScene.materials.getBSDF(v.sd, lod);

        double nprob = 1.0;
        float r = sampleNext1D(path.sg);
        uint nid = 1;
        //uint nodeId;
        bool deadBranch = selectVertex(nid, r, nprob, path, v, bsdf, sampleUpper, sampleLower);

        float one_over_prob = (nprob == 0.0) ? 0.0 : 1.0 / float(nprob);

        if (deadBranch) return;
        
        if (params.hasFlag(BDPTFlags::t1)) {
            
            nid = min(uint(counter * sampleNext1D(path.sg)), counter - 1) + leafNodeStart;
            one_over_prob = counter;
        }
        
        Node sampleNode = nodes[nid];
        uint sampleIndex = sampleNode.ID;

        VertexInfo sampleInfo = LightPathsVertexsBuffer[sampleIndex];

        // one_over_prob *= sampleNode.betaSum / getIntensity(sampleInfo.beta);
        //one_over_prob /= sampleNode.pdf;

        float3 f;
        Ray ray;

        if (sampleInfo.isLight()) {
            // light
            float3 toSample = sampleInfo.origin - v.sd.posW;
            float dist2 = dot(toSample, toSample);
            toSample /= sqrt(dist2);

            const uint lobes = bsdf.getLobes(v.sd);

            float g = dot(normalize(sampleInfo.dir), -toSample) / dist2;

            ray = getVisibiliyTestRay(v, sampleInfo);

            f = bsdf.eval(v.sd, toSample, path.sg) * g * sampleInfo.beta * BDPTMIS(path, sampleInfo, v, de);
        }
        else
        {
            // others
            Vertex sample = LightPathsVertexsBuffer[sampleIndex].unpack(false);
/*
            uint start = max(nid - 5, leafNodeStart);
            uint end = min(nid + 5, 2 * leafNodeStart - 1);
            uint filterCounter = 0;
            float3 accumulativeBeta = 0;

            for (; start + filterCounter <= end; ++filterCounter) {
                uint neignborNodeId = start + filterCounter;
                uint neighborIndex = nodes[neignborNodeId].ID;
                accumulativeBeta += LightPathsVertexsBuffer[neighborIndex].beta;
            }
*/
            // float3 result_q = RISWeight(v, sample, path, false);
            ray = getVisibiliyTestRay(v, sample);

            //float3 meanBeta = accumulativeBeta;

            f = RISWeight(v, sample, path, sample.beta) * BDPTMIS(path, sampleIndex, v, de);
        }

        bool visible =vq.traceVisibilityRay(ray);

        float inv_M = 1.0 / float(kLightPassHeight * kLightPassWidth);

        path.L += (params.hasFlag(BDPTFlags::s2) && !deadBranch && visible) ? inv_M * v.beta * f * one_over_prob : 0;
        //path.L += (params.hasFlag(BDPTFlags::s2) && !deadBranch && visible) ? v.beta * f : 0;
        //path.L += inv_M * f * one_over_prob;
    }

    float BDPTMIS(PathState path, uint sampleIndex, Vertex v, float de) {
        const bool isPrimaryHit = path.getVertexIndex() == 1;
        const HitType hitType = path.hit.getType();
        const bool isTriangleHit = hitType == HitType::Triangle;

        let lod = createTextureSampler(path, isPrimaryHit, isTriangleHit);

        VertexInfo sampleInfo = LightPathsVertexsBuffer[sampleIndex];
        float dl = sampleInfo.de;
        Vertex qs = sampleInfo.unpack(false);
        Vertex pt = v;

        float3 wo = normalize(pt.sd.posW - qs.sd.posW);
        const IBSDF bsdf1 = gScene.materials.getBSDF(qs.sd, lod);
        float pdfRev = qs.ConvertDensity(bsdf1.evalPdf(qs.sd, wo), pt);

        const IBSDF bsdf2 = gScene.materials.getBSDF(pt.sd, lod);
        float pdfFwd = pt.ConvertDensity(bsdf2.evalPdf(pt.sd, -wo), qs);

        float N = isPrimaryHit ? 2.0 : 1.0;

        return 1.f / (de * pdfRev + N + dl * pdfFwd);
    }

    float BDPTMIS(PathState path, VertexInfo lightVertexInfo, Vertex v, float de) {
        const bool isPrimaryHit = path.getVertexIndex() == 1;
        const HitType hitType = path.hit.getType();
        const bool isTriangleHit = hitType == HitType::Triangle;

        let lod = createTextureSampler(path, isPrimaryHit, isTriangleHit);

        Vertex pt = v;

        float dl = lightVertexInfo.de;
        float3 wi = normalize(pt.sd.posW - lightVertexInfo.origin);

        float pdfRev;
        if (dot(wi, wi) == 0) {
            pdfRev = 0;
        }
        else {
            float invDist2 = 1 / dot(wi, wi);
            pdfRev = dot(wi, lightVertexInfo.dir) * M_1_PI * abs(dot(pt.sd.N, wi * sqrt(invDist2))) * invDist2;
        }

        const IBSDF bsdf2 = gScene.materials.getBSDF(pt.sd, lod);
        float pdfFwd = pt.ConvertDensity(bsdf2.evalPdf(pt.sd, -wi), lightVertexInfo.origin, lightVertexInfo.dir);

        return 1.f / (de * pdfRev + 2 + dl * pdfFwd);
    }

    float BDPTMIS(PathState path, Vertex lightVertex, Vertex v, float de) {
        const bool isPrimaryHit = path.getVertexIndex() == 1;
        const HitType hitType = path.hit.getType();
        const bool isTriangleHit = hitType == HitType::Triangle;

        let lod = createTextureSampler(path, isPrimaryHit, isTriangleHit);

        Vertex qs = lightVertex;
        Vertex pt = v;

        float dl = 1.0 / qs.pdfFwd;
        float3 wi = normalize(pt.sd.posW - qs.sd.posW);
        float pdfRev = qs.ConvertDensity(dot(wi, qs.sd.N) * M_1_PI, pt);

        const IBSDF bsdf2 = gScene.materials.getBSDF(pt.sd, lod);
        float pdfFwd = pt.ConvertDensity(bsdf2.evalPdf(pt.sd, -wi), qs);

        return 1.f / (de * pdfRev + 2 + dl * pdfFwd);
    }

    float BDPTMIS(PathState path, TriangleLightHit hit, float de) {
        bool upperHemisphere = path.isLightSampledUpper() && !path.isLightSampledLower();
        float area = gScene.lightCollection.getTriangle(hit.triangleIndex).area;
        float pdfRev = getEmissiveSelectionProbability() * emissiveSampler.evalTriangleSelectionPdf(path.origin, path.normal, upperHemisphere, hit.triangleIndex) / area;

        return 1.f / (de * pdfRev + 1);
    }

    inline float remap0(float f) {
        return f != 0.0 ? f : 1.0;
    }

    /** Handle the case when a scatter ray misses the scene.
        \param[in,out] path The path state.
    */
    void handleMiss(inout PathState path)
    {
        // Upon miss:
        // - Compute MIS weight if previous path vertex sampled a light
        // - Evaluate environment map
        // - Write guiding data
        // - Terminate the path

        // Check if the scatter event is samplable by the light sampling technique.
        const bool isLightSamplable = path.isLightSamplable();

        // Add env radiance.
        bool computeEnv = kUseEnvLight && (!kUseNEE || kUseMIS || !path.isLightSampled() || !isLightSamplable);

        // With RTXDI enabled, we sample the full direct illumination contribution on the primary hit.
        // Skip any additional contribution on the secondary hit unless it comes from a scatter event
        // that RTXDI cannot handle, such as transmission, delta or volume scattering events.
        if (kUseRTXDI && path.getVertexIndex() == 2 && !path.isTransmission() && !path.isDelta()) computeEnv = false;

        float3 emitterRadiance = 0.f;

        if (computeEnv)
        {
            logPathVertex();

            float misWeight = 1.f;
            if (kUseNEE && kUseMIS && path.isLightSampled() && isLightSamplable)
            {
                // If NEE and MIS are enabled, and we've already sampled the env map,
                // then we need to evaluate the MIS weight here to account for the remaining contribution.

                // Evaluate PDF, had it been generated with light sampling.
                float lightPdf = getEnvMapSelectionProbability() * envMapSampler.evalPdf(path.dir);

                // Compute MIS weight by combining this with BSDF sampling.
                // Note we can assume path.pdf > 0.f since we shouldn't have got here otherwise.
                misWeight = evalMIS(1, path.pdf, 1, lightPdf);
            }

            float3 Le = envMapSampler.eval(path.dir);
            emitterRadiance = misWeight * Le;
            addToPathContribution(path, emitterRadiance);

        }

        path.terminate();
    }

    void BDPTHandleMiss(inout PathState path) {
        path.terminate();
    }

    /** Write path contribution to output buffer.
    */
    void writeOutput(const PathState path)
    {
        assert(!any(isnan(path.L)));

        // Log path length.
        logPathLength(getTerminatedPathLength(path));

        const uint2 pixel = path.getPixel();
        const uint outIdx = params.getSampleOffset(pixel, sampleOffset) + path.getSampleIdx();

        if (kSamplesPerPixel == 1)
        {
            // Write color directly to frame buffer.
            outputColor[pixel] = float4(path.L, 1.f);
        }
        else
        {
            // Write color to per-sample buffer.
            sampleColor[outIdx].set(path.L);
        }
    }

    void BDPTOutput(const PathState path) {
        logPathLength(getTerminatedPathLength(path));

        const uint2 pixel = path.getPixel();
        outputColor[pixel] = float4(path.L, 1.f);
    }
};
