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
#include "BDPTUtils.slangh"
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

struct ExtraPayload
{
    float d;
    float p;
    //uint prevIndex;
    float pPrim = 0.f;
    uint RISSampleIndex;
}

struct RISSampleInfo
{
    float3 pos;
    float dl;
    float3 radiance;
    float p;
    float2 normal;
}


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
    float globalRadius;

/*
    float gHashScaleFactor; // Scale factor calculated from photon radius
    uint gHashSize;         // Size of the hash
    uint gYExtend;
    float gGlobalRadius;
*/
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


    Texture2D<float4> InputFluxAndNumber;
    Texture2D<float4> InputNormalAndRadii;
    Texture2D<float4> InputPosAndNewRadii;
    Texture2D<uint> InputIteration;
    Texture2D<float2> motionVec;

    // sample info
    Texture2D<float4> samplePositionI;   // xyz := pos, w := dL
    Texture2D<float4> hitPointPositionI; // xyz := pos, w := p_toLeft
    Texture2D<float4> normalI;           // xyz := sample normal
    Texture2D<float4> radianceI;         // rgb := radiance, w := radii


    // reservoir
    Texture2D<uint4> ReservoirI; // x := sample weight(float), y := W (float), z := sample count, w := index

    // subspace weight
    Texture2D<uint> prevSubspaceWeight;
    Texture2D<float> prefixSum;
    Texture1D<float> weightSum;
    RWTexture2D<uint> SubspaceWeight;
    RWTexture2D<uint> SubspaceCount; 
    Texture1D<uint> countSum;
    Texture2D<uint> prefixSumOfCount;

    RWTexture2D<uint> SubspaceSecondaryMoment;
    Texture2D<float> prefixSumOfSecondaryMoment;
    Texture1D<float> secondaryMomentSum;

    Texture1D<float> maxVariance;
 
    // Outputs
    RWStructuredBuffer<ColorType> sampleColor;            ///< Output per-sample color if kSamplesPerPixel != 1.

    RWTexture2D<float4> outputColor;                      ///< Output color buffer if kSamplesPerPixel == 1.

    RWStructuredBuffer<VertexInfo> LightPathsVertexsBuffer; ///< Output the Vertex of all the light paths.
    RWStructuredBuffer<uint2> LightPathsIndexBuffer;
    RWStructuredBuffer<float4> LightPathsVertexsPositionBuffer;

    RWStructuredBuffer<AABB> hitPointAABB;

    // sample info
    
    RWTexture2D<float4> hitPointPositionO; // xyz := pos, w := p_toLeft
    RWTexture2D<float4> normalO;           // xy := sample normal, zw := hit point normal
    //RWTexture2D<float4> radianceO;         // rgb := radiance, w := radii

    // reservoir
    RWTexture2D<uint2> Reservoir;       // x := W (float), y := sample count(uint)
    RWTexture2D<float4> samplePosition; // xyz := pos, w := dL
    RWTexture2D<float4> hitPointPosition; // xyz := pos, w := p_toLeft
    RWTexture2D<float4> normal;          // xy := sample normal, zw := hit point normal
    RWTexture2D<float4> radiance;        // rgb := radiance
    RWTexture2D<uint> ReservoirSignal;

    RWTexture3D<uint4> pathPos;   // xyz := pos, w := index
    RWTexture3D<float4> pathRadiance; // xyz := radiance, w := de;
    RWTexture3D<uint4> pathHitInfo;

    /*
        RWTexture2D<PackedHitInfo> gatherPointVbuffer;
        RWTexture2D<float4> gatherPointThp;
        RWTexture2D<float4> gatherPointDir;
        RWTexture2D<uint> hashBuffer;
    */
    RWTexture2D<float4> OutputFluxAndNumber;
    RWTexture2D<float4> OutputNormalAndRadii;
    RWTexture2D<float4> OutputPosAndNewRadii;
    RWTexture2D<uint> OutputIteration;

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
        //path.pdf = 1.f;

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
            path.setFlag(PathFlags::isFirstNonDelta, false);
        }
    }

    /** Generate the first vertex of the light subpath
        \param[in] pathID Path ID which encodes pixel and sample index.
        \param[in] line Pixel per line.
        \param[out] path Path state for the primary hit.
    */
    void generateLightPath(const uint pathID, out PathState path, out ExtraPayload exPayload)
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
        vInfo.pe = 0;
        vInfo.prevIndex = 0;
        vInfo.beta *= dirPdf;
        exPayload.d = vInfo.de;
        exPayload.p = 0;
        
        //LightPathsIndexBuffer[LightPathsIndexBuffer.IncrementCounter()] = offset;
        //tempVertexBuffer[0] = lv;
        //LightPathsVertexsBuffer[(pixel.y * line_number + pixel.x) * kMaxSurfaceBounces] = lv;

        path.origin = lv.sd.posW;
        path.normal = lv.sd.N;
        path.dir = ray_dir;

        path.thp = lv.beta;
        path.pdf = dirPdf;
/*

        uint vertexID = LightPathsIndexBuffer.IncrementCounter();
        uint mortonCode = GenMortonCode(path.origin);
        LightPathsIndexBuffer[vertexID] = uint2(vertexID, mortonCode);
        LightPathsVertexsBuffer[vertexID] = vInfo;
        LightPathsVertexsPositionBuffer[vertexID] = float4(path.origin, getIntensity(vInfo.beta));

        exPayload.prevIndex = vertexID;
*/
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

    void lightPathHandleHit<VisibilityQuery : IVisibilityQuery>(inout PathState path, inout VisibilityQuery vq, inout ExtraPayload exPayload) {
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
        float N = 1;
        pdfRev = remap0(vertex.ConvertRevDensity(path, pdfRev));

        VertexInfo vInfo = VertexInfo::create(path, vertex);
        exPayload.d = (N + pdfRev * exPayload.d) / remap0(vertex.pdfFwd);
        vInfo.de = exPayload.d;
        exPayload.p = pdfRev * (1 + exPayload.p) / remap0(vertex.pdfFwd);
        vInfo.pe = exPayload.p;
        
        //vInfo.prevIndex = exPayload.prevIndex;
        // LightPathsVertexsBuffer[offset] = vInfo;
        bool isDiffuse = bs.isLobe(LobeType::Diffuse) || (bs.isLobe(LobeType::Reflection) && (bsdfProperties.roughness > 0.5));

        if (path.getVertexIndex() == 1) path.setFlag(PathFlags::isFirstNonDelta, !isDiffuse);
        else {
            bool isFirstNonDelta = path.hasFlag(PathFlags::isFirstNonDelta);
            path.setFlag(PathFlags::isFirstNonDelta, isFirstNonDelta && !isDiffuse);
        }

        if (!hasDelta) {
            uint vertexID = LightPathsIndexBuffer.IncrementCounter();
            uint mortonCode = GenMortonCode(vertex.sd.posW);

            vInfo.setFlag(!path.hasFlag(PathFlags::isFirstNonDelta), VertexFlag::diffuseHit);
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

    void CameraPathHandleHit<VisibilityQuery : IVisibilityQuery>(inout PathState path, inout VisibilityQuery vq, inout ExtraPayload exPayload)
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
        //bool isFirstNonDelta = (!exPayload.isFirstNonDelta) && (!hasDelta);
        //exPayload.isFirstNonDelta = isFirstNonDelta;
        vertex.delta = hasDelta;
        float N = 1.f;

        float convertRevPdf = remap0(vertex.ConvertRevDensity(path, pdfRev));

        bool sampleUpperHemisphere = isCurveHit || isCurvePolyTubeHit || ((lobes & (uint)LobeType::NonDeltaReflection) != 0);
        if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume()) sampleUpperHemisphere = false;
        bool sampleLowerHemisphere = isCurveHit || isCurvePolyTubeHit || ((lobes & (uint)LobeType::NonDeltaTransmission) != 0);

        bool onlyUsePrimary = true;//params.hasFlag(BDPTFlags::t1); // ? !exPayload.isFirstNonDelta : true;
        bool isDiffuseHit = !hasDelta; // bs.isLobe(LobeType::Diffuse) || (bs.isLobe(LobeType::Reflection) && (bsdfProperties.roughness > 0.7));
        bool useMerge = params.hasFlag(BDPTFlags::useVertexMerge);

        bool isFirstNonDelta = isDiffuseHit && !path.hasFlag(PathFlags::isFirstNonDelta);
        if (bs.isLobe(LobeType::Transmission)) path.setFlag(PathFlags::passedTransmission, true);
        // bool isFirstNonDelta = !hasDelta && !exPayload.isFirstNonDelta;
        if (isFirstNonDelta) path.setFlag(PathFlags::isFirstNonDelta, true);

        float radius = globalRadius;
        bool isCausticPath = false;

        // vertex merge

        if ((((isFirstNonDelta && onlyUsePrimary) || !onlyUsePrimary)) && useMerge) {

            if (isFirstNonDelta && onlyUsePrimary) { // temporal ppm
                uint2 pixel = path.getPixel();
                if (params.frameCount == 0 || params.hasFlag(BDPTFlags::t1)) { // init frame
                    radius = globalRadius;
                    float3 avgCenter = 0;
                    uint num = 0;
                    float4 fluxAndNum = vertexMerge(path, vertex, exPayload.p, exPayload.d, radius, avgCenter, num, onlyUsePrimary);
                    isCausticPath = num != 0;
                    path.L += vertex.beta * fluxAndNum.xyz * M_1_PI / (radius * radius);
                    OutputFluxAndNumber[pixel] = fluxAndNum;
                    OutputNormalAndRadii[pixel] = float4(vertex.sd.faceN, radius);
                    OutputPosAndNewRadii[pixel] = float4(vertex.sd.posW, radius);
                    OutputIteration[pixel] = 1;
                    
                }
                else { // temporal reuse
                    uint2 imageDim = uint2(1920, 1080);
                    float2 mv = motionVec[pixel] * float2(imageDim);
                    int2 prevPixel = int2(float2(pixel) + 0.5f + mv);
                    if (any(prevPixel < int2(1, 1)) || any(prevPixel > imageDim - int2(1, 1))) {
                        float3 avgCenter = 0;
                        radius = globalRadius;
                        uint num = 0;
                        float4 fluxAndNum = vertexMerge(path, vertex, exPayload.p, exPayload.d, radius, avgCenter, num, onlyUsePrimary);
                        isCausticPath = num != 0;
                        path.L += vertex.beta * fluxAndNum.xyz * M_1_PI / (radius * radius);
                        OutputFluxAndNumber[pixel] = fluxAndNum;
                        OutputNormalAndRadii[pixel] = float4(vertex.sd.faceN, radius);
                        OutputPosAndNewRadii[pixel] = float4(vertex.sd.posW, radius);
                        OutputIteration[pixel] = 1;
                    }
                    else {
                        float3 prevNormal = InputNormalAndRadii[prevPixel].xyz;
                        float prevRadius = InputNormalAndRadii[prevPixel].w;
                        float3 prevPos = InputPosAndNewRadii[prevPixel].xyz;
                        float newRadius = InputPosAndNewRadii[prevPixel].w;
                        float3 toPrevPos = prevPos - vertex.sd.posW;
                        float dist = length(toPrevPos);

                        uint2 dxPixel = uint2((prevPixel.x + 1 < imageDim.x) ? prevPixel.x + 1 : prevPixel.x - 1, prevPixel.y);
                        uint2 dyPixel = uint2(prevPixel.x, (prevPixel.y + 1 < imageDim.y) ? prevPixel.y + 1 : prevPixel.y - 1);
                        float dxPos = distance(prevPos, InputPosAndNewRadii[dxPixel].xyz);
                        float dyPos = distance(prevPos, InputPosAndNewRadii[dyPixel].xyz);
                        float pFwidth = dxPos + dyPos;

                        //float gamma = commonArea(prevRadius, newRadius, dist) * M_1_PI / (prevRadius * prevRadius);

                        if ((dist < prevRadius)) {                          // valid
                            radius = newRadius;
                            float3 avgCenter = 0;
                            uint num = 0;
                            float4 fluxAndNum = vertexMerge(path, vertex, exPayload.p, exPayload.d, radius, avgCenter, num, onlyUsePrimary);
                            isCausticPath = num != 0;
                            float3 prevTau = InputFluxAndNumber[prevPixel].xyz;
                            float prevN = InputFluxAndNumber[prevPixel].w;
                            float3 phi = fluxAndNum.xyz;
                            float M = fluxAndNum.w;
                            if (M == 0) {
                                if (InputIteration[prevPixel] == 0) {
                                    OutputFluxAndNumber[pixel] = fluxAndNum;
                                    OutputNormalAndRadii[pixel] = float4(vertex.sd.faceN, globalRadius);
                                    OutputPosAndNewRadii[pixel] = float4(vertex.sd.posW, globalRadius);
                                    OutputIteration[pixel] = 1;
                                }
                                else {
                                    path.L += vertex.beta * prevTau * M_1_PI / (radius * radius * InputIteration[prevPixel]);
                                    OutputFluxAndNumber[pixel] = InputFluxAndNumber[prevPixel];
                                    OutputNormalAndRadii[pixel] = float4(vertex.sd.faceN, prevRadius);
                                    OutputPosAndNewRadii[pixel] = float4(vertex.sd.posW, prevRadius);
                                    OutputIteration[pixel] = InputIteration[prevPixel];
                                }
                            }
                            else {
                                float gamma = commonArea(prevRadius, newRadius, dist) * M_1_PI / (prevRadius * prevRadius);
                                float alpha = 0.667f;
                                float nextN = prevN + alpha * M;
                                float reduction = nextN / (prevN + M);
                                uint iteration = InputIteration[prevPixel] + (M > 0 ? 1 : 0);
                                // iteration = min(iteration, 20);
                                float3 Tau = (prevTau + phi) * reduction;
                                // iteration = min(iteration, 20);
                                float nextRadius = newRadius * sqrt(reduction);

                                path.L += vertex.beta * Tau * M_1_PI / (nextRadius * nextRadius * iteration);
                                //path.L = float3(getIntensity(prevTau), 0.f, getIntensity(phi));
                                // path.L += vertex.beta * phi * M_1_PI / (radius * radius);//(nextRadius * nextRadius * iteration);
                                // path.L += 1;
                                OutputFluxAndNumber[pixel] = float4(Tau, nextN);
                                OutputNormalAndRadii[pixel] = float4(vertex.sd.faceN, prevRadius);
                                OutputPosAndNewRadii[pixel] = float4(prevPos, nextRadius);
                                OutputIteration[pixel] = iteration;
                            }
                            // type1
                            // float gamma = commonArea(prevRadius, newRadius, dist) * M_1_PI / (prevRadius * prevRadius);
                            /*
                            float gamma = Gaussian(dist, radius / 3);
                            float alpha = 0.9f;
                            uint totalNum = InputIteration[prevPixel];
                            float reduction = (gamma * prevN + alpha * M) / (gamma * prevN + num);
                            float3 Tau = (prevTau / prevN * gamma + phi);
                            // float3 Tau = phi * M_1_PI / (radius * radius);
                            float nextN = gamma * totalNum + M;
                            avgCenter = (avgCenter * M + (prevNormal + prevPos - vertex.sd.posW) * gamma * totalNum) / nextN;
                            float nextRadius = radius * sqrt(reduction);
                            path.L += vertex.beta * Tau;
                            */
                            //type2
                            

                        }
                        else {
                            radius = globalRadius;
                            float3 avgCenter = 0;
                            uint num = 0;
                            float4 fluxAndNum = vertexMerge(path, vertex, exPayload.p, exPayload.d, radius, avgCenter, num, onlyUsePrimary);
                            isCausticPath = num != 0;
                            path.L += vertex.beta * fluxAndNum.xyz * M_1_PI / (radius * radius);
                            OutputFluxAndNumber[pixel] = fluxAndNum;
                            OutputNormalAndRadii[pixel] = float4(vertex.sd.faceN, radius);
                            OutputPosAndNewRadii[pixel] = float4(vertex.sd.posW, radius);
                            OutputIteration[pixel] = 1;
                        }
                    }
                }
                exPayload.pPrim = 1.f;    
            }
            else {
                radius = globalRadius;
                float3 avgCenter = 0;
                uint num = 0;
                float4 fluxAndNum = vertexMerge(path, vertex, exPayload.p, exPayload.d, radius, avgCenter, num, onlyUsePrimary);
                path.L += vertex.beta * fluxAndNum.xyz * M_1_PI / (radius * radius);
            }
              
            
        }
        

        // s >= 1
        if (!hasDelta) {
            RISSampleInfo risInfo = {};
            bool validSample = ConnectBDPT(path, vertex, vq, exPayload.d, onlyUsePrimary ? exPayload.pPrim : exPayload.p, radius, isFirstNonDelta, onlyUsePrimary, isCausticPath, risInfo);
            if (validSample) {
                //uint aabbIndex = hitPointAABB.IncrementCounter();
                // TODO: move outside
                //hitPointAABB[aabbIndex] = AABB(vertex.sd.posW - globalRadius, vertex.sd.posW + globalRadius);
                //uint2 texIndex = uint2(aabbIndex % 2048, aabbIndex / 2048);
                //if (params.frameCount == 0) {
                    
                //}
            }
        }

        if (!hasDelta)
        { // s == 1
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

                ShadingData sd1 = sd;
                float3 wo = sd1.V;
                sd1.V = -wi;
                const IBSDF bsdf1 = gScene.materials.getBSDF(sd1, lod);
                float pdfRev1 = vertex.ConvertRevDensity(path, bsdf1.evalPdf(sd1, wo));
                float de1 = (1 + remap0(pdfRev1) * exPayload.d) / remap0(vertex.pdfFwd);

                if (params.hasFlag(BDPTFlags::s1) && any(Lr > 0.f))
                {
                    // Ray ray = ls.getVisibilityRay();
                    logTraceRay(PixelStatsRayType::Visibility);
                    bool visible = vq.traceVisibilityRay(ray);

                    //float radius = 0.005 * length(dimension);
                    float pdfToX = dirPdf;
                    pdfToX = dot(-dir, vertex.sd.N) * pdfToX / distSqr;
                    float OmegaMC = 0;
                    if (onlyUsePrimary) {
                        float pe1 = isFirstNonDelta ? 1.f : exPayload.pPrim * remap0(pdfRev1) / remap0(vertex.pdfFwd);
                        OmegaMC = M_PI * radius * radius * pe1 * pdfToX;
                    }
                    else {
                        float pe1 = (1 + exPayload.p) * remap0(pdfRev1) / remap0(vertex.pdfFwd);

                        float pdfFromX = bsdf.evalPdf(vertex.sd, dir) * abs(dot(dir, lightVertex.sd.N)) / distSqr;
                        float pl = remap0(pdfFromX) / remap0(pdfToX);
                        float OmegaMC = (pe1 + 1.f + pl) * pdfToX * M_PI * radius * radius;
                    }

                    float OmegaCC = 1 / BDPTMIS(path, lightVertex, vertex, de1);
                    if (!params.hasFlag(BDPTFlags::useVertexMerge)) OmegaMC = 0;
                    //OmegaMC = 0;
                    //float OmegaMC = (pe1 + 1.f + pl) * pdfToX * M_PI * radius * radius;// * kLightPassHeight * kLightPassWidth;

                    //if (params.hasFlag(BDPTFlags::t1)) OmegaMC = 0;
                    // visible = true;
                    if (visible) path.L += Lr / (OmegaCC + OmegaMC);
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

        float cosTheta0 = abs(dot(normalize(path.dir), hit.normalW));
        float pdfRev0 = vertex.ConvertRevDensity(path, cosTheta0 * M_1_PI);
        float de0 = (1 + pdfRev0 * exPayload.d) / remap0(vertex.pdfFwd);

        //bool sampleLowerHemisphere = isCurveHit || isCurvePolyTubeHit || ((lobes & (uint)LobeType::NonDeltaTransmission) != 0);
        if (params.hasFlag(BDPTFlags::s0) && computeEmissive && any(bsdfProperties.emission > 0.f)) {
            path.L += vertex.beta * bsdfProperties.emission * BDPTMIS(path, hit, de0);
        }

        exPayload.d = (N + exPayload.d * convertRevPdf) / remap0(vertex.pdfFwd);
        exPayload.p = convertRevPdf * (1 + exPayload.p) / remap0(vertex.pdfFwd);
        exPayload.pPrim = isFirstNonDelta ? 1.f : exPayload.pPrim * convertRevPdf / remap0(vertex.pdfFwd);

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
    void getArrayOfCells(float3 position, out int3 outCells[8])
    {
        float3 cell = (position.xyz * gHashScaleFactor);
        float radInScale = gGlobalRadius * gHashScaleFactor;
        float3 cellFloor = floor(cell);
        float3 relCellPos = abs(cell - cellFloor);
        // Get offset direction
        int3 offsetCell;
        offsetCell.x = relCellPos.x < 0.5 ? -1 : 1;
        offsetCell.y = relCellPos.y < 0.5 ? -1 : 1;
        offsetCell.z = relCellPos.z < 0.5 ? -1 : 1;

        // Fill the cell array

        outCells[0] = int3(cellFloor);
        outCells[1] = int3(floor(cell + radInScale * float3(offsetCell.x, 0, 0)));
        outCells[2] = int3(floor(cell + radInScale * float3(offsetCell.x, offsetCell.y, 0)));
        outCells[3] = int3(floor(cell + radInScale * float3(offsetCell.x, offsetCell.y, offsetCell.z)));
        outCells[4] = int3(floor(cell + radInScale * float3(offsetCell.x, 0, offsetCell.z)));
        outCells[5] = int3(floor(cell + radInScale * float3(0, offsetCell.y, 0)));
        outCells[6] = int3(floor(cell + radInScale * float3(0, offsetCell.y, offsetCell.z)));
        outCells[7] = int3(floor(cell + radInScale * float3(0, 0, offsetCell.z)));
    }
    */
    inline float commonArea(float r1, float r2, float d) {
        if (d >= r1 + r2) return 0;
        if (r1 > r2) {
            float tmp = r1;
            r1 = r2;
            r2 = tmp;
        }
        if (r2 - r1 >= d) return M_PI * r1 * r1;
        float ang1 = acos((r1 * r1 + d * d - r2 * r2) / (2 * r1 * d));
        float ang2 = acos((r2 * r2 + d * d - r1 * r1) / (2 * r2 * d));
        return ang1 * r1 * r1 + ang2 * r2 * r2 - r1 * d * sin(ang1);
    }

    float Gaussian(float r, float sigma) {
        //float M_1_SQRT2PI = 1.772453850905516f;
        return exp(-r * r * 0.5 / (sigma * sigma)) * M_1_2PI / (sigma * sigma);
    }


    float4 vertexMerge(inout PathState path, const Vertex v, in float pe, in float de, in float radius, out float3 avgCenter, out uint num, bool onlyUsePrimary) {
        const bool isPrimaryHit = path.getVertexIndex() == 1;
        const HitType hitType = path.hit.getType();
        const bool isTriangleHit = hitType == HitType::Triangle;

        let lod = createTextureSampler(path, false, isTriangleHit);

        const IBSDF bsdf = gScene.materials.getBSDF(v.sd, lod);

        //float radius = 0.005 * length(dimension); // TODO: User-defined parameters

        // searchTheTree(1, v.sd.posW, v.sd.faceN, r, sampleNext1D(path.sg), ret);
        Stack stack = {};
        //Stack result = {};
        // Stack ret = {};
        stack.push(1);
        uint nid = 0;

        float3 f = 0;
        float3 center = 0;
        float weight = 0;
        uint stopLevel = 1 << 3;
        uint count = 0;
        //path.L += 1;
        while (stack.pop(nid)) {
            Node node = nodes[nid];

            if (isAABBIntersectSphere(node.boundMin, node.boundMax, v.sd.posW, v.sd.faceN, radius)) {
                if (nid >= leafNodeStart / stopLevel) { // is leaf
                    for (uint i = 0; i < stopLevel; ++i) {
                        Node n = nodes[nid * stopLevel + i];
                        if (n.ID == 0xffffffff) break;
                        float3 fromTarget = LightPathsVertexsPositionBuffer[n.ID].xyz - v.sd.posW;
                        float r = length(fromTarget);
                        float cosTheta = abs(dot(fromTarget, v.sd.faceN) / r);
                        if (r < radius && cosTheta < 0.02f) {
                            
                            VertexInfo vInfo = LightPathsVertexsBuffer[n.ID];
                            if (vInfo.isLight()/* || vInfo.isCausticHit()*/) continue;
                            count++;
                            float3 wo = normalize(-vInfo.dir);
                            float cosTheta1 = abs(dot(v.sd.N, wo));

                            float dl = vInfo.de;
                            ShadingData sd1 = v.sd;
                            float3 woRev = sd1.V;
                            sd1.V = wo;
                            float pdfRev1 = v.ConvertRevDensity(path, bsdf.evalPdf(sd1, woRev));
                            float de1 = (1 + pdfRev1 * de) / remap0(v.pdfFwd);
                            float3 toSample = vInfo.origin - v.sd.posW;
                            float dist2 = dot(toSample, toSample);
                            float pdfFwd = bsdf.evalPdf(v.sd, wo) * abs(dot(toSample / sqrt(dist2), vInfo.originN)) / dist2;

                            float omegaCM = (de1 * remap0(vInfo.pdfFwd) + 1 + dl * pdfFwd) * M_1_PI / (radius * radius * remap0(vInfo.pdfFwd));

                            float pe1 = pdfRev1 * (1 + pe) / remap0(v.pdfFwd);
                            float pl1 = pdfFwd * (1 + vInfo.pe) / remap0(vInfo.pdfFwd);
                            float omegaMM = (onlyUsePrimary ? 1.f : (pe1 + 1 + pl1)) * (kLightPassHeight * kLightPassWidth);
                            //if (params.hasFlag(BDPTFlags::t1)) omegaCM = 0;
                            float MISweight = 1.f / (omegaCM + omegaMM);

                            float w = Gaussian(r, radius / 3);
                            float relRadius = r / radius;
                            // w = 4.75f * (1.f - (1.f - exp(-4.5f * relRadius * relRadius)) / (1.f - exp(-4.5f)));
                            // w = 0.918 * (1.f - (1.f - exp(-1.953f * 0.5f * relRadius * relRadius)) / (1.f - exp(-1.953f)));
                            w = 1.f;//3 * (1 - relRadius);
                            f += bsdf.eval(v.sd, wo, path.sg) * vInfo.beta * MISweight * w;
                            center += fromTarget * w;
                            weight += w;
                        }
                    }
                }
                else {
                    stack.push(2 * nid + 1);
                    stack.push(2 * nid);
                }
            }
        }
        // if (count > 0) f /= count;
        //path.L += f * M_1_PI / (radius * radius);
        avgCenter = center / weight;
        num = count;
        return float4(f, count);
    }

    struct Stack {
        uint s[30];
        int top = 0;

        [mutating] bool push(uint a) {
            if (top >= 29) return false;
            top++;
            s[top] = a;
            return true;
        }

        [mutating] bool pop(out uint result) {
            result = 0;
            if (top <= 0) return false;
            result = s[top];
            top--;
            return true;
        }
    }

    bool searchTheTree(uint startId, float3 targePos, float3 targetN, float radius, float rnd, out Stack ret) {
        Stack stack = {};
        Stack result = {};
        //Stack ret = {};
        stack.push(startId);
        uint nid = 0;

        uint stopLevel = 1 << 3;
        uint count = 0;


        // TODO: time consuming
        while (stack.pop(nid)) {
            Node node = nodes[nid];
            
            if (isAABBIntersectSphere(node.boundMin, node.boundMax, targePos, targetN, radius)) {
                if (nid >= leafNodeStart / stopLevel) { // is leaf
                    result.push(nid);
                }
                else {
                    stack.push(2 * nid + 1);
                    stack.push(2 * nid);
                }
            }
        }
        uint offset = min(uint(stopLevel * rnd), stopLevel - 1);
        float maxR = 0;
        while (result.pop(nid)) {

            for (uint i = 0; i < stopLevel; ++i) {
                Node node = nodes[nid * stopLevel + i];
                if (node.ID == 0xffffffff) break;
                float3 fromTarget = LightPathsVertexsPositionBuffer[node.ID].xyz - targePos;
                float r = length(fromTarget);
                float cosTheta = abs(dot(fromTarget, targetN) / r);
                if (r <= radius && cosTheta < 0.05) {
                    count++;
                    ret.push(node.ID);
                }
            }
            
        }
        return count > 0;
    }

    bool isAABBIntersectSphere(float3 boundMin, float3 boundMax, float3 center, float3 targetN, float radius) {
        //return all(center >= boundMin) && all(center <= boundMin);

        float3 rMin = center - radius;
        float3 rMax = center + radius;
        float xOver = calOver(rMin.x, rMax.x, boundMin.x, boundMax.x);
        float yOver = calOver(rMin.y, rMax.y, boundMin.y, boundMax.y);
        float zOver = calOver(rMin.z, rMax.z, boundMin.z, boundMax.z);
        bool isIntersect = xOver > 0 && yOver > 0 && zOver > 0;
        if (!isIntersect) return false;
        int signCount = 0;
        float3 bMin = boundMin - center;
        float3 bMax = boundMax - center;
        float3 a = float3(bMin.x, bMin.y, bMax.z);
        float3 b = float3(bMin.x, bMax.y, bMin.z);
        float3 c = float3(bMax.x, bMin.y, bMin.z);
        float3 d = float3(bMin.x, bMax.y, bMax.z);
        float3 e = float3(bMax.x, bMin.y, bMax.z);
        float3 f = float3(bMax.x, bMax.y, bMin.z);
        signCount += sign(dot(a, targetN)) + sign(dot(b, targetN)) + sign(dot(c, targetN)) + sign(dot(d, targetN)) + sign(dot(e, targetN)) + sign(dot(f, targetN))
                    + sign(dot(bMin, targetN)) + sign(dot(bMax, targetN));
        if (signCount < 8 && signCount > -8) return true;
        return false;
    }

    float calOver(float aMin, float aMax, float bMin, float bMax) {
        float minMax = 0;
        float maxMin = 0;
        if (aMax < bMax) {
            minMax = aMax;
            maxMin = bMin;
        }
        else {
            minMax = bMax;
            maxMin = aMin;
        }
        return minMax - maxMin;
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

        float intensGeom0 = c0_intensity; // / getSolidAngle(v.sd.posW, c0_boundMin, c0_boundMax);
        float intensGeom1 = c1_intensity;// / getSolidAngle(v.sd.posW, c1_boundMin, c1_boundMax);

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

    float solidanglecount(float startline, float endline, float startrow, float endrow, float high) {
        float result = 0;
        if (high != 0)
            result = asin(sqrt(endline * endline * endrow * endrow / ((endline * endline + high * high) * (endrow * endrow + high * high))))
		        - asin(sqrt(startline * startline * endrow * endrow / ((startline * startline + high * high) * (endrow * endrow + high * high))))
		       - asin(sqrt(endline * endline * startrow * startrow / ((endline * endline + high * high) * (startrow * startrow + high * high))))
		        + asin(sqrt(startline * startline * startrow * startrow / ((startline * startline + high * high) * (startrow * startrow + high * high))));
        else
            result = 0;
        return result;
    }

    float getSolidAngle(float3 p, float3 boundMin, float3 boundMax) {
        float3 toMin = boundMin - p;
        float3 toMax = boundMax - p;
        float3 a = float3(toMin.x, toMin.y, toMax.z);
        float3 b = float3(toMin.x, toMax.y, toMin.z);
        float3 c = float3(toMax.x, toMin.y, toMin.z);
        float3 d = float3(toMin.x, toMax.y, toMax.z);
        float3 e = float3(toMax.x, toMin.y, toMax.z);
        float3 f = float3(toMax.x, toMax.y, toMin.z);
        if (all(toMax >= 0) && all(toMin <= 0)) {
            if (all(toMax > 0) && all(toMin < 0)) { 
                //in the box
                return M_4PI;
            }
            if (all(toMin == 0) || all(toMax == 0) || all(a == 0) || all(b == 0) || all(c == 0) || all(d == 0) || all(e == 0) || all(f == 0)) { 
                //at the vertexs of the box
                return M_PI_2;
            }
            if (any(toMax == 0) && any(toMin == 0)) {
                //at the edges of the box
                return M_PI;
            }
            //at the surfaces of the box
            return M_2PI;
        }

        
        float result = 0;
        result += toMin.x > 0 ? solidanglecount(toMin.y, toMax.y, toMin.z, toMax.z, toMin.x) : 0;
        result += toMin.y > 0 ? solidanglecount(toMin.x, toMax.x, toMin.z, toMax.z, toMin.y) : 0;
        result += toMin.z > 0 ? solidanglecount(toMin.x, toMax.x, toMin.y, toMax.y, toMin.z) : 0;

        result += toMax.x < 0 ? solidanglecount(toMin.y, toMax.y, toMin.z, toMax.z, toMax.x) : 0;
        result += toMax.y < 0 ? solidanglecount(toMin.x, toMax.x, toMin.z, toMax.z, toMax.y) : 0;
        result += toMax.z < 0 ? solidanglecount(toMin.x, toMax.x, toMin.y, toMax.y, toMax.z) : 0;
        
        return result;
    }

    uint getCut(inout uint cut[16], const float3 p) {
        cut[0] = 1;
        uint endPoint = 0;

        float solidAngles[16];

        uint maxId = 0;
        float maxSolidAngle = -1e10;

        const uint clusterNum = 8;//TODO::params

        while (endPoint < clusterNum) {
            uint id = maxId;
            uint nodeId = cut[maxId];
            uint pChild = nodeId << 1;
            uint sChild = pChild + 1;

            Node pNode = nodes[pChild];
            Node sNode = nodes[sChild];

            cut[id] = pChild;
            solidAngles[id] = getSolidAngle(p, pNode.boundMin, pNode.boundMax);

            if (sNode.betaSum > 0) {
                endPoint++;
                cut[endPoint] = sChild;
                solidAngles[endPoint] = getSolidAngle(p, sNode.boundMin, sNode.boundMax);
            }

            maxSolidAngle = -1e10;
            for (uint i = 0; i <= endPoint; ++i) {
                if (solidAngles[i] > maxSolidAngle) {
                    maxSolidAngle = solidAngles[i];
                    maxId = i;
                }
            }
        }

        return endPoint;
    }

    uint inverseSample(uint y, float rnd) {
        float totalWeight = weightSum[y];
        if (totalWeight == 0) return 1 << kLogSize;

        float weight = rnd * totalWeight;
        uint begin = 0;
        uint end = 1 << kLogSize - 1;

        if (weight > prefixSum[uint2(end, y)]) return end;

        while (end - begin > 1) {
            uint mid = (begin + end) / 2;
            float cdf = prefixSum[uint2(mid, y)];
            float pmf = asfloat(prevSubspaceWeight[uint2(mid, y)]);
            if (cdf < weight) {
                begin = mid;
            }
            else if (cdf > weight) {
                end = mid;
            }
            else if (pmf == 0) {
                end = mid;
            }
            else {
                begin = mid;
                break;
            }
        }
        assert(asfloat(prevSubspaceWeight[uint2(begin, y)]) != 0);
        return begin;
    }

    uint2 sampleLightVertex(uint subspaceKey){
        uint begin = 0;
        uint end = counter - 1;
        uint sampleMortonCode = subspaceKey << 20;
        if (sampleMortonCode >= LightPathsIndexBuffer[end].y) return end;
        if (sampleMortonCode <= LightPathsIndexBuffer[begin].y) return begin;
        while ((end - begin) > 1) {
            uint mid = (begin + end) / 2;
            uint mortonCode = LightPathsIndexBuffer[mid].y;
            if (mortonCode < sampleMortonCode) {
                begin = mid;
            }
            else if (mortonCode > sampleMortonCode) {
                end = mid;
            }
            else {
                end = mid;
                begin = mid;
                break;
            }
        }
        assert(end >= begin);
        return uint2(begin, end);
    }

    float getAverageWeight(uint2 mortonCodeRange, uint y, out bool deadBranch) {
        uint2 mortonIndex = mortonCodeRange >> (30u - kLogSize);
        uint groupSize = 1 << kLogSize;
        float weight = (mortonIndex.y < groupSize - 1) ? (prefixSum[uint2(mortonIndex.y + 1, y)] - prefixSum[uint2(mortonIndex.x, y)]) : (weightSum[y] - prefixSum[uint2(mortonIndex.x, y)]);
        uint count = (mortonIndex.y < groupSize - 1) ? (prefixSumOfCount[uint2(mortonIndex.y + 1, y)] - prefixSumOfCount[uint2(mortonIndex.x, y)]) : (countSum[y] - prefixSumOfCount[uint2(mortonIndex.x, y)]);
        if (count == 0) {
            //deadBranch = true;
            return 0;
        }
        else {
            deadBranch = false;
            return weight / count;
        }
        
    }

    float getSecondaryMoment(uint2 mortonCodeRange, uint y) {
        uint2 mortonIndex = mortonCodeRange >> (30u - kLogSize);
        uint groupSize = 1 << kLogSize;
        float secondaryMoment = (mortonIndex.y < groupSize - 1) ? (prefixSumOfSecondaryMoment[uint2(mortonIndex.y + 1, y)] - prefixSumOfSecondaryMoment[uint2(mortonIndex.x, y)]) : (secondaryMomentSum[y] - prefixSumOfSecondaryMoment[uint2(mortonIndex.x, y)]);
        uint count = (mortonIndex.y < groupSize - 1) ? (prefixSumOfCount[uint2(mortonIndex.y + 1, y)] - prefixSumOfCount[uint2(mortonIndex.x, y)]) : (countSum[y] - prefixSumOfCount[uint2(mortonIndex.x, y)]);
        return (count == 0) ? FLT_MAX : secondaryMoment / count;
    }

    // https://graphics.stanford.edu/~seander/bithacks.html#IntegerLog
    inline uint uintLog2(uint v)
    {
        uint r; // result of log2(v) will go here
        uint shift;
        r = (v > (uint)0xFFFF) ? 1 << 4 : 0;
        v >>= r;
        shift = (v > (uint)0xFF) ? 1 << 3 : 0;
        v >>= shift;
        r |= shift;
        shift = (v > (uint)0xF) ? 1 << 2 : 0;
        v >>= shift;
        r |= shift;
        shift = (v > (uint)0x3) ? 1 << 1 : 0;
        v >>= shift;
        r |= shift;
        r |= (v >> 1);
        return r;
    }

    bool sampleThroughTheTree(uint nid, float rnd, uint y, out float p, out uint sampleIndex) {
        if (weightSum[y] == 0 || countSum[y] == 0) {
            return true;
        }
        
        float prob = 1.f;
        bool deadBranch = false;
        while (nid < leafNodeStart)
        {
            uint c0_id = nid << 1;
            uint c1_id = c0_id + 1;

            Node leftChild = nodes[c0_id];
            Node rightChild = nodes[c1_id];
            float prob0;

            float leftAverage = getAverageWeight(leftChild.mortonRange, y, deadBranch);
            float leftWeight = leftAverage * leftChild.betaSum;
            float rightAverage = getAverageWeight(rightChild.mortonRange, y, deadBranch);
            float rightWeight = rightAverage * rightChild.betaSum;
            //if (deadBranch) return deadBranch;

            //float localP = 0;
            if (leftWeight == 0) {
                if (rightWeight == 0) {
                    deadBranch = true;
                }
                else {
                    prob0 = 0;
                }
            }
            else if (rightWeight == 0) {
                prob0 = 1;
            }
            else {
                prob0 = leftWeight / (leftWeight + rightWeight);
            }

            if (!deadBranch)
            {
                nid = (rnd < prob0) ? c0_id : c1_id;
                prob *= (rnd < prob0) ? prob0 : (1 - prob0);
                rnd = (rnd < prob0) ? rnd / prob0 : (rnd - prob0) / (1 - prob0);
                float expection = (rnd < prob0) ? leftAverage : rightAverage;
                float secondaryMoment = getSecondaryMoment((rnd < prob0) ? leftChild.mortonRange : rightChild.mortonRange, y);
                float variance = secondaryMoment - expection * expection;
                if (variance < maxVariance[y] * 0.2f) {
                    uint level = uintLog2(leafNodeStart) - uintLog2(nid);
                    uint start = nid << level;
                    uint childCount = 1 << level;
                    uint end = min(start + childCount, leafNodeStart + counter);
                    uint realCount = end - start;
                    uint sampleLeaveNodeID = start + min(uint(rnd * realCount), realCount - 1);
                    sampleIndex = nodes[sampleLeaveNodeID].ID;
                    p = prob / realCount;
                    return false;
                }
            }
            else
            {
                break;
            }
        }

        if (!deadBranch) {
            sampleIndex = nodes[nid].ID;
            p = prob;
        }
        return deadBranch;
    }

    bool ConnectBDPT<VisibilityQuery : IVisibilityQuery>(inout PathState path, const Vertex v, inout VisibilityQuery vq, in float de, in float pe, in float radius, in bool isFirstNonDelta, in bool onlyUsePrimary, in bool isCausticPath, out RISSampleInfo risInfo){
        // select several valid vertex from the light path

        const bool isPrimaryHit = path.getVertexIndex() == 1;
        const HitType hitType = path.hit.getType();
        const bool isTriangleHit = hitType == HitType::Triangle;

        let lod = createTextureSampler(path, false, isTriangleHit);

        const IBSDF bsdf = gScene.materials.getBSDF(v.sd, lod);
        uint sampleIndex = 0;
        uint vertexMorton = GenMortonCode(v.sd.posW);
        uint yMorton = vertexMorton >> (30u - kLogSize);
        float one_over_prob = 1.f;
        float prob = 1.f;
        float inv_M = 1.0 / float(kLightPassHeight * kLightPassWidth);
        if (params.hasFlag(BDPTFlags::useSubspaceBDPT)) {

            
            bool deadBranch = sampleThroughTheTree(1, sampleNext1D(path.sg), yMorton, prob, sampleIndex);
            //bool deadBranch = true;
            if (deadBranch) {
                sampleIndex = min(uint(counter * sampleNext1D(path.sg)), counter - 1);
                one_over_prob = counter * inv_M;
            }
            else {
                assert(prob <= 1.f);
                one_over_prob = 1.f / prob * inv_M;
                //if (one_over_prob < 1.f) one_over_prob = 100.f;
                //one_over_prob = 10.f;
            }

            /*
            uint subspaceKey = inverseSample(yMorton, sampleNext1D(path.sg));
            if (subspaceKey >= (1 << kLogSize)) {
                sampleIndex = min(uint(counter * sampleNext1D(path.sg)), counter - 1);
                //one_over_prob = 1000000;
                one_over_prob = counter * inv_M;
            }
            else {
                uint2 begin = sampleLightVertex(subspaceKey);
                uint2 end = sampleLightVertex(subspaceKey + 1);
                sampleIndex = begin.y + min(uint((end.x - begin.y + 1) * sampleNext1D(path.sg)), end.x - begin.y);
                if (begin.y > end.x) one_over_prob = 1000000000;
                else one_over_prob = weightSum[yMorton] / asfloat(prevSubspaceWeight[uint2(subspaceKey, yMorton)]) * (end.x - begin.y + 1) * inv_M;// * float(kLightPassHeight * kLightPassWidth);
            }
            */
        }
        else {
            sampleIndex = min(uint(counter * sampleNext1D(path.sg)), counter - 1);
            one_over_prob = counter * inv_M;
        }
        

        //uint nodeID = sampleIndex + leafNodeStart;
        //sampleIndex = nodes[nodeID].ID;

        VertexInfo sampleInfo = LightPathsVertexsBuffer[sampleIndex];

        // one_over_prob *= sampleNode.betaSum / getIntensity(sampleInfo.beta);
        //one_over_prob /= sampleNode.pdf;

        float3 f = 0;
        float3 qStar = 0;
        Ray ray;

        ShadingData sd2 = v.sd;
        float3 wo = sd2.V;
        bool visible = false;
        
        // float radius = 0.005 * length(dimension);
        float3 samplePos = 0;

        if (sampleInfo.isLight()) {
            // light
            samplePos = sampleInfo.origin;
            float3 toSample = sampleInfo.origin - v.sd.posW;
            float dist2 = dot(toSample, toSample);
            toSample /= sqrt(dist2);

            const uint lobes = bsdf.getLobes(v.sd);

            float g = max(dot(normalize(sampleInfo.dir), -toSample) / dist2, 0);

            ray = getVisibiliyTestRay(v, sampleInfo);

            sd2.V = toSample;
            const IBSDF bsdf2 = gScene.materials.getBSDF(sd2, lod);
            float pdfRev2 = v.ConvertRevDensity(path, bsdf2.evalPdf(sd2, wo));
            float de2 = (1 + remap0(pdfRev2) * de) / remap0(v.pdfFwd);

            float pdfToX = abs(dot(toSample, sampleInfo.dir)) * M_1_PI;
            pdfToX = dot(-toSample, v.sd.N) * pdfToX / dist2;

            float OmegaMC = 0;
            if (onlyUsePrimary) {
                float pe2 = isFirstNonDelta ? 1.f : pe * pdfRev2 / remap0(v.pdfFwd);
                OmegaMC = M_PI * radius * radius * pe2 * pdfToX;
            }
            else {
                float pe2 = (1 + pe) * remap0(pdfRev2) / remap0(v.pdfFwd);
                
                float pdfFromX = bsdf.evalPdf(v.sd, toSample) * g / dist2;
                float pl = remap0(pdfFromX) / remap0(pdfToX);
                OmegaMC = (pe2 + 1.f + pl) * pdfToX * M_PI * radius * radius * (kLightPassHeight * kLightPassWidth);// * (kLightPassHeight * kLightPassWidth);
            }

            if (!isCausticPath) OmegaMC = 0;

            float OmegaCC = 1 / BDPTMIS(path, sampleInfo, v, de2);
            visible = vq.traceVisibilityRay(ray);
            //f = bsdf.eval(v.sd, toSample, path.sg) * g * sampleInfo.beta * BDPTMIS(path, sampleInfo, v, de2);
            //path.L += (visible) ? inv_M * v.beta * f * counter : 0;
        }
        else
        {
            // others
            Vertex sample = LightPathsVertexsBuffer[sampleIndex].unpack(false);
            samplePos = sample.sd.posW;
            // float3 result_q = RISWeight(v, sample, path, false);
            ray = getVisibiliyTestRay(v, sample);

            sd2.V = sample.sd.posW - v.sd.posW;
            float dist2 = dot(sd2.V, sd2.V);

            float3 toSample = sd2.V / sqrt(dist2);
            const IBSDF sampleBsdf = gScene.materials.getBSDF(sample.sd, lod);
            float pdfToX = sampleBsdf.evalPdf(sample.sd, -toSample);
            pdfToX = dot(toSample, v.sd.N) * pdfToX / dist2;

            qStar = sampleBsdf.eval(sample.sd, -toSample, path.sg) * sample.beta * G(sample, v);

            float3 mergeSamplePos = samplePos;
            float3 mergeSampleNormal = 0;
            float3 mergeRadiance = qStar;
            float mergeDL = sampleInfo.de;
            float mergeP_toLeft = pdfToX;
            float mergeW = 1.f;
            if (params.hasFlag(BDPTFlags::useReservoir)) {
                uint2 pixel = path.getPixel();
                uint threadSymbol = (pixel.y * 1920 + pixel.x) * kMaxSurfaceBounces + path.getVertexIndex() - 1;
                uint hash = vertexMorton >> 10;
                uint2 addr = uint2(hash % (1 << kLogSize), hash / (1 << kLogSize));
                float rnd = sampleNext1D(path.sg);
                uint outSymbol = 0;
                myLock(ReservoirSignal, addr, threadSymbol);
                /*
                RWTexture2D<uint2> Reservoir;       // x := W (float), y := sample count(uint)
                RWTexture2D<float4> samplePosition; // xyz := pos, w := dL
                RWTexture2D<float4> hitPointPosition; // xyz := pos,
                RWTexture2D<float4> normal;          // xyz := sample normal
                RWTexture2D<float4> radiance;        // rgb := radiance, a := p_toLeft
                ndir_to_oct_equal_area_unorm
                */
                uint2 localReservoir = Reservoir[addr];
                float weight = visible ? getIntensity(qStar) : 0;
                if (asfloat(localReservoir.x) + weight != 0) {
                    float3 reservoirRadiance = radiance[addr].xyz;
                    // float j = Jacobian(hitPointPosition[addr].xyz, v.sd.posW, samplePosition[addr].xyz, normal[addr].xyz);
                    float j = 1;
                    uint count = min(localReservoir.y, 100);
                    float reservoirWeight = getIntensity(reservoirRadiance);
                    float totalWeight = asfloat(localReservoir.x) * count * reservoirWeight * j + weight;
                    mergeW = totalWeight / (count + 1) / weight;
                    Reservoir[addr] = uint2(asuint(mergeW), count + 1);
                    if (rnd < weight / totalWeight) {
                        samplePosition[addr] = float4(samplePos, sampleInfo.de);
                        radiance[addr] = float4(qStar, pdfToX);
                    }
                    else {
                        mergeSamplePos = samplePosition[addr].xyz;
                        mergeDL = samplePosition[addr].w;
                        mergeRadiance = radiance[addr].rgb;
                        mergeP_toLeft = radiance[addr].w;
                    }
                }
                uint tmpID = myUnlock(ReservoirSignal, addr);
                assert(tmpID == threadSymbol);
            }

            sd2.V = mergeSamplePos - v.sd.posW;
            if (dot(sd2.V, sd2.N) <= 0) {
                visible = false;
            }
            else {
                dist2 = dot(sd2.V, sd2.V);
                toSample = sd2.V / sqrt(dist2);
                const IBSDF bsdf2 = gScene.materials.getBSDF(sd2, lod);
                float pdfRev2 = v.ConvertRevDensity(path, bsdf2.evalPdf(sd2, wo));
                float de2 = (1 + remap0(pdfRev2) * de) / remap0(v.pdfFwd);

                float pdfFromX = bsdf.evalPdf(v.sd, toSample) * abs(dot(toSample, sample.sd.N)) / dist2;

                float OmegaMC = 0;
                if (onlyUsePrimary) {
                    float pe2 = isFirstNonDelta ? 1.f : pe * pdfRev2 / remap0(v.pdfFwd);
                    OmegaMC = M_PI * radius * radius * pe2 * pdfToX;
                }
                else {
                    float pe2 = (1 + pe) * remap0(pdfRev2) / remap0(v.pdfFwd);
                    
                    float pl = (1 + sampleInfo.pe) * pdfFromX / remap0(pdfToX);
                    OmegaMC = (pe2 + 1.f + pl) * pdfToX * M_PI * radius * radius;
                }

                ShadingData sampleSd2 = sample.sd;
                float3 woSampleRev = sampleSd2.V;
                sampleSd2.V = -sd2.V;
                float pdfSampleRev = sampleBsdf.evalPdf(sampleSd2, woSampleRev);
                float distance = length(sampleInfo.origin - sample.sd.posW);
                pdfSampleRev *= abs(dot(normalize(woSampleRev), sampleInfo.originN)) / distance;

                // float pl = (1 + sampleInfo.pe) * pdfSampleRev / remap0(sampleInfo.pdfFwd);

                if (!isCausticPath || !params.hasFlag(BDPTFlags::useVertexMerge)) OmegaMC = 0;
                float OmegaCC = de2 * mergeP_toLeft + 1 + mergeDL * pdfFromX; //1 / BDPTMIS(path, sampleIndex, v, de2); // * kLightPassHeight * kLightPassWidth;

                // float OmegaMC = (pe2 + 1.f + pl) * pdfToX * M_PI * radius * radius;// * (kLightPassHeight * kLightPassWidth);// * (kLightPassHeight * kLightPassWidth);

                f = bsdf.eval(v.sd, toSample, path.sg) * mergeRadiance * mergeW / (OmegaMC + OmegaCC);
                visible = vq.traceVisibilityRay(ray);
                if (!params.hasFlag(BDPTFlags::useVertexMerge)) {
                    // if (params.hasFlag())
                    f *= one_over_prob;
                    // f = prob;
                }
            }
            
            
            //path.L += (visible) ? inv_M * v.beta * f * counter : 0;
            //f = RISWeight(v, sample, path, sample.beta) / (OmegaCC + OmegaMC);
        }

        

        path.L += (params.hasFlag(BDPTFlags::s2) && visible) ? v.beta * f : 0;

        uint xMorton = GenMortonCode(samplePos) >> (30u - kLogSize);
        
        uint2 subspaceIndex = uint2(xMorton, yMorton);
        if (visible) {
            float weight = getIntensity(qStar);
            float secondaryMoment = weight * weight;
            InterlockedAddFloat(SubspaceWeight, subspaceIndex, weight);
            InterlockedAddFloat(SubspaceSecondaryMoment, subspaceIndex, secondaryMoment);
        }
        //InterlockedAddFloat(SubspaceWeight, subspaceIndex, getIntensity(qStar));
        InterlockedAdd(SubspaceCount[subspaceIndex], 1);

        return getIntensity(qStar) > 0.01f;


        //if (all(qStar < 0.01f)) return false;

        //path.L += (params.hasFlag(BDPTFlags::s2) && visible) ? inv_M * v.beta * f * counter : 0;
        //return true;
        
        //path.L += inv_M * f * one_over_prob;
    }

    float Jacobian(float3 x1r, float3 x1q, float3 x2q, float3 normal) {
        float3 x1qToX2q = x2q - x1q;
        float3 x1rToX2q = x2q - x1r;
        float length1 = sqrt(dot(x1qToX2q, x1qToX2q));
        float length2 = sqrt(dot(x1rToX2q, x1rToX2q));
        float cosPhiR = abs(dot(x1rToX2q, normal) / length2);
        float cosPhiQ = abs(dot(x1qToX2q, normal) / length1);
        return cosPhiR / cosPhiQ * length1 / length2;
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

        ShadingData sdRev = qs.sd;
        float3 woRev = sdRev.V;
        sdRev.V = pt.sd.posW - qs.sd.posW;
        float3 toOrigin = sampleInfo.origin - qs.sd.posW;
        float dist2 = dot(toOrigin, toOrigin);
        float pdfRev_qs = bsdf1.evalPdf(sdRev, woRev) * abs(dot(toOrigin / sqrt(dist2), sampleInfo.originN)) / dist2;
        //dl = (1 + pdfRev_qs * dl) / remap0(sampleInfo.pdfFwd);
        //dl = 0;
        float N = 1.0;

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

        return 1.f / (de * pdfRev + 1 + dl * pdfFwd);
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

        return 1.f / (de * pdfRev + 1 + dl * pdfFwd);
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

/*

    bool handleCollectIntersection(const float3 origin, const uint primIndex) {
        
        const uint2 primIndex2D = uint2(primIndex % 2048, primIndex / 2048);

        float radius = globalRadius;
        float3 center = hitPointPositionI[primIndex2D].xyz;
        float3 toCenter = center - origin;
        return dot(toCenter, toCenter) < radius * radius;
    }

    void handleCollectAnyHit(inout uint4 payload, const float3 origin, const float3 normal, const uint primIndex, const float rnd) {
        const uint2 primIndex2D = uint2(primIndex % 2048, primIndex / 2048);
        const float3 hitPoinFaceN = oct_to_ndir_equal_area_unorm(normalI[primIndex2D].zw);

        if (dot(normal, hitPoinFaceN) < 0.9) return;

        const float3 sPos = samplePositionI[primIndex2D].xyz;
        const float3 sN = oct_to_ndir_equal_area_unorm(normalI[primIndex2D].xy);
        const float3 hPos = hitPointPositionI[primIndex2D].xyz;
        // float count1 = payload.z;
        // float weight = getIntensity(radianceI[primIndex2D].xyz);
        uint4 reservoirNew = ReservoirI[primIndex2D];
        float weight = asfloat(reservoirNew.x);

        float3 toNew = origin - sPos;
        float dist2New = dot(toNew, toNew);
        toNew /= sqrt(dist2New);
        float3 toPrev = hPos - sPos;
        float dist2Prev = dot(toPrev, toPrev);
        toPrev /= sqrt(dist2Prev);
        float cosNew = abs(dot(toNew, sN));
        float cosPrev = abs(dot(toPrev, sN));
        float j = cosNew * dist2Prev / (cosPrev * dist2New);

        float WNew = asfloat(reservoirNew.y);
        uint MNew = reservoirNew.z;

        float weightNew = weight / j * WNew * MNew;

        payload.z += MNew;
        float w = asfloat(payload.y) + weightNew;
        payload.y = asuint(w);
        if (rnd < weightNew / w) {
            payload.x = asuint(weight);
            payload.w = primIndex;
        }

    }

    RayDesc generateTestRay(inout HitInfo hit) {
        
        VertexData v = gScene.getVertexData(hit.getTriangleHit());

        RayDesc ray;
        ray.Origin = v.posW;
        ray.TMin = 0.0001f;
        ray.TMax = 100.f;
        ray.Direction = v.normalW;

        return ray;
    }
*/
/*
    void CameraPathHandleHit<VisibilityQuery : IVisibilityQuery>(inout PathState path, inout VisibilityQuery vq, inout ExtraPayload exPayload, in uint4 risPayload) {
        const bool isPrimaryHit = path.getVertexIndex() == 1;
        const uint index = path.getVertexIndex() - 1;
        const HitType hitType = path.hit.getType();
        const bool isTriangleHit = hitType == HitType::Triangle;
        
        let lod = createTextureSampler(path, isPrimaryHit, isTriangleHit);

        // Load shading data. This is a long latency operation.
        ShadingData sd = loadShadingData(path.hit, path.origin, path.dir, isPrimaryHit, lod);

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

        Vertex vertex = Vertex::create(sd, path.thp, path.pdf, 0, VertexType::SurfaceOrMedium);
        vertex.pdfFwd = vertex.ConvertDensity(path);

        const IBSDF bsdf = gScene.materials.getBSDF(sd, lod);
        BSDFProperties bsdfProperties = bsdf.getProperties(sd);

        const uint lobes = bsdf.getLobes(sd);
        const bool hasNonDeltaLobes = (lobes & (uint)LobeType::NonDelta) != 0;

        // Disable specular lobes if caustics are disabled and path already contains a diffuse vertex.
        bool isSpecular = bsdfProperties.roughness <= params.specularRoughnessThreshold;
       
        BSDFSample bs;
        bool valid = bsdf.sample(sd, path.sg, bs, kUseBSDFSampling);

        float3 wi = sd.V;
        ShadingData sdRev = sd;
        sdRev.V = bs.wo;
        const IBSDF bsdfRev = gScene.materials.getBSDF(sdRev, lod);
        float pdfRev = bsdfRev.evalPdf(sdRev, wi);

        bool hasDelta = (lobes & (uint)LobeType::Delta) != 0;
        // bool isFirstNonDelta = (!exPayload.isFirstNonDelta) && (!hasDelta);
        // exPayload.isFirstNonDelta = isFirstNonDelta;
        vertex.delta = hasDelta;
        float N = 1.f;

        float convertRevPdf = remap0(vertex.ConvertRevDensity(path, pdfRev));

        bool sampleUpperHemisphere = ((lobes & (uint)LobeType::NonDeltaReflection) != 0);
        if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume()) sampleUpperHemisphere = false;
        bool sampleLowerHemisphere = ((lobes & (uint)LobeType::NonDeltaTransmission) != 0);

        bool isDiffuseHit = !hasDelta;//bs.isLobe(LobeType::Diffuse) || (bs.isLobe(LobeType::Reflection) && (bsdfProperties.roughness > 0.5));

        bool isFirstNonDelta = isDiffuseHit && !path.hasFlag(PathFlags::isFirstNonDelta);
        if (isFirstNonDelta) path.setFlag(PathFlags::isFirstNonDelta, true);
        
        // s >= 1
        if (!hasDelta) {
           
            uint sampleIndex = min(uint(counter * sampleNext1D(path.sg)), counter - 1);

            VertexInfo sampleInfo = LightPathsVertexsBuffer[sampleIndex];

            float3 f = 0;
            float3 qStar = 0;
            Ray ray;

            bool visible = false;
            float inv_M = 1.0 / float(kLightPassHeight * kLightPassWidth);
            uint M = kLightPassHeight * kLightPassWidth;
            
            // float radius = 0.005 * length(dimension);

            if (sampleInfo.isLight()) {
                // light
                float3 toSample = sampleInfo.origin - sd.posW;
                float dist2 = dot(toSample, toSample);
                toSample /= sqrt(dist2);

                float g = max(dot(normalize(sampleInfo.dir), -toSample) / dist2, 0);

                ray = getVisibiliyTestRay(vertex, sampleInfo);

                sdRev.V = toSample;
                const IBSDF bsdf2 = gScene.materials.getBSDF(sdRev, lod);
                float pdfRev2 = vertex.ConvertRevDensity(path, bsdf2.evalPdf(sdRev, wi));
                float de2 = (1.f + remap0(pdfRev2) * exPayload.d) / remap0(vertex.pdfFwd);
                
                float pdfToX = abs(dot(toSample, sampleInfo.dir)) * M_1_PI;
                pdfToX = dot(-toSample, sd.N) * pdfToX / dist2;
                
                visible = vq.traceVisibilityRay(ray);
                f = bsdf.eval(sd, toSample, path.sg) * g * sampleInfo.beta * BDPTMIS(path, sampleInfo, vertex, de2);
                path.L += (visible) ? inv_M * vertex.beta * f * counter : 0;
            }
            else
            {
                // others
                Vertex sample = LightPathsVertexsBuffer[sampleIndex].unpack(false);

                // float3 result_q = RISWeight(v, sample, path, false);
                ray = getVisibiliyTestRay(vertex, sample);

                sdRev.V = sample.sd.posW - sd.posW;
                float dist2 = dot(sdRev.V, sdRev.V);
                const IBSDF bsdf2 = gScene.materials.getBSDF(sdRev, lod);
                float pdfRev2 = vertex.ConvertRevDensity(path, bsdf2.evalPdf(sdRev, wi));
                float de2 = (1 + remap0(pdfRev2) * exPayload.d) / remap0(vertex.pdfFwd);

                float3 toSample = sdRev.V / sqrt(dist2);
                const IBSDF sampleBsdf = gScene.materials.getBSDF(sample.sd, lod);
                float pdfToX = sampleBsdf.evalPdf(sample.sd, -toSample);
                pdfToX = dot(toSample, sd.N) * pdfToX / dist2;

                

                // float OmegaMC = (pe2 + 1.f + pl) * pdfToX * M_PI * radius * radius;// * (kLightPassHeight * kLightPassWidth);// * (kLightPassHeight * kLightPassWidth);
                visible = vq.traceVisibilityRay(ray);
                //visible = true;
                qStar = visible ? sampleBsdf.eval(sample.sd, -toSample, path.sg) * sample.beta * G(sample, vertex) : 0;

                if (risPayload.z == 0) {
                    float weight = getIntensity(qStar);
                    float W = inv_M * counter;
                    
                    if (weight > 0.01f) {
                        uint aabbIndex = hitPointAABB.IncrementCounter();
                        hitPointAABB[aabbIndex] = AABB(sd.posW - globalRadius, sd.posW + globalRadius);
                        uint2 index2D = uint2(aabbIndex % 2048, aabbIndex / 2048);
                        
                        ReservoirO[index2D] = uint4(asuint(weight), asuint(W), M, aabbIndex);
                        samplePositionO[index2D] = float4(sample.sd.posW, sampleInfo.de);
                        hitPointPositionO[index2D] = float4(sd.posW, pdfToX);
                        normalO[index2D] = float4(ndir_to_oct_equal_area_unorm(normalize(sample.sd.N)), ndir_to_oct_equal_area_unorm(normalize(sd.faceN)));
                        radianceO[index2D] = float4(qStar, globalRadius);

                        uint3 index3D = uint3(path.getPixel(), path.getVertexIndex() - 1);
                        pathPos[index3D] = uint4(asuint(sd.posW), aabbIndex);
                        pathRadiance[index3D] = float4(vertex.beta, exPayload.d);
                        pathHitInfo[index3D] = path.hit.getData();
                    }
                    else {
                        uint3 index3D = uint3(path.getPixel(), path.getVertexIndex() - 1);
                        pathPos[index3D] = uint4(asuint(sd.posW), 0xffffffff);
                    }
                    
                    //float mis = 1.f / (de2 * pdfToX + 1 + sampleInfo.de * vertex.ConvertDensity(bsdf.evalPdf(sd, toSample), sample));
                    //path.L += vertex.beta * bsdf.eval(sd, toSample, path.sg) * qStar * W * mis;
                    //path.L += float3(0, 1, 1);
                }
                else {
                    
                    uint4 reservoir = risPayload; // x := sample weight(float), y := total weight(float), z := sample count, w := index
                    float weight = asfloat(risPayload.x);
                    uint2 primIndex2D = uint2(risPayload.w % 2048, risPayload.w / 2048);
                    float3 sPos = samplePositionI[primIndex2D].xyz;
                    float3 hPos = hitPointPositionI[primIndex2D].xyz;
                    float3 sN = oct_to_ndir_equal_area_unorm(normalI[primIndex2D].xy);

                    float3 toNew = sd.posW - sPos;
                    float dist2New = dot(toNew, toNew);
                    toNew /= sqrt(dist2New);
                    float3 toPrev = hPos - sPos;
                    float dist2Prev = dot(toPrev, toPrev);
                    toPrev /= sqrt(dist2Prev);
                    float cosNew = abs(dot(toNew, sN));
                    float cosPrev = abs(dot(toPrev, sN));
                    float j = cosNew * dist2Prev / (cosPrev * dist2New);

                    float WNew = asfloat(risPayload.y);
                    uint MNew = min(20 * M, risPayload.z);

                    float weightNew = weight / j * WNew * MNew;

                    float qWeight = getIntensity(qStar);
                    float w = qWeight * counter + weightNew;
                    uint count = M + MNew;
                    if (sampleNext1D(path.sg) < weightNew / w) {
                        
                        float W = w / (weight * count);
                        float3 radiance = radianceI[primIndex2D].xyz;
                        // TODO
                        float mis = 1.f / (de2 * hitPointPositionI[primIndex2D].w + 1.f + samplePositionI[primIndex2D].w * vertex.ConvertDensity(bsdf.evalPdf(sd, -toNew), sPos, sN));
                        //path.L += vertex.beta * bsdf.eval(sd, -toNew, path.sg) * radiance * W * mis;

                        uint aabbIndex = hitPointAABB.IncrementCounter();
                        hitPointAABB[aabbIndex] = AABB(vertex.sd.posW - globalRadius, vertex.sd.posW + globalRadius);
                        uint2 index2D = uint2(aabbIndex % 2048, aabbIndex / 2048);

                        ReservoirO[index2D] = uint4(asuint(weight), asuint(W), count, aabbIndex);
                        samplePositionO[index2D] = samplePositionI[primIndex2D];
                        hitPointPositionO[index2D] = float4(sd.posW, pdfToX);
                        normalO[index2D] = float4(normalI[primIndex2D].xy, ndir_to_oct_equal_area_unorm(normalize(sd.faceN)));
                        radianceO[index2D] = float4(radiance, globalRadius);
                    }
                    else {
                        float W = w / (qWeight * count);
                        float mis = 1.f / (de2 * pdfToX + 1 + sampleInfo.de * vertex.ConvertDensity(bsdf.evalPdf(sd, toSample), sample));
                        //path.L += vertex.beta * bsdf.eval(sd, toSample, path.sg) * qStar * W * mis;

                        uint aabbIndex = hitPointAABB.IncrementCounter();
                        hitPointAABB[aabbIndex] = AABB(vertex.sd.posW - globalRadius, vertex.sd.posW + globalRadius);
                        uint2 index2D = uint2(aabbIndex % 2048, aabbIndex / 2048);

                        ReservoirO[index2D] = uint4(asuint(qWeight), asuint(W), count, aabbIndex);
                        samplePositionO[index2D] = float4(sample.sd.posW, sampleInfo.de);
                        hitPointPositionO[index2D] = float4(sd.posW, pdfToX);
                        normalO[index2D] = float4(ndir_to_oct_equal_area_unorm(normalize(sample.sd.N)), ndir_to_oct_equal_area_unorm(normalize(sd.faceN)));
                        radianceO[index2D] = float4(qStar, globalRadius);
                    }
                    //path.L += 1;
                    
                }
                //f = bsdf.eval(sd, toSample, path.sg) * qStar * BDPTMIS(path, sampleIndex, vertex, de2);
                
                //path.L += inv_M * vertex.beta * f * counter;
            } 
        }

        if (!hasDelta)
        { // s == 1
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

                ShadingData sd1 = sd;
                float3 wo = sd1.V;
                sd1.V = -wi;
                const IBSDF bsdf1 = gScene.materials.getBSDF(sd1, lod);
                float pdfRev1 = vertex.ConvertRevDensity(path, bsdf1.evalPdf(sd1, wo));
                float de1 = (1 + remap0(pdfRev1) * exPayload.d) / remap0(vertex.pdfFwd);

                if (params.hasFlag(BDPTFlags::s1) && any(Lr > 0.f))
                {
                    // Ray ray = ls.getVisibilityRay();
                    logTraceRay(PixelStatsRayType::Visibility);
                    bool visible = vq.traceVisibilityRay(ray);

                    // float radius = 0.005 * length(dimension);
                    float pdfToX = dirPdf;
                    pdfToX = dot(-dir, vertex.sd.N) * pdfToX / distSqr;
                    if (visible) {
                        float pdfRev = lightVertex.ConvertDensity(abs(dot(dir, lightVertex.sd.N)) * M_1_PI, vertex);
                        float pdfFwd = vertex.ConvertDensity(bsdf.evalPdf(sd, dir), lightVertex);
                        // todo dl
                        float mis = 1.f / (de1 * pdfRev + 1.f + 1.f / lightVertex.pdfFwd * pdfFwd);
                        path.L += Lr * mis;
                    }
                    
                }
            }
        }

        // s == 0
        bool computeEmissive = kUseEmissiveLights;

        // bool sampleLowerHemisphere = isCurveHit || isCurvePolyTubeHit || ((lobes & (uint)LobeType::NonDeltaTransmission) != 0);
        if (params.hasFlag(BDPTFlags::s0) && computeEmissive && any(bsdfProperties.emission > 0.f)) {
            TriangleHit triangleHit = path.hit.getTriangleHit();
            TriangleLightHit hit;
            hit.triangleIndex = gScene.lightCollection.getTriangleIndex(triangleHit.instanceID, triangleHit.primitiveIndex);
            hit.posW = sd.posW;
            hit.normalW = sd.frontFacing ? sd.faceN : -sd.faceN;

            float cosTheta0 = abs(dot(normalize(path.dir), hit.normalW));
            float pdfRev0 = vertex.ConvertRevDensity(path, cosTheta0 * M_1_PI);
            float de0 = (1 + pdfRev0 * exPayload.d) / remap0(vertex.pdfFwd);
            path.L += vertex.beta * bsdfProperties.emission * BDPTMIS(path, hit, de0);
        }

        exPayload.d = (N + exPayload.d * convertRevPdf) / remap0(vertex.pdfFwd);

        path.origin = sd.computeNewRayOrigin();

        if (valid) valid = generateScatterRay(bs, sd, bsdf, path);

        const bool isLastVertex = hasFinishedSurfaceBounces(path) || path.getVertexIndex() >= kMaxSurfaceBounces;

        if (isLastVertex || !valid) {
            path.terminate();
        }
    }
*/
};


/*
Texture2D<float4> samplePosition;   // xyz := pos, w := dL
Texture2D<float4> hitPointPosition; // xyz := pos, w := p_toLeft
Texture2D<float4> normal;           // xy := sample normal, zw := hit point normal
Texture2D<float4> radiance;         // rgb := radiance, w := radii

// reservoir
Texture2D<uint4> Reservoir; // x := sample weight(float), y := W (float), z := sample count, w := index
*/