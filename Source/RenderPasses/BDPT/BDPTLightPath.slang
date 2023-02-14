#include "Scene/SceneDefines.slangh"
#include "BDPTUtils.slangh"
import Scene.Raytracing;
import Scene.RaytracingInline; // For visibility queries.
import Scene.Intersection;
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Lights.LightHelpers;
import Rendering.Materials.Microfacet;
import Rendering.Materials.InteriorListHelpers;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import PathData;

static const bool kUseEnvLight = USE_ENV_LIGHT;
static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;

shared cbuffer GlobalCB
{
    float gMinT;            // Min distance to start a ray to avoid self-occlusion
    uint gFrameCount;       // An integer changing every frame to update the random number
    uint gMatIndex;         // The index of material brdf to use
    float gRefractiveIndex; // The refractive index of dielectric materials
    uint gMaxDepth;         // Maximum number of recursive bounces to allow
    float gEmitMult;        // Multiply emissive amount by this factor (set to 1, usually)
    float gClampUpper;      // Clamping upper bound
}



enum class BDLightType
{
    EnvMap = 0,
    Emissive,
    Analytic
};

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

bool generateEmissiveSample_BDPT(inout SampleGenerator sg, out Vertex lv, out float3 ray_dir, out float dirPdf)
{
    lv = {}; // Default initialization to avoid divergence at returns.
    if (!kUseEmissiveLights) return false;

    // sample the lights uniformly.
    float uLight = sampleNext1D(sg);
    uint triangleCount = gScene.lightCollection.getActiveTriangleCount();
    uint idx = min((uint)(uLight * triangleCount), triangleCount - 1); // Safety precaution as the result of the multiplication may be rounded to triangleCount even if uLight < 1.0 when triangleCount is large.
    uint triangleIndex = gScene.lightCollection.activeTriangles[idx];
    float triangleSelectionPdf = 1.f / (float)triangleCount;

    // sample the triangle uniformly.
    float2 u = sampleNext2D(sg);
    const float3 barycentrics = sample_triangle(u);
    const EmissiveTriangle tri = gScene.lightCollection.getTriangle(triangleIndex);
    float3 pos = tri.getPosition(barycentrics);
    float3 normal = tri.normal;
    float2 uv = tri.getTexCoord(barycentrics);
    float3 Le = gScene.materials.evalEmissive(tri.materialID, uv);

    float denom = max(FLT_MIN, tri.area);
    float posPdf = 1.f / denom;

    // sample a cosine-weighted outgoing direction w.
    float2 v = sampleNext2D(sg);
    float3 w = sample_cosine_hemisphere_polar(v, dirPdf);
    float3 v2, v3;
    CoordinateSystem(normal, v2, v3);
    w = w.x * v2 + w.y * v3 + w.z * normal; // ray direction
    ray_dir = w;

    if (posPdf <= 0.f || dirPdf <= 0.f || all(Le <= 0.f)) return false;

    // Setup returned sample.
    lv.beta = Le * abs(dot(normal, w)) / (triangleSelectionPdf * posPdf * dirPdf); // the "beta" of the subpath
    lv.pdfFwd = triangleSelectionPdf * posPdf;
    lv.type = VertexType::Light;
    lv.sd.N = normal;
    lv.sd.posW = pos;

    return any(lv.beta > 0.f);
}

bool generateAnalyticLightSample_BDPT(inout SampleGenerator sg, out Vertex lv, out float3 ray_dir, out float dirPdf)
{
    lv = {}; // Default initialization to avoid divergence at returns.

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
    lv.beta = lightSample.Li * dot(lv.sd.N, ray_dir) / (lightCountPdf * lightSample.pdf * lightSample.distance);
    lv.pdfFwd = lightCountPdf * lightSample.pdf;
    lv.type = VertexType::Light;
    dirPdf = lightSample.distance;

    return any(lv.beta > 0.f);
}

bool generateLightSample_BDPT(inout SampleGenerator sg, out Vertex lv, out float3 ray_dir, out float dirPdf)
{
    lv = {};

    uint lightType;
    float selectionPdf;
    if (!selectLightType(lightType, selectionPdf, sg)) return false;

    bool valid = false;
    if (kUseEnvLight && lightType == (uint)BDLightType::EnvMap) valid = false;
    if (kUseEmissiveLights && lightType == (uint)BDLightType::Emissive)
    {
        // Emissive light samplers have an option to exclusively sample the upper hemisphere.
        valid = generateEmissiveSample_BDPT(sg, lv, ray_dir, dirPdf);
    }
    if (kUseAnalyticLights && lightType == (uint)BDLightType::Analytic)
    {
        valid = generateAnalyticLightSample_BDPT(sg, lv, ray_dir, dirPdf);
    }
    if (!valid) return false;

    // Account for light type selection.
    lv.pdfFwd *= selectionPdf;
    lv.beta /= selectionPdf;

    return true;
}

[shader("raygeneration")]
void GenLightPath()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;

    uint2 pixel = launchIndex;

    SampleGenerator sg = SampleGenerator(launchIndex, gFrameCount);

    BDPathVertex lightPath[9];
    for (uint i = 0; i < 9; i++)
    {
        lightPath[i] = BDPathVertex.init();
    }
    Vertex lv;
    float3 ray_dir;
    float dirPdf;
    if (!generateLightSample_BDPT(sg, lv, ray_dir, dirPdf)) return;
    lightPath[0].update(lv);

    RayPayload lightRayPayload = initPayload(lightPath[0].posW, ray_dir, lv.sd.N, dirPdf, sg);

    for (uint depth = 0; depth < gMaxDepth && !lightRayPayload.terminated; ++depth) {
        shootRay(lightRayPayload);
        lightPath[depth + 1] = BDPathVertex.create(lightRayPayload.color,
                                                   lightRayPayload.posW,
                                                   lightRayPayload.N,
                                                   lightRayPayload.V,
                                                   lightRayPayload.dif,
                                                   lightRayPayload.spec,
                                                   lightRayPayload.rough,
                                                   lightRayPayload.isSpecular,
                                                   lightRayPayload.color)
    }
}