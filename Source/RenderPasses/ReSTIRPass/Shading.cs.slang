#include "D:/ReSTIR_PT/ReSTIR_PT/Source/Falcor/Utils/Math/MathConstants.slangh"
import Utils.Sampling.SampleGenerator;
import Utils.Math.MathHelpers;
import Utils.Color.ColorHelpers;
import Scene.RaytracingInline;
import Scene.HitInfo;
import Scene.Scene;
import Scene.Shading;
import Rendering.Materials.StandardMaterial;
import Rendering.Materials.MaterialHelpers;
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;


import ReSTIR;
import Helpers;
import ShadingDataLoader;

cbuffer CB
{
    uint2 gViewportDims;
    uint gFrameIndex;
    uint gInputBufferIndex;
    ResamplingRuntimeParameters gParams;
    ShadingDataLoader gShadingDataLoader;
    EnvMapSampler gEnvMapSampler;
    EmissiveLightSampler gEmissiveLightSampler;
}

static struct Data
{
    StandardMaterial standardMaterial;
} gData;

RWStructuredBuffer<Reservoir> gReservoirs;
RWTexture2D<float4> gShadingOutput;

[numthreads(16, 16, 1)]
void main(uint2 threadIndex : SV_DispatchThreadID, uint3 groupIndex: SV_GROUPID)
{
    if (any(threadIndex >= gViewportDims)) 
    {
        return;
    }

    float3 res = 0.0;
    
    SampleGenerator sg = SampleGenerator(threadIndex, gFrameIndex);
    ShadingData sd;
    bool valid = gShadingDataLoader.loadShadingData(threadIndex, gViewportDims, gScene.camera, sd);
    if (valid)
    {
        float3 posWOrigin = sd.computeNewRayOrigin();

#if !_BRUTE_FORCE
        // Reservoir Shading
        {
            Reservoir r = loadReservoir(gParams, gReservoirs, threadIndex, gInputBufferIndex);
            if (isValidReservoir(r))
            {
                uint lightIndex = getReservoirLightIndex(r);
                float2 lightUV = getReservoirSampleUV(r);
                float3 Le;

                RayDesc ray;
                ray.Origin = posWOrigin;
                ray.TMin = 0.001f;
                if (lightIndex == EnvMapIndex)
                {
                    ray.Direction = gScene.envMap.toWorld(latlong_map_to_world(lightUV));
                    ray.TMax = 1.0e38f;

                    Le = gScene.envMap.eval(ray.Direction);
                }
                else
                {
                    EmissiveTriangle tri = gScene.lightCollection.getTriangle(lightIndex);
                    float3 barycentric = sample_triangle(lightUV);
                    float3 lightPos = tri.getPosition(barycentric);
                    float3 toLight = lightPos - posWOrigin; // Unnormalized light vector
                    const float distSqr = max(FLT_MIN, dot(toLight, toLight)); // Clamp to avoid NaNs below
                    float distance = sqrt(distSqr);
                    ray.Direction = toLight / distance;
                    ray.TMax = distance - 0.001f;

                    float2 uv = tri.getTexCoord(barycentric);
                    Le = gScene.getEmissive(tri.materialID, uv, gScene.materialResources[tri.materialID]);
                }

                if (traceSceneVisibilityRay<1>(ray, RAY_FLAG_NONE, 0xff))
                {
                    res += Le * gData.standardMaterial.eval(sd, ray.Direction) * getReservoirInvPdf(r);
                }
#if _STORE_FINAL_VISIBILITY
                else
                {
                    uint pointer = reservoirPositionToPointer(gParams, threadIndex, gInputBufferIndex);
                    gReservoirs[pointer].weightSum = 0.0f;
                }
#endif
            }
        }
#else
        // Groundtruth Shading
        {
            EnvMapSample envMapSample;
            if (gEnvMapSampler.sample(sampleNext2D(sg), envMapSample))
            {
                RayDesc ray;
                ray.Origin = posWOrigin;
                ray.Direction = envMapSample.dir;
                ray.TMin = 0.001f;
                ray.TMax = 1.0e38f;

                if (traceSceneVisibilityRay<1>(ray, RAY_FLAG_NONE, 0xff))
                {
                    res += envMapSample.Le * gData.standardMaterial.eval(sd, envMapSample.dir) / envMapSample.pdf;
                }
            }

            TriangleLightSample lightSample;
            if (gEmissiveLightSampler.sampleLight(posWOrigin, sd.N, true, sg, lightSample) && dot(sd.N, lightSample.dir) >= kMinCosTheta)
            {
                RayDesc ray;
                ray.Origin = posWOrigin;
                ray.Direction = lightSample.dir;
                ray.TMin = 0.001f;
                ray.TMax = lightSample.distance - 0.001f;
                if (traceSceneVisibilityRay<1>(ray, RAY_FLAG_NONE, 0xff))
                {
                    res += lightSample.Le * gData.standardMaterial.eval(sd, lightSample.dir) / lightSample.pdf;
                }
            }
        }
#endif
        res += sd.emissive;
    }
    else
    {
        RayDesc ray = gScene.camera.computeRayPinhole(threadIndex, gViewportDims).toRayDesc();
        res += gScene.envMap.eval(ray.Direction);
    }

    gShadingOutput[threadIndex] = float4(res, 1.0);
}
