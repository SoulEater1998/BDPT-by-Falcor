#include "D:/ReSTIR_PT/ReSTIR_PT/Source/Falcor/Utils/Math/MathConstants.slangh"
import Utils.Sampling.SampleGenerator;
import Utils.Math.MathHelpers;
import Utils.Color.ColorHelpers;
import Scene.RaytracingInline;
import Scene.Scene;
import Scene.Shading;
import Rendering.Materials.StandardMaterial;
import Rendering.Materials.MaterialHelpers;
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;

import ReSTIR;
import ShadingDataLoader;
import Helpers;

cbuffer CB
{
    uint2 gViewportDims;
    uint gFrameIndex;
    uint gInputBufferIndex;
    uint gHistoryBufferIndex;
    uint gOutputBufferIndex;
    ResamplingRuntimeParameters gParams;
    ShadingDataLoader gShadingDataLoader;
    EnvMapSampler gEnvMapSampler;
    EmissiveLightSampler gEmissiveLightSampler;
}

RWStructuredBuffer<Reservoir> gReservoirs;

static struct Data
{
    StandardMaterial standardMaterial;
} gData;

static const int maxHistoryLength = 20;
static const float positionThreshold = 0.1f;
static const float normalThreshold = 0.65f;

[numthreads(16, 16, 1)]
void main(uint2 threadIndex : SV_DispatchThreadID, uint3 groupIndex: SV_GROUPID)
{
    if (any(threadIndex >= gViewportDims)) 
    {
        return;
    }

    SampleGenerator sg = SampleGenerator(threadIndex, gFrameIndex + 2 * 13);
    Reservoir temporalResult = emptyReservoir();
    ShadingData sd;
    bool valid = gShadingDataLoader.loadShadingData(threadIndex, gViewportDims, gScene.camera, sd);
    if (valid)
    {
        float3 posWOrigin = sd.computeNewRayOrigin();

        Reservoir curSample = loadReservoir(gParams, gReservoirs, threadIndex, gInputBufferIndex);
        
        int historyLimit = min(Reservoir::c_MaxM, maxHistoryLength * curSample.M);

        int selectedLightPrevID = -1;
        if (isValidReservoir(curSample))
        {
            selectedLightPrevID = getReservoirLightIndex(curSample);
        }

        Reservoir state = emptyReservoir();
        combineReservoirs(state, curSample, 0.5f, curSample.targetPdf);

        float2 motion = gShadingDataLoader.motionVector[threadIndex];
        motion.xy += (sampleNext2D(sg) - 0.5);
        float2 reprojectedSamplePosition = float2(threadIndex) + motion.xy;
        int2 prevPos = int2(round(reprojectedSamplePosition));

        ShadingData neighborSd;
        bool foundNeighbor = false;
        const float radius = 4;

        for (int i = 0; i < 9; i++)
        {
            int2 offset = 0;
            if(i > 0)
            {
                offset.x = int((sampleNext1D(sg) - 0.5) * radius);
                offset.y = int((sampleNext1D(sg) - 0.5) * radius);
            }
            int2 idx = prevPos + offset;

            if (!isReprjTextureCoordValid(idx, gViewportDims))
            {
                continue;
            }

            if (!gShadingDataLoader.loadShadingDataPrevFrame(idx, gViewportDims, gScene.camera, neighborSd))
            {
                continue;
            }

            if (!isReprjNormalValid(sd.N, neighborSd.N, normalThreshold))
            {
                continue;
            }

            if (!isReprjPositionValid(sd.posW, neighborSd.posW, positionThreshold))
            {
                continue;
            }

            prevPos = idx;
            foundNeighbor = true;
            break;
        }

        bool selectedPreviousSample = false;
        uint previousM = 0;

        if (foundNeighbor)
        {
            Reservoir prevSample = loadReservoir(gParams, gReservoirs, prevPos, gHistoryBufferIndex);
            prevSample.M = min(prevSample.M, historyLimit);
            uint originalPrevLightID = getReservoirLightIndex(prevSample);
            float weightAtCurrent = 0;
            if (isValidReservoir(prevSample))
            {
                uint lightIndex = originalPrevLightID;
                float2 lightUV = getReservoirSampleUV(prevSample);
                float3 dir;
                float3 Le;
                if (lightIndex == EnvMapIndex)
                {
                    dir = gScene.envMap.toWorld(latlong_map_to_world(lightUV));
                    Le = gScene.envMap.eval(dir);
                }
                else
                {
                    EmissiveTriangle tri = gScene.lightCollection.getTriangle(lightIndex);
                    float3 barycentric = sample_triangle(lightUV);
                    float3 lightPos = tri.getPosition(barycentric);
                    float3 toLight = lightPos - posWOrigin; // Unnormalized light vector
                    const float distSqr = max(FLT_MIN, dot(toLight, toLight)); // Clamp to avoid NaNs below
                    float distance = sqrt(distSqr);
                    dir = toLight / distance;
                    float2 uv = tri.getTexCoord(barycentric);
                    Le = gScene.getEmissive(tri.materialID, uv, gScene.materialResources[tri.materialID]);
                }
                weightAtCurrent = luminance(Le * gData.standardMaterial.eval(sd, dir));
            }

            previousM = prevSample.M;

            bool sampleSelected = combineReservoirs(state, prevSample, sampleNext1D(sg), weightAtCurrent);
            if(sampleSelected)
            {
                selectedPreviousSample = true;
                selectedLightPrevID = int(originalPrevLightID);
            }
        }
        
        // If you want to do debias, you need to store previous frame GBuffer
        {
            finalizeResampling(state, 1.0, state.M);
            temporalResult = state;
        }
    }

    storeReservoir(gParams, temporalResult, gReservoirs, threadIndex, gOutputBufferIndex);
}
