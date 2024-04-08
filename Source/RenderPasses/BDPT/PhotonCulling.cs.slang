#include "BDPTUtils.slangh"
#include "Utils/Math/MathConstants.slangh"

import Utils.Math.AABB;


struct VertexInfo
{
    uint4 hitData;
    float3 origin;
    float pdfFwd = 0;
    float3 dir;
    float pdfRev = 0;
    float3 beta;
    uint flagAndType = 0;
    float3 originN;
    float de = 0;
    float pe = 0;
    uint prevIndex;
}

StructuredBuffer<VertexInfo> LightPathsVertexsBuffer;
StructuredBuffer<float4> LightPathsVertexsPositionBuffer;
RWStructuredBuffer<float4> gPhotonFlux[2];
RWStructuredBuffer<float4> gPhotonDir[2];
RWStructuredBuffer<AABB> gPhotonAABB[2];
RWTexture2D<uint> hashBuffer;

//static const uint kInfoTexHeight = INFO_TEXTURE_HEIGHT;

cbuffer CSConstants
{
    uint num;
    uint gHashSize;         // Size of the hash
    uint gYExtent;
    float gGlobalRadius;
};

#define DEFAULT_BLOCK_SIZE 512
[numthreads(DEFAULT_BLOCK_SIZE, 1, 1)]
void main(uint3 DTid: SV_DispatchThreadID) {
    uint index = DTid.x;
    if (index > num) return;

    VertexInfo vInfo = LightPathsVertexsBuffer[index];
    uint flag = vInfo.flagAndType;

    if ((flag & 0x0004) == 0) return; // invalid
    if ((flag & 0x0003) == 1) return; // light

    float3 pos = LightPathsVertexsPositionBuffer[index].xyz;

    if (!cullingTest(pos)) return; // culling 

    bool isCaustic = (flag & 0x0020) == 0;
    uint type = isCaustic ? 0 : 1;
    uint photonIndex = gPhotonFlux[type].IncrementCounter();

    gPhotonFlux[type][photonIndex] = float4(vInfo.beta, 0);
    gPhotonDir[type][photonIndex] = float4(vInfo.dir, 0);
    gPhotonAABB[type][photonIndex] = AABB(pos - gGlobalRadius, pos + gGlobalRadius);
}

bool cullingTest(float3 origin)
{
    // Check if position is inside camera frustum
    const float4 pos = float4(origin, 1);

    // Check if hash cell is set
    int3 cell = int3(floor(origin / gGlobalRadius * 0.5f));
    uint h = hash(cell) & (gHashSize - 1);
    uint2 hIdx = uint2(h % gYExtent, h / gYExtent);

    return hashBuffer[hIdx] == 1 ? false : true;
}