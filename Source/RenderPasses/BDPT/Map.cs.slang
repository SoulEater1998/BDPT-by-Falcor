// import PathTracer;
// import PathData;
//Texture2D<float4> outputColor;
Texture2D<uint> subspaceWeight;
Texture2D<uint> subspaceCount;
RWTexture2D<uint> subspaceSecondaryMoment;
RWTexture2D<uint> nextSubspaceWeight;
RWTexture2D<uint> totalCount;
RWTexture1D<float> sumWeight;
RWTexture2D<float> prefixSum;

RWTexture2D<float> prefixSumOfSecondaryMoment;
RWTexture1D<float> sumSecondaryMoment;

RWTexture1D<uint> sumCount;
RWTexture2D<uint> prefixSumOfCount;

RWTexture1D<float> maxVariance;
RWTexture2D<float4> output;

cbuffer CB {
    uint frameCount;
}


[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (all(pixel < uint2(4 * GROUP_SIZE, 4 * GROUP_SIZE))) {
        float weight = asfloat(subspaceWeight[pixel]);
        uint count = subspaceCount[pixel];
        if (count == 0 /*|| frameCount >= 200*/) {
            // output[pixel] = 0;
            // nextSubspaceWeight[pixel] = nextSubspaceWeight[pixel];
            // return;
            uint historyCount = totalCount[pixel];
            float totalWeight = asfloat(nextSubspaceWeight[pixel]) / historyCount;
            float secondaryMoment = asfloat(subspaceSecondaryMoment[pixel]) / historyCount;
            if (frameCount == 0) output[pixel].xyz = prefixSum[pixel] / sumWeight[pixel.y];//totalWeight * 0.333;//(secondaryMoment - totalWeight * totalWeight) * 0.111;
        }
        else {
            uint historyCount = totalCount[pixel];
            float totalWeight = (asfloat(nextSubspaceWeight[pixel]) /* * historyCount*/ + weight) / (historyCount + count);
            float secondaryMoment = asfloat(subspaceSecondaryMoment[pixel]) / (historyCount + count);
            //float totalWeight = min(asfloat(nextSubspaceWeight[pixel]) + weight, 100 * weight);
            uint nextCount = min(historyCount + count, 20 * count);
            subspaceSecondaryMoment[pixel] = asuint(secondaryMoment * nextCount);
            nextSubspaceWeight[pixel] = asuint(totalWeight * nextCount);
            totalCount[pixel] = nextCount;
            if (frameCount == 0) output[pixel].xyz = prefixSum[pixel] / sumWeight[pixel.y];//totalWeight * 0.333; //(secondaryMoment - totalWeight * totalWeight) * 0.111;//totalWeight * 0.333f;// prefixSum[pixel] / sumWeight[pixel.y];
        }
        //output[pixel].xyz = asfloat(nextSubspaceWeight[pixel]) * 0.333; //prefixSum[pixel] / sumWeight[pixel.y];//
        //output[pixel].w = 0;
        
    }
}

groupshared float gSharedData[2 * GROUP_SIZE];
groupshared float gSharedDataOfSecondaryMoment[2 * GROUP_SIZE];
groupshared uint gSharedDataOfCount[2 * GROUP_SIZE];
groupshared float gSharedDataOfMaxVariance[2 * GROUP_SIZE];


[numthreads(GROUP_SIZE, 1, 1)]
void scan(uint3 groupID: SV_GroupID, uint3 groupThreadID: SV_GroupThreadID)
{
    //if (frameCount > 200) return;
    
    const uint thid = groupThreadID.x; // Local thread ID in the range 0..N-1.
    const uint groupIdx = groupID.y;    // Group index where each group represents 2N elements.

    gSharedData[thid] = (thid < 2 * GROUP_SIZE ? asfloat(nextSubspaceWeight[uint2(thid, groupIdx)]) : 0);
    gSharedData[thid + GROUP_SIZE] = ((thid + GROUP_SIZE) < 2 * GROUP_SIZE ? asfloat(nextSubspaceWeight[uint2(thid + GROUP_SIZE, groupIdx)]) : 0);

    gSharedDataOfCount[thid] = (thid < 2 * GROUP_SIZE ? totalCount[uint2(thid, groupIdx)] : 0);
    gSharedDataOfCount[thid + GROUP_SIZE] = ((thid + GROUP_SIZE) < 2 * GROUP_SIZE ? totalCount[uint2(thid + GROUP_SIZE, groupIdx)] : 0);

    gSharedDataOfSecondaryMoment[thid] = (thid < 2 * GROUP_SIZE ? subspaceSecondaryMoment[uint2(thid, groupIdx)] : 0);
    gSharedDataOfSecondaryMoment[thid + GROUP_SIZE] = ((thid + GROUP_SIZE) < 2 * GROUP_SIZE ? subspaceSecondaryMoment[uint2(thid + GROUP_SIZE, groupIdx)] : 0);

    float expection1 = gSharedData[thid] / gSharedDataOfCount[thid];
    float secondaryMoment1 = gSharedDataOfSecondaryMoment[thid] / gSharedDataOfCount[thid];
    gSharedDataOfMaxVariance[thid] = secondaryMoment1 - expection1 * expection1;

    float expection2 = gSharedData[thid + GROUP_SIZE] / gSharedDataOfCount[thid + GROUP_SIZE];
    float secondaryMoment2 = gSharedDataOfSecondaryMoment[thid + GROUP_SIZE] / gSharedDataOfCount[thid + GROUP_SIZE];
    gSharedDataOfMaxVariance[thid + GROUP_SIZE] = secondaryMoment2 - expection2 * expection2;
    uint offset = 1;
    for (uint d = GROUP_SIZE; d > 0; d >>= 1)
    {
        GroupMemoryBarrierWithGroupSync();

        if (thid < d)
        {
            uint ai = offset * (2 * thid + 1) - 1;
            uint bi = ai + offset;

            gSharedData[bi] += gSharedData[ai];
            gSharedDataOfCount[bi] += gSharedDataOfCount[ai];
            gSharedDataOfSecondaryMoment[bi] += gSharedDataOfSecondaryMoment[ai];
            gSharedDataOfMaxVariance[bi] = max(gSharedDataOfMaxVariance[bi], gSharedDataOfMaxVariance[ai]);
        }
        offset *= 2; // offset = 1, 2, ... N
    }

    GroupMemoryBarrierWithGroupSync();

    if (thid == 0) {
        sumWeight[groupIdx] = gSharedData[2 * GROUP_SIZE - 1];
        sumCount[groupIdx] = gSharedDataOfCount[2 * GROUP_SIZE - 1];
        sumSecondaryMoment[groupIdx] = gSharedDataOfSecondaryMoment[2 * GROUP_SIZE - 1];
        maxVariance[groupIdx] = gSharedDataOfMaxVariance[2 * GROUP_SIZE - 1];

        gSharedData[2 * GROUP_SIZE - 1] = 0;
        gSharedDataOfCount[2 * GROUP_SIZE - 1] = 0;
        gSharedDataOfSecondaryMoment[2 * GROUP_SIZE - 1] = 0;
    }

    for (uint d = 1; d <= GROUP_SIZE; d *= 2)
    {
        offset >>= 1; // offset = N, N/2, ..., 1

        GroupMemoryBarrierWithGroupSync();

        if (thid < d)
        {
            uint ai = offset * (2 * thid + 1) - 1;
            uint bi = ai + offset;

            float tmp = gSharedData[ai];
            gSharedData[ai] = gSharedData[bi];
            gSharedData[bi] += tmp;

            uint tmpCount = gSharedDataOfCount[ai];
            gSharedDataOfCount[ai] = gSharedDataOfCount[bi];
            gSharedDataOfCount[bi] += tmpCount;

            float tmpSecondaryMoment = gSharedDataOfSecondaryMoment[ai];
            gSharedDataOfSecondaryMoment[ai] = gSharedDataOfSecondaryMoment[bi];
            gSharedDataOfSecondaryMoment[bi] += tmpSecondaryMoment;
        }
    }

    GroupMemoryBarrierWithGroupSync();

    if (thid < 2 * GROUP_SIZE) prefixSum[uint2(thid, groupIdx)] = gSharedData[thid];
    if ((thid + GROUP_SIZE) < 2 * GROUP_SIZE) prefixSum[uint2(thid + GROUP_SIZE, groupIdx)] = gSharedData[thid + GROUP_SIZE];

    if (thid < 2 * GROUP_SIZE) prefixSumOfCount[uint2(thid, groupIdx)] = gSharedDataOfCount[thid];
    if ((thid + GROUP_SIZE) < 2 * GROUP_SIZE) prefixSumOfCount[uint2(thid + GROUP_SIZE, groupIdx)] = gSharedDataOfCount[thid + GROUP_SIZE];

    if (thid < 2 * GROUP_SIZE) prefixSumOfSecondaryMoment[uint2(thid, groupIdx)] = gSharedDataOfSecondaryMoment[thid];
    if ((thid + GROUP_SIZE) < 2 * GROUP_SIZE) prefixSumOfSecondaryMoment[uint2(thid + GROUP_SIZE, groupIdx)] = gSharedDataOfSecondaryMoment[thid + GROUP_SIZE];

    //if (thid < gTotalNumElems) prefixSum[uint2(thid, groupIdx)] = 1;
    //if ((thid + GROUP_SIZE) < gTotalNumElems) prefixSum[uint2(thid + GROUP_SIZE, groupIdx)] = 1;
}
