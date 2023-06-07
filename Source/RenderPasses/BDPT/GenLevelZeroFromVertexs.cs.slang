// Copyright (c) 2022, Fengqi Liu <M202173624@hust.edu.cn>
// All rights reserved.
// This code is licensed under the MIT License (MIT).

#define DEFAULT_BLOCK_SIZE 512

import Node;

cbuffer CSConstants
{
    int numLeafNodes;
    int numLevels;
    int counter;
};

StructuredBuffer<uint2> keyIndexList;
StructuredBuffer<float4> posAndIntensityBuffer;
RWStructuredBuffer<Node> nodes;


[numthreads(DEFAULT_BLOCK_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    int levelNodeId = DTid.x;

    if (levelNodeId < numLeafNodes)
    {
        Node node;
        int nodeArr = (1 << (numLevels - 1)) + levelNodeId;

        if (levelNodeId < counter)
        {
            uint2 KeyIndexPair = keyIndexList[levelNodeId];
            uint index = KeyIndexPair.x;
            float4 posAndIntensity = posAndIntensityBuffer[index];
            node.ID = index;
			// For 16 bit version this doesn't matter (if we really need to construct the similar invalid bound it 
			// needs to be corner + (1+eps)dimension)
            float3 boundMin = 1e10;
            float3 boundMax = -1e10;
            node.betaSum = posAndIntensity.w;

            if (node.betaSum > 0) //valid vertex
            {
                boundMin = posAndIntensity.xyz;
            }

            node.boundMax = node.boundMin = boundMin;
        }
        else
        {
            node.ID = 0xffffffff;
            node.betaSum = 0;
            node.boundMin = 1e10;
            node.boundMax = -1e10;
        }

        nodes[nodeArr] = node;
    }
}
