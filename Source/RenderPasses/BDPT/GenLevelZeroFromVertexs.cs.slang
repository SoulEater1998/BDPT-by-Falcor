// Copyright (c) 2022, Fengqi Liu <M202173624@hust.edu.cn>
// All rights reserved.
// This code is licensed under the MIT License (MIT).

#define DEFAULT_BLOCK_SIZE 512
#define F 20.0

import Node;
//import PathData;
import Utils.Sampling.TinyUniformSampleGenerator;

cbuffer CSConstants
{
    int numLeafNodes;
    int numLevels;
    int counter;
    uint frame;
    float scaling;
};

struct VertexInfo
{
    uint4 hitData;
    float3 origin;
    float pdfFwd;
    float3 dir;
    float pdfRev;
    float3 beta;
    uint flagAndType;
    float de;
}

StructuredBuffer<uint2> keyIndexList;
RWStructuredBuffer<uint2> targetKeyIndexList;

StructuredBuffer<float4> NewPosAndIntensityBuffer;
RWStructuredBuffer<float4> PosAndIntensityBuffer;

StructuredBuffer<VertexInfo> NewLightPathsVertexsBuffer;
RWStructuredBuffer<VertexInfo> LightPathsVertexsBuffer;

StructuredBuffer<Node> prevNodes;
RWStructuredBuffer<Node> nodes;

Node getNode(uint2 keyIndexPair, out bool isPrev) {
    Node node;
    switch (keyIndexPair.y & 0x3) {
    case 0: // empty pair
        node.ID = 0xffffffff;
        node.betaSum = 0;
        node.boundMin = 1e10;
        node.boundMax = -1e10;
        node.pdf = 1;
        return node;
    case 1: // a new vertex
        uint index = keyIndexPair.x;
        float4 posAndIntensity = NewPosAndIntensityBuffer[index];
        node.ID = index;
        // For 16 bit version this doesn't matter (if we really need to construct the similar invalid bound it
        // needs to be corner + (1+eps)dimension)
        float3 boundMin = 1e10;
        float3 boundMax = -1e10;
        node.betaSum = posAndIntensity.w;

        if (node.betaSum > 0) // valid vertex
        {
            boundMin = posAndIntensity.xyz;
        }

        node.boundMax = node.boundMin = boundMin;
        node.pdf = 1;
        return node;
    default: // preview vertex
        isPrev = true;
        uint nodeIndex = keyIndexPair.x;
        node = prevNodes[nodeIndex];
        return node;
    }
}

[numthreads(DEFAULT_BLOCK_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    int levelNodeId = DTid.x;
    uint2 pixel = uint2(levelNodeId % 1024, levelNodeId / 1024);
    TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(pixel, frame);

    if (levelNodeId < numLeafNodes)
    {
        Node node;
        int nodeArr = (1 << (numLevels - 1)) + levelNodeId;

        if (levelNodeId < counter)
        {
            uint2 KeyIndexPairLeft = keyIndexList[2 * levelNodeId];
            uint2 KeyIndexPairRight = keyIndexList[2 * levelNodeId + 1];
            bool isPrev1 = false;
            bool isPrev2 = false;
            Node child1 = getNode(KeyIndexPairLeft, isPrev1);
            Node child2 = getNode(KeyIndexPairRight, isPrev2);
            bool isPrev = false;
            uint mortonCode = 0;
            float rnd = sampleNext1D(sg);

            // Generate node
            float betaSum = child1.betaSum + child2.betaSum;
            // TODO: BOX CLAMP
            node.betaSum = betaSum * scaling;
            node.boundMax = max(child1.boundMax, child2.boundMax);
            node.boundMin = min(child1.boundMin, child2.boundMin);
            node.ID = levelNodeId;

            uint updateID;
            float pdf = child1.betaSum / betaSum;
            if (rnd < pdf) {
                updateID = child1.ID;
                isPrev = isPrev1;
                mortonCode = KeyIndexPairLeft.y | 0x2;
                node.pdf = child1.pdf * pdf;
            }
            else {
                updateID = child2.ID;
                isPrev = isPrev2;
                mortonCode = KeyIndexPairRight.y | 0x2;
                node.pdf = child2.pdf * (1 - pdf);
            }

            // update vertex information
            LightPathsVertexsBuffer[levelNodeId] = NewLightPathsVertexsBuffer[updateID];
            targetKeyIndexList[levelNodeId] = uint2(nodeArr, mortonCode);
            PosAndIntensityBuffer[levelNodeId] = NewPosAndIntensityBuffer[updateID];
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
