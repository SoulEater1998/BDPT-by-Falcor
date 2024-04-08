// Copyright (c) 2022, Fengqi Liu <M202173624@hust.edu.cn>
// All rights reserved.
// This code is licensed under the MIT License (MIT).

#define DEFAULT_BLOCK_SIZE 512
import Node;

cbuffer CSConstants : register(b0)
{
    int srcLevel;
    int dstLevelStart;
    int dstLevelEnd;
    int numLevels;
    int numDstLevelsNodes;
};

// todo: add dstLevelStart and dstLevelEnd

RWStructuredBuffer<Node> nodes;

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

[numthreads(DEFAULT_BLOCK_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    int nodeId = DTid.x;
    if (nodeId < numDstLevelsNodes)
    {
        int dstNodeArr, dstLevel;
        if (dstLevelEnd == dstLevelStart + 1)
        {
            dstLevel = dstLevelStart;
            dstNodeArr = (1 << (numLevels - dstLevel - 1)) + nodeId;
        }
        else
        {
            int offset = 1 << (numLevels - dstLevelEnd);
            dstNodeArr = offset + nodeId;
            dstLevel = numLevels - 1 - uintLog2(dstNodeArr);
        }

        int startNodeId = dstNodeArr << (dstLevel - srcLevel);
        int endNodeId = startNodeId + (1 << (dstLevel - srcLevel));

        Node node = nodes[startNodeId];
        float maximumBeta = 0;

        for (int nodeId = startNodeId + 1; nodeId < endNodeId; nodeId++)
        {
            Node srcNode = nodes[nodeId];
            
            if (srcNode.betaSum > 0) //valid Vertex
            {
                node.ID = (srcNode.betaSum > maximumBeta) ? srcNode.ID : node.ID;
                node.betaSum += srcNode.betaSum;
                node.boundMin = min(srcNode.boundMin, node.boundMin);
                node.boundMax = max(srcNode.boundMax, node.boundMax);
                node.mortonRange.y = nodes[nodeId].mortonRange.y;
            }
        }
        
        nodes[dstNodeArr] = node;
    }
}