#pragma once
#include "Falcor.h"

using namespace Falcor;

struct Node
{
    float3 boundMin;
    float betaSum;
    float3 boundMax;
    uint ID;
};

struct VertexTreeBuilder
{
    VertexTreeBuilder(Buffer::SharedPtr _KeyIndexBuffer, Buffer::SharedPtr _VertexBuffer, uint maxCounter);
    void update(uint counter, uint frame,
        Buffer::SharedPtr _KeyIndexBuffer,
        Buffer::SharedPtr _PosAndIntensityBuffer,
        Buffer::SharedPtr _targetKeyIndexBuffer,
        Buffer::SharedPtr _NewPosAndIntensityBuffer,
        Buffer::SharedPtr _NewLightPathsVertexsBuffer,
        Buffer::SharedPtr _LightPathsVertexsBuffer);
    void build(RenderContext* pRenderContext);
    void endFrame(RenderContext* pRenderContext);

    Buffer::SharedPtr getTree() { return Nodes; };
    uint getLeafNodeStartIndex() { return leafNodesNum; };
    uint getPrevCounter() { return prevCounter; };
    uint getAccumulativeCounter() { return std::min(accumulativeCounter, 20 * realLeafNodesNum); };

    float getTotalBetaSum() {
        auto kNodesPtr = (float*)Nodes->map(Buffer::MapType::Read);
        float counter = *(kNodesPtr + 12);
        return counter;
    }

private:
    void GenLevelZero(RenderContext* pRenderContext);
    void GenInternalLevel(RenderContext* pRenderContext);

    Buffer::SharedPtr Nodes;
    Buffer::SharedPtr KeyIndexBuffer;
    Buffer::SharedPtr VertexBuffer;

    Buffer::SharedPtr PrevNodes;

    uint realLeafNodesNum;
    uint leafNodesNum;
    uint nodesNum;
    uint treeLevels;

    uint prevCounter;

    uint accumulativeCounter;

    ComputePass::SharedPtr GenLevelZeroCS;
    ComputePass::SharedPtr GenInternalLevelCS;
};
