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
    void update(uint counter, float m_radius);
    void build(RenderContext* pRenderContext);

    Buffer::SharedPtr getTree() { return Nodes; };
    uint getLeafNodeStartIndex() { return leafNodesNum; };

private:
    void GenLevelZero(RenderContext* pRenderContext);
    void GenInternalLevel(RenderContext* pRenderContext);

    Buffer::SharedPtr Nodes;
    Buffer::SharedPtr KeyIndexBuffer;
    Buffer::SharedPtr VertexBuffer;

    uint realLeafNodesNum;
    uint leafNodesNum = 0;
    uint nodesNum;
    uint treeLevels;
    float radius;

    ComputePass::SharedPtr GenLevelZeroCS;
    ComputePass::SharedPtr GenInternalLevelCS;
};
