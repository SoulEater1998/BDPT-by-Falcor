#pragma once
#include "Falcor.h"

using namespace Falcor;

struct MordenCodeSort
{
    MordenCodeSort(Buffer::SharedPtr& _positionBuffer, uint _positionBufferUpbound);
    void execute(RenderContext* pRenderContext);
    Buffer::SharedPtr getResult() { return KeyIndexList; }
    Buffer::SharedPtr getBound() { return positionBound; }
    //void Sort(RenderContext* pRenderContext);
    //void SortTest(RenderContext* pRenderContext);

private:
    void FindBoundingBox(RenderContext* pRenderContext);
    void GenerateMordenCode(RenderContext* pRenderContext);

    Buffer::SharedPtr KeyIndexList;
    Buffer::SharedPtr positionBound;
    Buffer::SharedPtr positionBuffer;
    

    Buffer::SharedPtr bboxReductionBuffer[2];
    
    uint positionBufferCount;
    uint positionBufferUpbound;

    ComputePass::SharedPtr GenMordenCode;
    ComputePass::SharedPtr FindMax;
    ComputePass::SharedPtr FindMin;
    
    //ComputePass::SharedPtr SortTestCS;

    

    float4 corner;
    float4 dim;
};

