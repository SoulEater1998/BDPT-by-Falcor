// Copyright (c) 2022, Fengqi Liu <M202173624@hust.edu.cn>
// All rights reserved.
// This code is licensed under the MIT License (MIT).

#pragma once
#include "Falcor.h"

using namespace Falcor;

struct MordenCodeSort
{
    MordenCodeSort(Buffer::SharedPtr& _positionBuffer, Buffer::SharedPtr& _resultBuffer, uint _positionBufferUpbound);
    void execute(RenderContext* pRenderContext);
    Buffer::SharedPtr getBound() { return positionBound; }
    uint getPositionBufferCount() { return positionBufferCount; };
    //void Sort(RenderContext* pRenderContext);
    //void SortTest(RenderContext* pRenderContext);

private:
    void FindBoundingBox(RenderContext* pRenderContext);
    void GenerateMordenCode(RenderContext* pRenderContext);
    uint3 GetDimension(uint threadNum);

    Buffer::SharedPtr KeyIndexList;
    Buffer::SharedPtr positionBound;
    Buffer::SharedPtr positionBuffer;
    

    Buffer::SharedPtr bboxReductionBuffer[2];
    
    uint positionBufferCount;
    uint positionBufferUpbound;
    uint groupSize = 1024;

    ComputePass::SharedPtr GenMordenCode;
    ComputePass::SharedPtr FindMax;
    ComputePass::SharedPtr FindMin;
    
    //ComputePass::SharedPtr SortTestCS;

    

    float4 corner;
    float4 dim;
};

