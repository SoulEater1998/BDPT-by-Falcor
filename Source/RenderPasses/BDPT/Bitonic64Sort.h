#pragma once
#include "Falcor.h"

using namespace Falcor;

struct Bitonic64Sort
{
    Bitonic64Sort(Buffer::SharedPtr& _KeyIndexList, uint _listCount);
    void sort(RenderContext* pRenderContext);
    void setListCount(uint _listCount, Buffer::SharedPtr& _KeyIndexList) {
        listCount = _listCount;
        Bitonic64PreSortCS["g_SortBuffer"] = _KeyIndexList;
        Bitonic64OuterSortCS["g_SortBuffer"] = _KeyIndexList;
        Bitonic64InnerSortCS["g_SortBuffer"] = _KeyIndexList;
        SortTestCS["keyIndexList"] = _KeyIndexList;
        KeyIndexList = _KeyIndexList;
    };
    void sortTest(RenderContext* pRenderContext, Texture::SharedPtr& _output, uint2 frameDim);
    //Buffer::SharedPtr

private:
    uint ElementSizeBytes = sizeof(uint64_t);
    uint listCount;
    //uint prevListCount;

    Buffer::SharedPtr KeyIndexList;
    Buffer::SharedPtr DispatchArgs;

    ComputePass::SharedPtr BitonicArgsCS;
    ComputePass::SharedPtr Bitonic64PreSortCS;
    ComputePass::SharedPtr Bitonic64OuterSortCS;
    ComputePass::SharedPtr Bitonic64InnerSortCS;
    ComputePass::SharedPtr SortTestCS;
    ComputePass::SharedPtr MapCS;
};
