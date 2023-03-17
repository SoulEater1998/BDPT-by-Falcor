#pragma once
#include "Falcor.h"

using namespace Falcor;

struct Bitonic64Sort
{
    Bitonic64Sort(Buffer::SharedPtr& _KeyIndexList, uint _listCount);
    void sort(RenderContext* pRenderContext);
    void setListCount(uint _listCount) { listCount = _listCount; };
    //Buffer::SharedPtr

private:
    uint ElementSizeBytes = sizeof(uint64_t);
    uint listCount;

    Buffer::SharedPtr KeyIndexList;
    Buffer::SharedPtr DispatchArgs;

    ComputePass::SharedPtr BitonicArgsCS;
    ComputePass::SharedPtr Bitonic64PreSortCS;
    ComputePass::SharedPtr Bitonic64OuterSortCS;
    ComputePass::SharedPtr Bitonic64InnerSortCS;
};
