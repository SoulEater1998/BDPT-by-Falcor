#include "Bitonic64Sort.h"

namespace
{
    const std::string kBitonicIndirectArgsFilename = "RenderPasses/BDPT/BitonicIndirectArgsCS.slang";
    const std::string kBitonic64PreSortFilename = "RenderPasses/BDPT/Bitonic64PreSortCS.slang";
    const std::string kBitonic64OuterSortFilename = "RenderPasses/BDPT/Bitonic64OuterSortCS.slang";
    const std::string kBitonic64InnerSortFilename = "RenderPasses/BDPT/Bitonic64InnerSortCS.slang";
}

Bitonic64Sort::Bitonic64Sort(Buffer::SharedPtr& _KeyIndexList, uint _listCount) :KeyIndexList(_KeyIndexList), listCount(_listCount) {
    ElementSizeBytes = sizeof(uint64_t);

    BitonicArgsCS = ComputePass::create(kBitonicIndirectArgsFilename, "main");
    Bitonic64PreSortCS = ComputePass::create(kBitonic64PreSortFilename, "main");
    Bitonic64OuterSortCS = ComputePass::create(kBitonic64OuterSortFilename, "main");
    Bitonic64InnerSortCS = ComputePass::create(kBitonic64InnerSortFilename, "main");

    DispatchArgs = Buffer::createStructured(sizeof(uint3), 22 * 23 / 2, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::IndirectArg);
    //Sort
    BitonicArgsCS["g_IndirectArgsBuffer"] = DispatchArgs;
    Bitonic64PreSortCS["g_SortBuffer"] = KeyIndexList;
    Bitonic64OuterSortCS["g_SortBuffer"] = KeyIndexList;
    Bitonic64InnerSortCS["g_SortBuffer"] = KeyIndexList;
}

void Bitonic64Sort::sort(RenderContext* pRenderContext) {
    FALCOR_ASSERT(listCount != 0);

    const uint MaxNumElements = 1 << int(ceil(log2((double)listCount)));
    const uint AlignedMaxNumElements = MaxNumElements;
    const uint MaxIterations = int(ceil(log2(std::max(2048u, AlignedMaxNumElements)))) - 10;

    FALCOR_ASSERT(ElementSizeBytes == 4 || ElementSizeBytes == 8);

    BitonicArgsCS["Constants"]["MaxIterations"] = MaxIterations;
    BitonicArgsCS["Constants"]["constListCount"] = listCount;
    
    BitonicArgsCS->execute(pRenderContext, uint3(22, 1, 1));

    Bitonic64PreSortCS["Constants"]["constListCount"] = listCount;

    pRenderContext->uavBarrier(KeyIndexList.get());
    Bitonic64PreSortCS->executeIndirect(pRenderContext, DispatchArgs.get());
    
    uint IndirectArgsOffset = 12;

    Bitonic64OuterSortCS["Constants"]["constListCount"] = listCount;
    Bitonic64InnerSortCS["Constants"]["constListCount"] = listCount;

    for (uint k = 4096; k <= AlignedMaxNumElements; k *= 2) {
        for (uint j = k / 2; j >= 2048; j /= 2) {
            Bitonic64OuterSortCS["Constants"]["k"] = k;
            Bitonic64OuterSortCS["Constants"]["j"] = j;

            pRenderContext->uavBarrier(KeyIndexList.get());
            Bitonic64OuterSortCS->executeIndirect(pRenderContext, DispatchArgs.get(), IndirectArgsOffset);
            
            IndirectArgsOffset += 12;
        }

        pRenderContext->uavBarrier(KeyIndexList.get());
        Bitonic64InnerSortCS->executeIndirect(pRenderContext, DispatchArgs.get(), IndirectArgsOffset);
        //pRenderContext->uavBarrier(KeyIndexList.get());
        IndirectArgsOffset += 12;
    }
}
