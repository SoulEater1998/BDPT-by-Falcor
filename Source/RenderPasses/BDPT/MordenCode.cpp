#include "MordenCode.h"

static const float4 kClearColor(0.0f, 0.0f, 0.0f, 0.0f);
static const uint zero = 0;

namespace
{
    const std::string kFindMaxPositionFilename = "RenderPasses/BDPT/FindMaxPosition.cs.slang";
    const std::string kFindMinPositionFilename = "RenderPasses/BDPT/FindMinPosition.cs.slang";
    const std::string kGenMordenCodeFilename = "RenderPasses/BDPT/GenMordenCode.cs.slang";

    const std::string kBitonicIndirectArgsFilename = "RenderPasses/BDPT/BitonicIndirectArgsCS.slang";
    const std::string kBitonic64PreSortFilename = "RenderPasses/BDPT/Bitonic64PreSortCS.slang";
    const std::string kBitonic64OuterSortFilename = "RenderPasses/BDPT/Bitonic64OuterSortCS.slang";
    const std::string kBitonic64InnerSortFilename = "RenderPasses/BDPT/Bitonic64InnerSortCS.slang";

}

MordenCodeSort::MordenCodeSort(Buffer::SharedPtr& _positionBuffer, uint _positionBufferUpbound) : positionBuffer(_positionBuffer), positionBufferUpbound(_positionBufferUpbound){
    positionBufferCount = 0;

    FindMax = ComputePass::create(kFindMaxPositionFilename, "main");
    FindMin = ComputePass::create(kFindMinPositionFilename, "main");
    GenMordenCode = ComputePass::create(kGenMordenCodeFilename, "main");

    positionBound = Buffer::createStructured(sizeof(float4), 2);
    KeyIndexList = Buffer::createStructured(sizeof(uint64_t), positionBufferUpbound);

    bboxReductionBuffer[0] = Buffer::createStructured(sizeof(float4), positionBufferUpbound);
    bboxReductionBuffer[1] = Buffer::createStructured(sizeof(float4), positionBufferUpbound);

    FindMax["SceneBoundBuffer"] = positionBound;
    FindMin["SceneBoundBuffer"] = positionBound;

    //Morden Code
    GenMordenCode["Positions"] = positionBuffer;
    GenMordenCode["keyIndexList"] = KeyIndexList;
    GenMordenCode["Bound"] = positionBound;

}

void MordenCodeSort::execute(RenderContext* pRenderContext) {
    pRenderContext->uavBarrier(positionBuffer.get());
    pRenderContext->uavBarrier(positionBuffer->getUAVCounter().get());
    auto temp = (uint*)positionBuffer->map(Buffer::MapType::Read);
    positionBufferCount = *temp;

    FindBoundingBox(pRenderContext);

    GenerateMordenCode(pRenderContext);
}

void MordenCodeSort::FindBoundingBox(RenderContext* pRenderContext) {
    

    int numGroups = (positionBufferCount + 2047) / 2048;
    bool largeNum = numGroups > 2048;

    FindMax["consts"]["n"] = positionBufferCount;
    FindMax["consts"]["isLastPass"] = positionBufferCount <= 2048;
    FindMax["g_data"] = positionBuffer;
    FindMax["g_odata"] = bboxReductionBuffer[1];

    pRenderContext->uavBarrier(bboxReductionBuffer[1].get());
    FindMax->execute(pRenderContext, uint3(positionBufferCount, 1, 1));
    if (positionBufferCount > 2048) {
        FindMax["consts"]["n"] = numGroups;
        FindMax["consts"]["isLastPass"] = !largeNum;
        FindMax["g_data"] = bboxReductionBuffer[1];
        FindMax["g_odata"] = bboxReductionBuffer[0];

        pRenderContext->uavBarrier(bboxReductionBuffer[1].get());
        pRenderContext->uavBarrier(bboxReductionBuffer[0].get());
        FindMax->execute(pRenderContext, uint3(numGroups, 1, 1));
    }
    if (largeNum) {
        uint numGroups_2 = (numGroups + 2047) / 2048;
        FindMax["consts"]["n"] = numGroups_2;
        FindMax["consts"]["isLastPass"] = true;
        FindMax["g_data"] = bboxReductionBuffer[0];
        FindMax["g_odata"] = bboxReductionBuffer[1];

        pRenderContext->uavBarrier(bboxReductionBuffer[1].get());
        pRenderContext->uavBarrier(bboxReductionBuffer[0].get());
        FindMax->execute(pRenderContext, uint3(numGroups_2, 1, 1));
    }

    FindMin["consts"]["n"] = positionBufferCount;
    FindMin["consts"]["isLastPass"] = positionBufferCount <= 2048;
    FindMin["g_data"] = positionBuffer;
    FindMin["g_odata"] = bboxReductionBuffer[1];

    pRenderContext->uavBarrier(positionBound.get());
    pRenderContext->uavBarrier(bboxReductionBuffer[1].get());
    FindMin->execute(pRenderContext, uint3(positionBufferCount, 1, 1));
    if (positionBufferCount > 2048) {
        FindMin["consts"]["n"] = numGroups;
        FindMin["consts"]["isLastPass"] = !largeNum;
        FindMin["g_data"] = bboxReductionBuffer[1];
        FindMin["g_odata"] = bboxReductionBuffer[0];

        pRenderContext->uavBarrier(bboxReductionBuffer[1].get());
        pRenderContext->uavBarrier(bboxReductionBuffer[0].get());
        FindMin->execute(pRenderContext, uint3(numGroups, 1, 1));
    }
    if (largeNum) {
        uint numGroups_2 = (numGroups + 2047) / 2048;
        FindMin["consts"]["n"] = numGroups_2;
        FindMin["consts"]["isLastPass"] = true;
        FindMin["g_data"] = bboxReductionBuffer[0];
        FindMin["g_odata"] = bboxReductionBuffer[1];

        pRenderContext->uavBarrier(bboxReductionBuffer[1].get());
        pRenderContext->uavBarrier(bboxReductionBuffer[0].get());
        
        FindMin->execute(pRenderContext, uint3(numGroups_2, 1, 1));
    }

    //auto vplbound = (float4*)VPLsBound->map(Buffer::MapType::Read);
    //corner = vplbound[0];
    //dim = vplbound[1];
}

void MordenCodeSort::GenerateMordenCode(RenderContext* pRenderContext) {
    pRenderContext->clearUAV(KeyIndexList->getUAV().get(), kClearColor);
    
    uint quantLevels = 1 << 10;

    auto var1 = GenMordenCode["CSConstants"];
    var1["num"] = positionBufferCount;
    var1["quantLevels"] = quantLevels;

    pRenderContext->uavBarrier(positionBound.get());
    GenMordenCode->execute(pRenderContext, uint3(positionBufferCount, 1, 1));
}

