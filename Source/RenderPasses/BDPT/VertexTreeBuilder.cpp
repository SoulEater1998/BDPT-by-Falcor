#include "VertexTreeBuilder.h"


namespace
{
    const std::string kGenInternalLevelFilename = "RenderPasses/BDPT/GenInternalLevel.cs.slang";
    const std::string kGenLevelZeroFromVertexsFilename = "RenderPasses/BDPT/GenLevelZeroFromVertexs.cs.slang";
}

VertexTreeBuilder::VertexTreeBuilder(Buffer::SharedPtr _KeyIndexBuffer, Buffer::SharedPtr _VertexBuffer, uint maxCounter) :KeyIndexBuffer(_KeyIndexBuffer), VertexBuffer(_VertexBuffer) {
    uint maxTreeLevels = uint(ceil(log2((double)maxCounter))) + 1;
    uint maxLeafNodesNum = 1 << (maxTreeLevels - 1);
    uint maxNodesNum = maxLeafNodesNum << 1;

    Nodes = Buffer::createStructured(sizeof(Node), maxNodesNum);
    PrevNodes = Buffer::createStructured(sizeof(Node), maxNodesNum);

    GenLevelZeroCS = ComputePass::create(kGenLevelZeroFromVertexsFilename, "main");
    //GenLevelZeroCS->getProgram()->setDefines(defines);
    GenInternalLevelCS = ComputePass::create(kGenInternalLevelFilename, "main");

    //GenLevelZeroCS["keyIndexList"] = _KeyIndexBuffer;
    //GenLevelZeroCS["posAndIntensityBuffer"] = _VertexBuffer;
    //GenLevelZeroCS["nodes"] = Nodes;

    //GenInternalLevelCS["nodes"] = Nodes;
    prevCounter = 0;
    accumulativeCounter = 0;
}

void VertexTreeBuilder::update(uint counter, uint frame,
                            Buffer::SharedPtr _KeyIndexBuffer,
                            Buffer::SharedPtr _PosAndIntensityBuffer,
                            Buffer::SharedPtr _targetKeyIndexBuffer,
                            Buffer::SharedPtr _NewPosAndIntensityBuffer,
                            Buffer::SharedPtr _NewLightPathsVertexsBuffer,
                            Buffer::SharedPtr _LightPathsVertexsBuffer) {
    realLeafNodesNum = (counter + prevCounter + 1) / 2;
    leafNodesNum = realLeafNodesNum == 0 ? 0 : 1 << int(ceil(log2((double)realLeafNodesNum)));
    treeLevels = uint(ceil(log2((double)realLeafNodesNum))) + 1;
    nodesNum = leafNodesNum << 1;
    accumulativeCounter += counter;
    //accumulativeCounter = std::max(accumulativeCounter, 20 * counter);

    GenLevelZeroCS["keyIndexList"] = _KeyIndexBuffer;
    GenLevelZeroCS["targetKeyIndexList"] = _targetKeyIndexBuffer;
    GenLevelZeroCS["NewPosAndIntensityBuffer"] = _NewPosAndIntensityBuffer;
    GenLevelZeroCS["PosAndIntensityBuffer"] = _PosAndIntensityBuffer;
    GenLevelZeroCS["NewLightPathsVertexsBuffer"] = _NewLightPathsVertexsBuffer;
    GenLevelZeroCS["LightPathsVertexsBuffer"] = _LightPathsVertexsBuffer;
    GenLevelZeroCS["CSConstants"]["frame"] = frame;
    
}

void VertexTreeBuilder::GenLevelZero(RenderContext* pRenderContext) {
    GenLevelZeroCS["CSConstants"]["numLeafNodes"] = leafNodesNum;
    GenLevelZeroCS["CSConstants"]["numLevels"] = treeLevels;
    GenLevelZeroCS["CSConstants"]["counter"] = realLeafNodesNum;
    GenLevelZeroCS["CSConstants"]["scaling"] = (accumulativeCounter < 20 * realLeafNodesNum) ? 1 : (float(20 * realLeafNodesNum) / accumulativeCounter);
    accumulativeCounter = std::min(accumulativeCounter, 20 * realLeafNodesNum);
    GenLevelZeroCS->execute(pRenderContext, leafNodesNum, 1);
}

void VertexTreeBuilder::GenInternalLevel(RenderContext* pRenderContext) {
   
    const uint maxWorkLoad = 2048;
    int srcLevel = 0;
    for (int dstLevelStart = 1; dstLevelStart < treeLevels; )
    {
        int dstLevelEnd;
        int workLoad = 0;
        for (dstLevelEnd = dstLevelStart + 1; dstLevelEnd < treeLevels; dstLevelEnd++)
        {
            workLoad += 1 << (treeLevels - 1 - srcLevel);
            if (workLoad > maxWorkLoad) break;
        }
        pRenderContext->uavBarrier(Nodes.get());
        GenInternalLevelCS["CSConstants"]["srcLevel"] = srcLevel;
        GenInternalLevelCS["CSConstants"]["dstLevelStart"] = dstLevelStart;
        GenInternalLevelCS["CSConstants"]["dstLevelEnd"] = dstLevelEnd;
        GenInternalLevelCS["CSConstants"]["numLevels"] = treeLevels;
        int numDstLevelsLights = (1 << (treeLevels - dstLevelStart)) - (1 << (treeLevels - dstLevelEnd));
        GenInternalLevelCS["CSConstants"]["numDstLevelsNodes"] = numDstLevelsLights;
        GenInternalLevelCS->execute(pRenderContext, numDstLevelsLights, 1);
        //GenerateMultipleLevels(cptContext, srcLevel, dstLevelStart, dstLevelEnd, nodes.GetUAV(), numLevels);

        srcLevel = dstLevelEnd - 1;
        dstLevelStart = dstLevelEnd;
    }
}

void VertexTreeBuilder::build(RenderContext* pRenderContext) {
    //GenLevelZeroCS["gScene"] = gScene;
    //GenInternalLevelCS["gScene"] = gScene;
    pRenderContext->uavBarrier(KeyIndexBuffer.get());

    
    GenLevelZeroCS["nodes"] = Nodes;
    GenLevelZeroCS["prevNodes"] = PrevNodes;
    GenInternalLevelCS["nodes"] = Nodes;

    GenLevelZero(pRenderContext);
    GenInternalLevel(pRenderContext);
}

void VertexTreeBuilder::endFrame(RenderContext* pRenderContext) {
    std::swap(Nodes, PrevNodes);
    prevCounter = realLeafNodesNum;
}
