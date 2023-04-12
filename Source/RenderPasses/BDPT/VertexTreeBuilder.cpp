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

    GenLevelZeroCS = ComputePass::create(kGenLevelZeroFromVertexsFilename, "main");
    //GenLevelZeroCS->getProgram()->setDefines(defines);
    GenInternalLevelCS = ComputePass::create(kGenInternalLevelFilename, "main");

    GenLevelZeroCS["keyIndexList"] = _KeyIndexBuffer;
    GenLevelZeroCS["posAndIntensityBuffer"] = _VertexBuffer;
    GenLevelZeroCS["nodes"] = Nodes;

    GenInternalLevelCS["nodes"] = Nodes;
}

void VertexTreeBuilder::update(uint counter) {
    realLeafNodesNum = counter;
    leafNodesNum = counter == 0 ? 0 : 1 << int(ceil(log2((double)counter)));
    treeLevels = uint(ceil(log2((double)counter))) + 1;
    nodesNum = leafNodesNum << 1;
}

void VertexTreeBuilder::GenLevelZero(RenderContext* pRenderContext) {
    GenLevelZeroCS["CSConstants"]["numLeafNodes"] = leafNodesNum;
    GenLevelZeroCS["CSConstants"]["numLevels"] = treeLevels;
    GenLevelZeroCS["CSConstants"]["counter"] = realLeafNodesNum;
    
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
    GenLevelZero(pRenderContext);
    GenInternalLevel(pRenderContext);
}
