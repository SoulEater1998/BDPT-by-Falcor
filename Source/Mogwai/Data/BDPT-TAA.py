from falcor import *

def render_graph_BDPT():
    g = RenderGraph('BDPT')
    loadRenderPassLibrary('AccumulatePass.dll')
    loadRenderPassLibrary('BSDFViewer.dll')
    loadRenderPassLibrary('Antialiasing.dll')
    loadRenderPassLibrary('BlitPass.dll')
    loadRenderPassLibrary('SimplePostFX.dll')
    loadRenderPassLibrary('FLIPPass.dll')
    loadRenderPassLibrary('BDPT.dll')
    loadRenderPassLibrary('CSM.dll')
    loadRenderPassLibrary('DebugPasses.dll')
    #loadRenderPassLibrary('PathTracer.dll')
    loadRenderPassLibrary('DepthPass.dll')
    loadRenderPassLibrary('NRC.dll')
    loadRenderPassLibrary('DLSSPass.dll')
    loadRenderPassLibrary('ErrorMeasurePass.dll')
    loadRenderPassLibrary('ForwardLightingPass.dll')
    loadRenderPassLibrary('GBuffer.dll')
    loadRenderPassLibrary('WhittedRayTracer.dll')
    loadRenderPassLibrary('ImageLoader.dll')
    loadRenderPassLibrary('MinimalPathTracer.dll')
    loadRenderPassLibrary('ModulateIllumination.dll')
    loadRenderPassLibrary('NRDPass.dll')
    loadRenderPassLibrary('PixelInspectorPass.dll')
    loadRenderPassLibrary('SkyBox.dll')
    loadRenderPassLibrary('RTXDIPass.dll')
    loadRenderPassLibrary('RTXGIPass.dll')
    loadRenderPassLibrary('SceneDebugger.dll')
    loadRenderPassLibrary('SDFEditor.dll')
    loadRenderPassLibrary('SSAO.dll')
    loadRenderPassLibrary('SVGFPass.dll')
    loadRenderPassLibrary('TemporalDelayPass.dll')
    loadRenderPassLibrary('TestPasses.dll')
    loadRenderPassLibrary('ToneMapper.dll')
    loadRenderPassLibrary('Utils.dll')
    BDPT = createPass('BDPT', {'samplesPerPixel': 1, 'maxSurfaceBounces': 5, 'maxDiffuseBounces': 3, 'maxSpecularBounces': 3, 'maxTransmissionBounces': 5, 'sampleGenerator': 0, 'useBSDFSampling': True, 'useRussianRoulette': False, 'useNEE': True, 'useMIS': True, 'misHeuristic': MISHeuristic.Balance, 'misPowerExponent': 2.0, 'emissiveSampler': EmissiveLightSamplerType.LightBVH, 'lightBVHOptions': LightBVHSamplerOptions(buildOptions=LightBVHBuilderOptions(splitHeuristicSelection=SplitHeuristic.BinnedSAOH, maxTriangleCountPerLeaf=10, binCount=16, volumeEpsilon=0.0010000000474974513, splitAlongLargest=False, useVolumeOverSA=False, useLeafCreationCost=True, createLeavesASAP=True, allowRefitting=True, usePreintegration=True, useLightingCones=True), useBoundingCone=True, useLightingCone=True, disableNodeFlux=False, useUniformTriangleSampling=True, solidAngleBoundMethod=SolidAngleBoundMethod.Sphere), 'useRTXDI': False, 'RTXDIOptions': RTXDIOptions(mode=RTXDIMode.SpatiotemporalResampling, presampledTileCount=128, presampledTileSize=1024, storeCompactLightInfo=True, localLightCandidateCount=24, infiniteLightCandidateCount=8, envLightCandidateCount=8, brdfCandidateCount=1, brdfCutoff=0.0, testCandidateVisibility=True, biasCorrection=RTXDIBiasCorrection.Basic, depthThreshold=0.10000000149011612, normalThreshold=0.5, samplingRadius=30.0, spatialSampleCount=1, spatialIterations=5, maxHistoryLength=20, boilingFilterStrength=0.0, rayEpsilon=0.0010000000474974513, useEmissiveTextures=False, enableVisibilityShortcut=False, enablePermutationSampling=False), 'useAlphaTest': True, 'adjustShadingNormals': False, 'maxNestedMaterials': 2, 'useLightsInDielectricVolumes': False, 'disableCaustics': False, 'specularRoughnessThreshold': 0.25, 'primaryLodMode': TexLODMode.Mip0, 'lodBias': 0.0, 'outputSize': IOSize.Default, 'colorFormat': ColorFormat.LogLuvHDR})
    g.addPass(BDPT, 'BDPT')
    VBufferRT = createPass('VBufferRT', {'outputSize': IOSize.Default, 'samplePattern': SamplePattern.Center, 'sampleCount': 16, 'useAlphaTest': True, 'adjustShadingNormals': True, 'forceCullMode': False, 'cull': CullMode.CullBack, 'useTraceRayInline': False, 'useDOF': True})
    g.addPass(VBufferRT, 'VBufferRT')
    ToneMapper = createPass('ToneMapper', {'outputSize': IOSize.Default, 'useSceneMetadata': True, 'exposureCompensation': 0.0, 'autoExposure': False, 'filmSpeed': 100.0, 'whiteBalance': False, 'whitePoint': 6500.0, 'operator': ToneMapOp.Aces, 'clamp': True, 'whiteMaxLuminance': 1.0, 'whiteScale': 11.199999809265137, 'fNumber': 1.0, 'shutter': 1.0, 'exposureMode': ExposureMode.AperturePriority})
    g.addPass(ToneMapper, 'ToneMapper')
    TAA = createPass('TAA', {'alpha': 0.10000000149011612, 'colorBoxSigma': 1.0})
    g.addPass(TAA, 'TAA')
    AccumulatePass = createPass('AccumulatePass', {'enabled': True, 'outputSize': IOSize.Default, 'autoReset': True, 'precisionMode': AccumulatePrecision.Single, 'subFrameCount': 0, 'maxAccumulatedFrames': 0})
    g.addPass(AccumulatePass, 'AccumulatePass')
    g.addEdge('AccumulatePass.output', 'TAA.colorIn')
    g.addEdge('VBufferRT.mvec', 'TAA.motionVecs')
    g.addEdge('TAA.colorOut', 'ToneMapper.src')
    g.addEdge('VBufferRT.vbuffer', 'BDPT.vbuffer')
    g.addEdge('VBufferRT.mvec', 'BDPT.mvec')
    g.addEdge('VBufferRT.viewW', 'BDPT.viewW')
    g.addEdge('BDPT.color', 'AccumulatePass.input')
    g.markOutput('ToneMapper.dst')
    return g

BDPT = render_graph_BDPT()
try: m.addGraph(BDPT)
except NameError: None