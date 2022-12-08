from falcor import *

def render_graph_Realtime():
    g = RenderGraph('Realtime')
    loadRenderPassLibrary('SceneDebugger.dll')
    loadRenderPassLibrary('BSDFViewer.dll')
    loadRenderPassLibrary('OptixDenoiser.dll')
    loadRenderPassLibrary('AccumulatePass.dll')
    loadRenderPassLibrary('Antialiasing.dll')
    loadRenderPassLibrary('DepthPass.dll')
    loadRenderPassLibrary('SkyBox.dll')
    loadRenderPassLibrary('DebugPasses.dll')
    loadRenderPassLibrary('BlitPass.dll')
    loadRenderPassLibrary('CSM.dll')
    loadRenderPassLibrary('ErrorMeasurePass.dll')
    loadRenderPassLibrary('SimplePostFX.dll')
    loadRenderPassLibrary('FLIPPass.dll')
    loadRenderPassLibrary('ForwardLightingPass.dll')
    loadRenderPassLibrary('GBuffer.dll')
    loadRenderPassLibrary('ImageLoader.dll')
    loadRenderPassLibrary('MegakernelPathTracer.dll')
    loadRenderPassLibrary('TestPasses.dll')
    loadRenderPassLibrary('MinimalPathTracer.dll')
    loadRenderPassLibrary('ModulateIllumination.dll')
    loadRenderPassLibrary('NRDPass.dll')
    loadRenderPassLibrary('PassLibraryTemplate.dll')
    loadRenderPassLibrary('WhittedRayTracer.dll')
    loadRenderPassLibrary('PixelInspectorPass.dll')
    loadRenderPassLibrary('ReSTIRPass.dll')
    loadRenderPassLibrary('ReSTIRPTPass.dll')
    loadRenderPassLibrary('ScreenSpaceReSTIRPass.dll')
    loadRenderPassLibrary('SSAO.dll')
    loadRenderPassLibrary('SVGFPass.dll')
    loadRenderPassLibrary('TemporalDelayPass.dll')
    loadRenderPassLibrary('ToneMapper.dll')
    loadRenderPassLibrary('Utils.dll')
    VBufferRT = createPass('VBufferRT', {'outputSize': IOSize.Default, 'samplePattern': SamplePattern.Halton, 'sampleCount': 16, 'useAlphaTest': True, 'adjustShadingNormals': True, 'forceCullMode': False, 'cull': CullMode.CullBack, 'useTraceRayInline': False})
    g.addPass(VBufferRT, 'VBufferRT')
    ToneMapper = createPass('ToneMapper', {'outputSize': IOSize.Default, 'useSceneMetadata': True, 'exposureCompensation': 0.0, 'autoExposure': False, 'filmSpeed': 100.0, 'whiteBalance': False, 'whitePoint': 6500.0, 'operator': ToneMapOp.Aces, 'clamp': True, 'whiteMaxLuminance': 1.0, 'whiteScale': 11.199999809265137, 'fNumber': 1.0, 'shutter': 1.0, 'exposureMode': ExposureMode.AperturePriority, 'irayExposure': False})
    g.addPass(ToneMapper, 'ToneMapper')
    ReSTIRPass = createPass('ReSTIRPass', {'enableTemporalResampling': True, 'enableSpatialResampling': True, 'storeFinalVisibility': True})
    g.addPass(ReSTIRPass, 'ReSTIRPass')
    g.addEdge('VBufferRT.vbuffer', 'ReSTIRPass.vbuffer')
    g.addEdge('VBufferRT.mvec', 'ReSTIRPass.motionVecs')
    g.addEdge('ReSTIRPass.color', 'ToneMapper.src')
    g.markOutput('ToneMapper.dst')
    return g

Realtime = render_graph_Realtime()
try: m.addGraph(Realtime)
except NameError: None
