add_renderpass(BDPT)

target_sources(BDPT PRIVATE
    BDPT.cpp
    BDPT.h
    BDPTLightPath.slang
    BDPTParams.slang
    BDPTUtils.slangh
    Bitonic32InnerSortCS.slang
    Bitonic32OuterSortCS.slang
    Bitonic32PreSortCS.slang
    Bitonic64InnerSortCS.slang
    Bitonic64OuterSortCS.slang
    Bitonic64PreSortCS.slang
    Bitonic64Sort.cpp
    Bitonic64Sort.h
    BitonicIndirectArgsCS.slang
    BitonicSortCommon.slangh
    ColorType.slang
    FindMaxPosition.cs.slang
    FindMinPosition.cs.slang
    GenInternalLevel.cs.slang
    GenLevelZeroFromVertexs.cs.slang
    GenMordenCode.cs.slang
    GeneratePaths.cs.slang
    GuideData.slang
    LoadShadingData.slang
    Map.cs.slang
    MordenCode.cpp
    MordenCode.h
    NRDHelpers.slang
    Node.slang
    PathData.slang
    PathState.slang
    PathTracer.slang
    PathTracerNRD.slang
    PhotonCulling.cs.slang
    ReflectTypes.cs.slang
    ResolvePass.cs.slang
    SortTest.cs.slang
    SpatiotemporalReuse.cs.slang
    StaticParams.slang
    TraceCameraPath.rt.slang
    TraceLightPath.rt.slang
    TracePass.rt.slang
    VertexTreeBuilder.cpp
    VertexTreeBuilder.h
)

target_copy_shaders(BDPT RenderPasses/BDPT)

target_source_group(BDPT "RenderPasses")
