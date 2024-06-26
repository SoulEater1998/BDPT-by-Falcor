/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "BDPT.h"
#include "RenderGraph/RenderPassLibrary.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Rendering/Lights/EmissiveUniformSampler.h"

//#include "PathData.slang"

static const uint zero = 0;
static const bool fastBuild = true;
static const uint4 zero4 = uint4(0, 0, 0, 0);
const RenderPass::Info BDPT::kInfo { "BDPT", "Insert pass description here." };


namespace
{
    const std::string kGeneratePathsFilename = "RenderPasses/BDPT/GeneratePaths.cs.slang";
    const std::string kTracePassFilename = "RenderPasses/BDPT/TracePass.rt.slang";
    const std::string kTraceLightPathFilename = "RenderPasses/BDPT/TraceLightPath.rt.slang";
    const std::string kTraceCameraPathFilename = "RenderPasses/BDPT/TraceCameraPath.rt.slang";
    const std::string kResolvePassFilename = "RenderPasses/BDPT/ResolvePass.cs.slang";
    const std::string kMapPassFilename = "RenderPasses/BDPT/Map.cs.slang";
    const std::string kReflectTypesFile = "RenderPasses/BDPT/ReflectTypes.cs.slang";
    const std::string kSpatiotemporalReuseFilename = "RenderPasses/BDPT/SpatiotemporalReuse.cs.slang";
    const std::string kCullingPassFilename = "RenderPasses/BDPT/PhotonCulling.cs.slang";

    const std::string kShaderModel = "6_5";

    // Render pass inputs and outputs.
    const std::string kInputVBuffer = "vbuffer";
    const std::string kInputMotionVectors = "mvec";
    const std::string kInputViewDir = "viewW";
    const std::string kInputSampleCount = "sampleCount";

    const Falcor::ChannelList kInputChannels =
    {
        { kInputVBuffer,        "gVBuffer",         "Visibility buffer in packed format" },
        { kInputMotionVectors,  "gMotionVectors",   "Motion vector buffer (float format)", true /* optional */ },
        { kInputViewDir,        "gViewW",           "World-space view direction (xyz float format)", true /* optional */ },
        { kInputSampleCount,    "gSampleCount",     "Sample count buffer (integer format)", true /* optional */, ResourceFormat::R8Uint },
    };

    const std::string kOutputColor = "color";
    const std::string kOutputAlbedo = "albedo";
    const std::string kOutputSpecularAlbedo = "specularAlbedo";
    const std::string kOutputIndirectAlbedo = "indirectAlbedo";
    const std::string kOutputNormal = "normal";
    const std::string kOutputReflectionPosW = "reflectionPosW";
    const std::string kOutputRayCount = "rayCount";
    const std::string kOutputPathLength = "pathLength";

    //NRD outputs
    /*
    const std::string kOutputNRDDiffuseRadianceHitDist = "nrdDiffuseRadianceHitDist";
    const std::string kOutputNRDSpecularRadianceHitDist = "nrdSpecularRadianceHitDist";
    const std::string kOutputNRDEmission = "nrdEmission";
    const std::string kOutputNRDDiffuseReflectance = "nrdDiffuseReflectance";
    const std::string kOutputNRDSpecularReflectance = "nrdSpecularReflectance";
    const std::string kOutputNRDDeltaReflectionRadianceHitDist = "nrdDeltaReflectionRadianceHitDist";
    const std::string kOutputNRDDeltaReflectionReflectance = "nrdDeltaReflectionReflectance";
    const std::string kOutputNRDDeltaReflectionEmission = "nrdDeltaReflectionEmission";
    const std::string kOutputNRDDeltaReflectionNormWRoughMaterialID = "nrdDeltaReflectionNormWRoughMaterialID";
    const std::string kOutputNRDDeltaReflectionPathLength = "nrdDeltaReflectionPathLength";
    const std::string kOutputNRDDeltaReflectionHitDist = "nrdDeltaReflectionHitDist";
    const std::string kOutputNRDDeltaTransmissionRadianceHitDist = "nrdDeltaTransmissionRadianceHitDist";
    const std::string kOutputNRDDeltaTransmissionReflectance = "nrdDeltaTransmissionReflectance";
    const std::string kOutputNRDDeltaTransmissionEmission = "nrdDeltaTransmissionEmission";
    const std::string kOutputNRDDeltaTransmissionNormWRoughMaterialID = "nrdDeltaTransmissionNormWRoughMaterialID";
    const std::string kOutputNRDDeltaTransmissionPathLength = "nrdDeltaTransmissionPathLength";
    const std::string kOutputNRDDeltaTransmissionPosW = "nrdDeltaTransmissionPosW";
    const std::string kOutputNRDResidualRadianceHitDist = "nrdResidualRadianceHitDist";
    */
    const Falcor::ChannelList kOutputChannels =
    {
        { kOutputColor,                                     "",     "Output color (linear)", true /* optional */, ResourceFormat::RGBA32Float },
        { kOutputAlbedo,                                    "",     "Output albedo (linear)", true /* optional */, ResourceFormat::RGBA8Unorm },
        { kOutputSpecularAlbedo,                            "",     "Output specular albedo (linear)", true /* optional */, ResourceFormat::RGBA8Unorm },
        { kOutputIndirectAlbedo,                            "",     "Output indirect albedo (linear)", true /* optional */, ResourceFormat::RGBA8Unorm },
        { kOutputNormal,                                    "",     "Output normal (linear)", true /* optional */, ResourceFormat::RGBA16Float },
        { kOutputReflectionPosW,                            "",     "Output reflection pos (world space)", true /* optional */, ResourceFormat::RGBA32Float },
        { kOutputRayCount,                                  "",     "Per-pixel ray count", true /* optional */, ResourceFormat::R32Uint },
        { kOutputPathLength,                                "",     "Per-pixel path length", true /* optional */, ResourceFormat::R32Uint },
        // NRD outputs
        
        //{ kOutputNRDDiffuseRadianceHitDist,                 "",     "Output demodulated diffuse color (linear) and hit distance", true /* optional */, ResourceFormat::RGBA32Float },
        //{ kOutputNRDSpecularRadianceHitDist,                "",     "Output demodulated specular color (linear) and hit distance", true /* optional */, ResourceFormat::RGBA32Float },
        //{ kOutputNRDEmission,                               "",     "Output primary surface emission", true /* optional */, ResourceFormat::RGBA32Float },
        //{ kOutputNRDDiffuseReflectance,                     "",     "Output primary surface diffuse reflectance", true /* optional */, ResourceFormat::RGBA16Float },
        //{ kOutputNRDSpecularReflectance,                    "",     "Output primary surface specular reflectance", true /* optional */, ResourceFormat::RGBA16Float },
        //{ kOutputNRDDeltaReflectionRadianceHitDist,         "",     "Output demodulated delta reflection color (linear)", true /* optional */, ResourceFormat::RGBA32Float },
        //{ kOutputNRDDeltaReflectionReflectance,             "",     "Output delta reflection reflectance color (linear)", true /* optional */, ResourceFormat::RGBA16Float },
        //{ kOutputNRDDeltaReflectionEmission,                "",     "Output delta reflection emission color (linear)", true /* optional */, ResourceFormat::RGBA32Float },
        //{ kOutputNRDDeltaReflectionNormWRoughMaterialID,    "",     "Output delta reflection world normal, roughness, and material ID", true /* optional */, ResourceFormat::RGB10A2Unorm },
        //{ kOutputNRDDeltaReflectionPathLength,              "",     "Output delta reflection path length", true /* optional */, ResourceFormat::R16Float },
        //{ kOutputNRDDeltaReflectionHitDist,                 "",     "Output delta reflection hit distance", true /* optional */, ResourceFormat::R16Float },
        //{ kOutputNRDDeltaTransmissionRadianceHitDist,       "",     "Output demodulated delta transmission color (linear)", true /* optional */, ResourceFormat::RGBA32Float },
        //{ kOutputNRDDeltaTransmissionReflectance,           "",     "Output delta transmission reflectance color (linear)", true /* optional */, ResourceFormat::RGBA16Float },
        //{ kOutputNRDDeltaTransmissionEmission,              "",     "Output delta transmission emission color (linear)", true /* optional */, ResourceFormat::RGBA32Float },
        //{ kOutputNRDDeltaTransmissionNormWRoughMaterialID,  "",     "Output delta transmission world normal, roughness, and material ID", true /* optional */, ResourceFormat::RGB10A2Unorm },
        //{ kOutputNRDDeltaTransmissionPathLength,            "",     "Output delta transmission path length", true /* optional */, ResourceFormat::R16Float },
        //{ kOutputNRDDeltaTransmissionPosW,                  "",     "Output delta transmission position", true /* optional */, ResourceFormat::RGBA32Float },
        //{ kOutputNRDResidualRadianceHitDist,                "",     "Output residual color (linear) and hit distance", true /* optional */, ResourceFormat::RGBA32Float },
        
    };

    // UI variables.
    const Gui::DropdownList kColorFormatList =
    {
        { (uint32_t)ColorFormat::RGBA32F, "RGBA32F (128bpp)" },
        { (uint32_t)ColorFormat::LogLuvHDR, "LogLuvHDR (32bpp)" },
    };

    const Gui::DropdownList kMISHeuristicList =
    {
        { (uint32_t)MISHeuristic::Balance, "Balance heuristic" },
        { (uint32_t)MISHeuristic::PowerTwo, "Power heuristic (exp=2)" },
        { (uint32_t)MISHeuristic::PowerExp, "Power heuristic" },
    };

    const Gui::DropdownList kEmissiveSamplerList =
    {
        { (uint32_t)EmissiveLightSamplerType::Uniform, "Uniform" },
        { (uint32_t)EmissiveLightSamplerType::LightBVH, "LightBVH" },
        { (uint32_t)EmissiveLightSamplerType::Power, "Power" },
    };

    const Gui::DropdownList kLODModeList =
    {
        { (uint32_t)TexLODMode::Mip0, "Mip0" },
        { (uint32_t)TexLODMode::RayDiffs, "Ray Diffs" }
    };

    // Scripting options.
    const std::string kSamplesPerPixel = "samplesPerPixel";
    const std::string kMaxSurfaceBounces = "maxSurfaceBounces";
    const std::string kMaxDiffuseBounces = "maxDiffuseBounces";
    const std::string kMaxSpecularBounces = "maxSpecularBounces";
    const std::string kMaxTransmissionBounces = "maxTransmissionBounces";

    const std::string kSampleGenerator = "sampleGenerator";
    const std::string kFixedSeed = "fixedSeed";
    const std::string kUseBSDFSampling = "useBSDFSampling";
    const std::string kUseRussianRoulette = "useRussianRoulette";
    const std::string kUseNEE = "useNEE";
    const std::string kUseMIS = "useMIS";
    const std::string kMISHeuristic = "misHeuristic";
    const std::string kMISPowerExponent = "misPowerExponent";
    const std::string kEmissiveSampler = "emissiveSampler";
    const std::string kLightBVHOptions = "lightBVHOptions";
    const std::string kUseRTXDI = "useRTXDI";
    const std::string kRTXDIOptions = "RTXDIOptions";

    const std::string kUseAlphaTest = "useAlphaTest";
    const std::string kAdjustShadingNormals = "adjustShadingNormals";
    const std::string kMaxNestedMaterials = "maxNestedMaterials";
    const std::string kUseLightsInDielectricVolumes = "useLightsInDielectricVolumes";
    const std::string kDisableCaustics = "disableCaustics";
    const std::string kSpecularRoughnessThreshold = "specularRoughnessThreshold";
    const std::string kPrimaryLodMode = "primaryLodMode";
    const std::string kLODBias = "lodBias";

    const std::string kOutputSize = "outputSize";
    const std::string kFixedOutputSize = "fixedOutputSize";
    const std::string kColorFormat = "colorFormat";

    //const std::string kUseNRDDemodulation = "useNRDDemodulation";
}

// Don't remove this. it's required for hot-reload to function properly
extern "C" FALCOR_API_EXPORT const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" FALCOR_API_EXPORT void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerPass(BDPT::kInfo, BDPT::create);
}

void BDPT::registerBindings(pybind11::module& m)
{
    pybind11::enum_<ColorFormat> colorFormat(m, "ColorFormat");
    colorFormat.value("RGBA32F", ColorFormat::RGBA32F);
    colorFormat.value("LogLuvHDR", ColorFormat::LogLuvHDR);

    pybind11::enum_<MISHeuristic> misHeuristic(m, "MISHeuristic");
    misHeuristic.value("Balance", MISHeuristic::Balance);
    misHeuristic.value("PowerTwo", MISHeuristic::PowerTwo);
    misHeuristic.value("PowerExp", MISHeuristic::PowerExp);

    pybind11::class_<BDPT, RenderPass, BDPT::SharedPtr> pass(m, "BDPT");
    pass.def_property_readonly("pixelStats", &BDPT::getPixelStats);

    pass.def_property("useFixedSeed",
        [](const BDPT* pt) { return pt->mParams.useFixedSeed ? true : false; },
        [](BDPT* pt, bool value) { pt->mParams.useFixedSeed = value ? 1 : 0; }
    );
    pass.def_property("fixedSeed",
        [](const BDPT* pt) { return pt->mParams.fixedSeed; },
        [](BDPT* pt, uint32_t value) { pt->mParams.fixedSeed = value; }
    );
}

BDPT::SharedPtr BDPT::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    SharedPtr pPass = SharedPtr(new BDPT(dict));
    return pPass;
}

BDPT::BDPT(const Dictionary& dict) : RenderPass(kInfo)
{
    if (!gpDevice->isShaderModelSupported(Device::ShaderModel::SM6_5))
    {
        throw RuntimeError("PathTracer: Shader Model 6.5 is not supported by the current device");
    }
    if (!gpDevice->isFeatureSupported(Device::SupportedFeatures::RaytracingTier1_1))
    {
        throw RuntimeError("PathTracer: Raytracing Tier 1.1 is not supported by the current device");
    }

    parseDictionary(dict);
    validateOptions();

    // Create sample generator.
    mpSampleGenerator = SampleGenerator::create(mStaticParams.sampleGenerator);

    // Create resolve pass. This doesn't depend on the scene so can be created here.
    auto defines = mStaticParams.getDefines(*this);
    mpResolvePass = ComputePass::create(Program::Desc(kResolvePassFilename).setShaderModel(kShaderModel).csEntry("main"), defines, false);

    Program::DefineList subSpaceDefines;
    subSpaceDefines.add("GROUP_SIZE", std::to_string(1 << (mStaticParams.logSubspaceSize - 1)));
    mpMergePass = ComputePass::create(kMapPassFilename, "main", subSpaceDefines);
    mpPrefixSumPass = ComputePass::create(kMapPassFilename, "scan", subSpaceDefines);

    // Note: The other programs are lazily created in updatePrograms() because a scene needs to be present when creating them.

    mpPixelStats = PixelStats::create();
    mpPixelDebug = PixelDebug::create();
}

void BDPT::parseDictionary(const Dictionary& dict)
{
    for (const auto& [key, value] : dict)
    {
        // Rendering parameters
        if (key == kSamplesPerPixel) mStaticParams.samplesPerPixel = value;
        else if (key == kMaxSurfaceBounces) mStaticParams.maxSurfaceBounces = value;
        else if (key == kMaxDiffuseBounces) mStaticParams.maxDiffuseBounces = value;
        else if (key == kMaxSpecularBounces) mStaticParams.maxSpecularBounces = value;
        else if (key == kMaxTransmissionBounces) mStaticParams.maxTransmissionBounces = value;

        // Sampling parameters
        else if (key == kSampleGenerator) mStaticParams.sampleGenerator = value;
        else if (key == kFixedSeed) { mParams.fixedSeed = value; mParams.useFixedSeed = true; }
        else if (key == kUseBSDFSampling) mStaticParams.useBSDFSampling = value;
        else if (key == kUseRussianRoulette) mStaticParams.useRussianRoulette = value;
        else if (key == kUseNEE) mStaticParams.useNEE = value;
        else if (key == kUseMIS) mStaticParams.useMIS = value;
        else if (key == kMISHeuristic) mStaticParams.misHeuristic = value;
        else if (key == kMISPowerExponent) mStaticParams.misPowerExponent = value;
        else if (key == kEmissiveSampler) mStaticParams.emissiveSampler = value;
        else if (key == kLightBVHOptions) mLightBVHOptions = value;
        else if (key == kUseRTXDI) mStaticParams.useRTXDI = value;
        else if (key == kRTXDIOptions) mRTXDIOptions = value;

        // Material parameters
        else if (key == kUseAlphaTest) mStaticParams.useAlphaTest = value;
        else if (key == kAdjustShadingNormals) mStaticParams.adjustShadingNormals = value;
        else if (key == kMaxNestedMaterials) mStaticParams.maxNestedMaterials = value;
        else if (key == kUseLightsInDielectricVolumes) mStaticParams.useLightsInDielectricVolumes = value;
        else if (key == kDisableCaustics) mStaticParams.disableCaustics = value;
        else if (key == kSpecularRoughnessThreshold) mParams.specularRoughnessThreshold = value;
        else if (key == kPrimaryLodMode) mStaticParams.primaryLodMode = value;
        else if (key == kLODBias) mParams.lodBias = value;

        // Denoising parameters
        //else if (key == kUseNRDDemodulation) mStaticParams.useNRDDemodulation = value;

        // Output parameters
        else if (key == kOutputSize) mOutputSizeSelection = value;
        else if (key == kFixedOutputSize) mFixedOutputSize = value;
        else if (key == kColorFormat) mStaticParams.colorFormat = value;

        else logWarning("Unknown field '{}' in BDPT dictionary.", key);
    }

    if (dict.keyExists(kMaxSurfaceBounces))
    {
        // Initialize bounce counts to 'maxSurfaceBounces' if they weren't explicitly set.
        if (!dict.keyExists(kMaxDiffuseBounces)) mStaticParams.maxDiffuseBounces = mStaticParams.maxSurfaceBounces;
        if (!dict.keyExists(kMaxSpecularBounces)) mStaticParams.maxSpecularBounces = mStaticParams.maxSurfaceBounces;
        if (!dict.keyExists(kMaxTransmissionBounces)) mStaticParams.maxTransmissionBounces = mStaticParams.maxSurfaceBounces;
    }
    else
    {
        // Initialize surface bounces.
        mStaticParams.maxSurfaceBounces = std::max(mStaticParams.maxDiffuseBounces, std::max(mStaticParams.maxSpecularBounces, mStaticParams.maxTransmissionBounces));
    }

    bool maxSurfaceBouncesNeedsAdjustment =
        mStaticParams.maxSurfaceBounces < mStaticParams.maxDiffuseBounces ||
        mStaticParams.maxSurfaceBounces < mStaticParams.maxSpecularBounces ||
        mStaticParams.maxSurfaceBounces < mStaticParams.maxTransmissionBounces;

    // Show a warning if maxSurfaceBounces will be adjusted in validateOptions().
    if (dict.keyExists(kMaxSurfaceBounces) && maxSurfaceBouncesNeedsAdjustment)
    {
        logWarning("'{}' is set lower than '{}', '{}' or '{}' and will be increased.", kMaxSurfaceBounces, kMaxDiffuseBounces, kMaxSpecularBounces, kMaxTransmissionBounces);
    }
}

void BDPT::validateOptions()
{
    if (mParams.specularRoughnessThreshold < 0.f || mParams.specularRoughnessThreshold > 1.f)
    {
        logWarning("'specularRoughnessThreshold' has invalid value. Clamping to range [0,1].");
        mParams.specularRoughnessThreshold = clamp(mParams.specularRoughnessThreshold, 0.f, 1.f);
    }

    // Static parameters.
    if (mStaticParams.samplesPerPixel < 1 || mStaticParams.samplesPerPixel > kMaxSamplesPerPixel)
    {
        logWarning("'samplesPerPixel' must be in the range [1, {}]. Clamping to this range.", kMaxSamplesPerPixel);
        mStaticParams.samplesPerPixel = std::clamp(mStaticParams.samplesPerPixel, 1u, kMaxSamplesPerPixel);
    }

    auto clampBounces = [](uint32_t& bounces, const std::string& name)
    {
        if (bounces > kMaxBounces)
        {
            logWarning("'{}' exceeds the maximum supported bounces. Clamping to {}.", name, kMaxBounces);
            bounces = kMaxBounces;
        }
    };

    clampBounces(mStaticParams.maxSurfaceBounces, kMaxSurfaceBounces);
    clampBounces(mStaticParams.maxDiffuseBounces, kMaxDiffuseBounces);
    clampBounces(mStaticParams.maxSpecularBounces, kMaxSpecularBounces);
    clampBounces(mStaticParams.maxTransmissionBounces, kMaxTransmissionBounces);

    // Make sure maxSurfaceBounces is at least as many as any of diffuse, specular or transmission.
    uint32_t minSurfaceBounces = std::max(mStaticParams.maxDiffuseBounces, std::max(mStaticParams.maxSpecularBounces, mStaticParams.maxTransmissionBounces));
    mStaticParams.maxSurfaceBounces = std::max(mStaticParams.maxSurfaceBounces, minSurfaceBounces);

    if (mStaticParams.primaryLodMode == TexLODMode::RayCones)
    {
        logWarning("Unsupported tex lod mode. Defaulting to Mip0.");
        mStaticParams.primaryLodMode = TexLODMode::Mip0;
    }
}

Dictionary BDPT::getScriptingDictionary()
{
    if (auto lightBVHSampler = std::dynamic_pointer_cast<LightBVHSampler>(mpEmissiveSampler))
    {
        mLightBVHOptions = lightBVHSampler->getOptions();
    }

    Dictionary d;

    // Rendering parameters
    d[kSamplesPerPixel] = mStaticParams.samplesPerPixel;
    d[kMaxSurfaceBounces] = mStaticParams.maxSurfaceBounces;
    d[kMaxDiffuseBounces] = mStaticParams.maxDiffuseBounces;
    d[kMaxSpecularBounces] = mStaticParams.maxSpecularBounces;
    d[kMaxTransmissionBounces] = mStaticParams.maxTransmissionBounces;

    // Sampling parameters
    d[kSampleGenerator] = mStaticParams.sampleGenerator;
    if (mParams.useFixedSeed) d[kFixedSeed] = mParams.fixedSeed;
    d[kUseBSDFSampling] = mStaticParams.useBSDFSampling;
    d[kUseRussianRoulette] = mStaticParams.useRussianRoulette;
    d[kUseNEE] = mStaticParams.useNEE;
    d[kUseMIS] = mStaticParams.useMIS;
    d[kMISHeuristic] = mStaticParams.misHeuristic;
    d[kMISPowerExponent] = mStaticParams.misPowerExponent;
    d[kEmissiveSampler] = mStaticParams.emissiveSampler;
    if (mStaticParams.emissiveSampler == EmissiveLightSamplerType::LightBVH) d[kLightBVHOptions] = mLightBVHOptions;
    d[kUseRTXDI] = mStaticParams.useRTXDI;
    d[kRTXDIOptions] = mRTXDIOptions;

    // Material parameters
    d[kUseAlphaTest] = mStaticParams.useAlphaTest;
    d[kAdjustShadingNormals] = mStaticParams.adjustShadingNormals;
    d[kMaxNestedMaterials] = mStaticParams.maxNestedMaterials;
    d[kUseLightsInDielectricVolumes] = mStaticParams.useLightsInDielectricVolumes;
    d[kDisableCaustics] = mStaticParams.disableCaustics;
    d[kSpecularRoughnessThreshold] = mParams.specularRoughnessThreshold;
    d[kPrimaryLodMode] = mStaticParams.primaryLodMode;
    d[kLODBias] = mParams.lodBias;

    // Denoising parameters
    //d[kUseNRDDemodulation] = mStaticParams.useNRDDemodulation;

    // Output parameters
    d[kOutputSize] = mOutputSizeSelection;
    if (mOutputSizeSelection == RenderPassHelpers::IOSize::Fixed) d[kFixedOutputSize] = mFixedOutputSize;
    d[kColorFormat] = mStaticParams.colorFormat;

    return d;
}

RenderPassReflection BDPT::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    const uint2 sz = RenderPassHelpers::calculateIOSize(mOutputSizeSelection, mFixedOutputSize, compileData.defaultTexDims);

    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels, ResourceBindFlags::UnorderedAccess, sz);
    return reflector;
}

void BDPT::setFrameDim(const uint2 frameDim)
{
    auto prevFrameDim = mParams.frameDim;
    auto prevScreenTiles = mParams.screenTiles;

    mParams.frameDim = frameDim;
    if (mParams.frameDim.x > kMaxFrameDimension || mParams.frameDim.y > kMaxFrameDimensionY)
    {
        throw RuntimeError("Frame dimensions up to {} pixels width/height are supported.", kMaxFrameDimension);
    }

    // Tile dimensions have to be powers-of-two.
    FALCOR_ASSERT(isPowerOf2(kScreenTileDim.x) && isPowerOf2(kScreenTileDim.y));
    FALCOR_ASSERT(kScreenTileDim.x == (1 << kScreenTileBits.x) && kScreenTileDim.y == (1 << kScreenTileBits.y));
    mParams.screenTiles = div_round_up(mParams.frameDim, kScreenTileDim);

    if (mParams.frameDim != prevFrameDim || mParams.screenTiles != prevScreenTiles)
    {
        mVarsChanged = true;
    }
}

void BDPT::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    mpScene = pScene;
    mParams.frameCount = 0;
    mParams.frameDim = {};
    mParams.screenTiles = {};

    // Need to recreate the RTXDI module when the scene changes.
    mpRTXDI = nullptr;

    // Need to recreate the trace passes because the shader binding table changes.
    mpTracePass = nullptr;
    mpTraceLightPath = nullptr;
    mpTraceCameraPath = nullptr;
    //mpTraceDeltaReflectionPass = nullptr;
    //mpTraceDeltaTransmissionPass = nullptr;
    mpGeneratePaths = nullptr;
    mpReflectTypes = nullptr;
    mpSpatiotemporalReuse = nullptr;

    resetLighting();

    if (mpScene)
    {
        if (pScene->hasGeometryType(Scene::GeometryType::Custom))
        {
            logWarning("PathTracer: This render pass does not support custom primitives.");
        }

        validateOptions();

        mRecompile = true;
    }
}

void BDPT::resetLighting()
{
    // Retain the options for the emissive sampler.
    if (auto lightBVHSampler = std::dynamic_pointer_cast<LightBVHSampler>(mpEmissiveSampler))
    {
        mLightBVHOptions = lightBVHSampler->getOptions();
    }

    mpEmissiveSampler = nullptr;
    mpEnvMapSampler = nullptr;
    mRecompile = true;
}

void BDPT::sortPosition(RenderContext* pRenderContext) {

}

BDPT::CollectPass::CollectPass(const std::string& shaderFile, const Scene::SharedPtr& pScene) {
    uint maxPayloadSize = 48u;

    RtProgram::Desc desc;
    desc.addShaderLibrary(shaderFile);
    desc.setMaxPayloadSize(maxPayloadSize);
    desc.setMaxAttributeSize(8u);
    desc.setMaxTraceRecursionDepth(2u);

    pBindingTable = RtBindingTable::create(1, 1, pScene->getGeometryCount());
    auto& sbt = pBindingTable;
    sbt->setRayGen(desc.addRayGen("rayGen"));
    sbt->setMiss(0, desc.addMiss("miss"));
    auto hitShader = desc.addHitGroup("", "anyHit", "intersection");
    sbt->setHitGroup(0, 0, hitShader);

    pProgram = RtProgram::create(desc, pScene->getSceneDefines());
}

void BDPT::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    //initTimer();

    //pGpuTimer->begin();
    if (!beginFrame(pRenderContext, renderData)) return;

    // Update shader program specialization.
    updatePrograms();

    // Prepare resources.
    prepareResources(pRenderContext, renderData);

    // Prepare the path tracer parameter block.
    // This should be called after all resources have been created.
    preparePathTracer(renderData);

    //if (mRebuild) prepareAccelerationStructure(pRenderContext);

    // Generate paths at primary hits.
    generatePaths(pRenderContext, renderData);

    // Update RTXDI.
    if (mpRTXDI)
    {
        const auto& pMotionVectors = renderData.getTexture(kInputMotionVectors);
        mpRTXDI->update(pRenderContext, pMotionVectors);
    }

    // Trace pass.
    //mParams.LightPathsIndexBufferLength = 0;
    //FALCOR_ASSERT(mpTracePass);
    //tracePass(pRenderContext, renderData, *mpTracePass, mParams.frameDim);

    // Generate light path
    //mParams.LightPathsIndexBufferLength = 0;
    //pRenderContext->clearUAV(mpLightPathVertexBuffer->getUAV().get(), zero4);
    //pRenderContext->clearUAV(mpLightPathsIndexBuffer->getUAV().get(), zero4);
    pRenderContext->uavBarrier(mpLightPathsIndexBuffer->getUAVCounter().get());
    pRenderContext->clearUAVCounter(mpLightPathsIndexBuffer, 0);
    //pGpuTimer->end();
    //pGpuTimer->resolve();
    //logWarning("init part " + std::to_string(pGpuTimer->getElapsedTime()));

    //pGpuTimer->begin();
    FALCOR_ASSERT(mpTraceLightPath);
    tracePass(pRenderContext, renderData, *mpTraceLightPath, uint2(mStaticParams.lightPassWidth, mStaticParams.lightPassHeight));

    pRenderContext->uavBarrier(mpLightPathVertexBuffer.get());
    //pRenderContext->uavBarrier(mpLightPathsVertexsPositionBuffer.get());
    pRenderContext->uavBarrier(mpLightPathsIndexBuffer.get());

    //pGpuTimer->end();
    //pGpuTimer->resolve();
    //logWarning("light path: " + std::to_string(pGpuTimer->getElapsedTime()));
    //pGpuTimer->begin();
    // Generate camera path
    
    auto kCounterPtr = (uint*)mpLightPathsIndexBuffer->getUAVCounter()->map(Buffer::MapType::Read);
    uint counter = *kCounterPtr;
    logWarning(std::to_string(counter));
    mpSortPass->setListCount(counter);
    mpSortPass->sort(pRenderContext);
    //pGpuTimer->end();
    //pGpuTimer->resolve();
    //logWarning("sort: " + std::to_string(pGpuTimer->getElapsedTime()));
    //pGpuTimer->begin();
    mpTreeBuilder->update(counter, radius);
    mpTreeBuilder->build(pRenderContext);
    //mpSortPass->sortTest(pRenderContext, renderData.getTexture(kOutputColor), mParams.frameDim);

    pRenderContext->uavBarrier(mpTreeBuilder->getTree().get());
    //pGpuTimer->end();
    //pGpuTimer->resolve();
    //logWarning("build tree: " + std::to_string(pGpuTimer->getElapsedTime()));
    //pGpuTimer->begin();
    //mpPrevGatherPoints->uavBarrier(pRenderContext);
        
    //pRenderContext->clearUAVCounter(mpAABB, 0);  
   

    FALCOR_ASSERT(mpTraceCameraPath);
    {
        auto var = mpPathTracerBlock->getRootVar();
        var["leafNodeStart"] = mpTreeBuilder->getLeafNodeStartIndex();
        var["counter"] = counter;
    }
    tracePass(pRenderContext, renderData, *mpTraceCameraPath, mParams.frameDim);
    mpSubspaceReservoir->uavBarrier(pRenderContext);
    mpGatherPoints->uavBarrier(pRenderContext);
    //mpPairs->uavBarrier(pRenderContext);
    pRenderContext->uavBarrier(mpSubspaceWeight[0].get());
    pRenderContext->uavBarrier(mpSubspaceCount[0].get());
    pRenderContext->uavBarrier(mpSubspaceSecondaryMoment.get());
    //pRenderContext->uavBarrier(mpAABB->getUAVCounter().get());
    pRenderContext->uavBarrier(mpOutput.get());
    //pGpuTimer->end();
    //pGpuTimer->resolve();
    //logWarning("eye path: " + std::to_string(pGpuTimer->getElapsedTime()));
    //pGpuTimer->begin();
    //auto kValidCounterPtr = (uint*)mpAABB->getUAVCounter()->map(Buffer::MapType::Read);
    //uint validCounter = *kValidCounterPtr;
    //logWarning(std::to_string(validCounter));
    
    //pRenderContext->uavBarrier(mpHashBuffer.get());

    // culling
    /*
    {
        pRenderContext->clearUAVCounter(mpCausticBuffers.infoFlux, 0);
        pRenderContext->clearUAVCounter(mpGlobalBuffers.infoFlux, 0);
        auto var = mpCullingPass->getRootVar();
        var["CSConstants"]["num"] = counter;
        uint hashSize = 1 << mStaticParams.cullingHashBufferSizeBytes;
        var["CSConstants"]["gHashSize"] = hashSize;
        var["CSConstants"]["gYExtent"] = static_cast<uint>(sqrt(hashSize));
        var["CSConstants"]["gGlobalRadius"] = radius;

        if (mVarsChanged) {
            var["LightPathsVertexsBuffer"] = mpLightPathVertexBuffer;
            var["LightPathsVertexsPositionBuffer"] = mpLightPathsVertexsPositionBuffer;

            for (uint32_t i = 0; i <= 1; i++) {
                var["gPhotonAABB"][i] = i == 0 ? mpCausticBuffers.aabb : mpGlobalBuffers.aabb;
                var["gPhotonFlux"][i] = i == 0 ? mpCausticBuffers.infoFlux : mpGlobalBuffers.infoFlux;
                var["gPhotonDir"][i] = i == 0 ? mpCausticBuffers.infoDir : mpGlobalBuffers.infoDir;
            }

            var["hashBuffer"] = mpHashBuffer;
        }

        mpCullingPass->execute(pRenderContext, counter, 1);
    }
    */

    //spatiotemporalReuse(pRenderContext, renderData);

    //buildAccelerationStructure(pRenderContext);

    pRenderContext->copyResource(renderData.getTexture(kOutputColor).get(), mpOutput.get());
    if(mUseSubspace) buildSubspaceWeightMatrix(pRenderContext, renderData);
    //pGpuTimer->end();
    //pGpuTimer->resolve();
    //logWarning("update matrix: " + std::to_string(pGpuTimer->getElapsedTime()));
    //spatiotemporalReuse(pRenderContext, renderData);
    // Resolve pass.
    //resolvePass(pRenderContext, renderData);

    //pRenderContext->uavBarrier(mpCameraPathsVertexsReservoirBuffer.get());
    //pRenderContext->uavBarrier(mpOutput.get());
    //pGpuTimer->begin();
    endFrame(pRenderContext, renderData);
    //pGpuTimer->end();
    //pGpuTimer->resolve();
    //logWarning("end: " + std::to_string(pGpuTimer->getElapsedTime()));
}

double BDPT::checkTime(RenderContext* pRenderContext) {
    //pRenderContext->readti
    auto currentTime = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsedSec = currentTime - mCurrentTime;
    mCurrentTime = currentTime;
    return elapsedSec.count();
}

void BDPT::prepareAccelerationStructure(RenderContext* pRenderContext) {
    //mPhotonInstanceDesc = nullptr;
    mpTlasScratch = nullptr;
    mPhotonTlas.pInstanceDescs = nullptr; mPhotonTlas.pSrv = nullptr; mPhotonTlas.pTlas = nullptr;

    mBlasScratchMaxSize = 0;
    mTlasScratchMaxSize = 0;

    // prepare blas
    { 
        auto& blas = mBlasData;

        //Create geometry description
        D3D12_RAYTRACING_GEOMETRY_DESC& desc = blas.geomDescs;
        desc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
        desc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION;         //< Important! So that photons are not collected multiple times
        desc.AABBs.AABBCount = kMaxFrameDimension * kMaxFrameDimensionY * mStaticParams.maxSurfaceBounces;
        desc.AABBs.AABBs.StartAddress = mpAABB->getGpuAddress();
        desc.AABBs.AABBs.StrideInBytes = sizeof(D3D12_RAYTRACING_AABB);

        //Create input for blas
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS& inputs = blas.buildInputs;
        inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        inputs.NumDescs = 1;
        inputs.pGeometryDescs = &blas.geomDescs;

        //Updating is no option for BLAS as individual photons move to much.

        inputs.Flags = fastBuild ? D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD : D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
        //get prebuild Info
        FALCOR_GET_COM_INTERFACE(gpDevice->getApiHandle(), ID3D12Device5, pDevice5);
        pDevice5->GetRaytracingAccelerationStructurePrebuildInfo(&blas.buildInputs, &blas.prebuildInfo);

        // Figure out the padded allocation sizes to have proper alignment.
        FALCOR_ASSERT(blas.prebuildInfo.ResultDataMaxSizeInBytes > 0);
        blas.blasByteSize = align_to((uint64_t)D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT, blas.prebuildInfo.ResultDataMaxSizeInBytes);

        uint64_t scratchByteSize = std::max(blas.prebuildInfo.ScratchDataSizeInBytes, blas.prebuildInfo.UpdateScratchDataSizeInBytes);
        blas.scratchByteSize = align_to((uint64_t)D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT, scratchByteSize);

        mBlasScratchMaxSize = std::max(blas.scratchByteSize, mBlasScratchMaxSize);
        

        //Create the scratch and blas buffers
        mpBlasScratch = Buffer::create(mBlasScratchMaxSize, Buffer::BindFlags::UnorderedAccess, Buffer::CpuAccess::None);
        mpBlasScratch->setName("BDPT::BlasScratch");

        mpBlas = Buffer::create(mBlasData.blasByteSize, Buffer::BindFlags::AccelerationStructure, Buffer::CpuAccess::None);
    }

    // prepare tlas
    {
        D3D12_RAYTRACING_INSTANCE_DESC desc = {};
        desc.AccelerationStructure = mpBlas->getGpuAddress();
        desc.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
        desc.InstanceID = 0;
        desc.InstanceMask = 1;  //0b01 for Caustic and 0b10 for Global
        desc.InstanceContributionToHitGroupIndex = 0;

        //Create a identity matrix for the transform and copy it to the instance desc
        //glm::mat4 transform4x4 = glm::identity<glm::mat4>();
        //std::memcpy(desc.Transform, &transform4x4, sizeof(desc.Transform));
        mPhotonInstanceDesc = desc;

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {};
        inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        inputs.NumDescs = 1;
        inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;

        //Prebuild
        FALCOR_GET_COM_INTERFACE(gpDevice->getApiHandle(), ID3D12Device5, pDevice5);
        pDevice5->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &mTlasPrebuildInfo);
        mpTlasScratch = Buffer::create(std::max(mTlasPrebuildInfo.ScratchDataSizeInBytes, mTlasScratchMaxSize), Buffer::BindFlags::UnorderedAccess, Buffer::CpuAccess::None);
        mpTlasScratch->setName("BDPT::TLAS_Scratch");

        //Create buffers for the TLAS
        mPhotonTlas.pTlas = Buffer::create(mTlasPrebuildInfo.ResultDataMaxSizeInBytes, Buffer::BindFlags::AccelerationStructure, Buffer::CpuAccess::None);
        mPhotonTlas.pTlas->setName("BDPT::TLAS");
        mPhotonTlas.pInstanceDescs = Buffer::create(sizeof(D3D12_RAYTRACING_INSTANCE_DESC), Buffer::BindFlags::None, Buffer::CpuAccess::Write, &mPhotonInstanceDesc);
        mPhotonTlas.pInstanceDescs->setName("BDPT::TLAS_Instance_Description");

        //Acceleration Structure Buffer view for access in shader
        mPhotonTlas.pSrv = ShaderResourceView::createViewForAccelerationStructure(mPhotonTlas.pTlas);
    }

    if (mRebuild) mRebuild = false;
}

void BDPT::buildAccelerationStructure(RenderContext* pContext) {
    uint aabbCount;
    auto pCounter = (uint*)mpAABB->getUAVCounter()->map(Buffer::MapType::Read);
    aabbCount = *pCounter;
    //logWarning(std::to_string(aabbCount));
    

    // build blas
    {
        if (!gpDevice->isFeatureSupported(Device::SupportedFeatures::Raytracing))
        {
            throw std::exception("Raytracing is not supported by the current device");
        }

        //aabb buffers need to be ready
        pContext->uavBarrier(mpAABB.get());

        auto& blas = mBlasData;

        //barriers for the scratch and blas buffer
        pContext->uavBarrier(mpBlasScratch.get());
        pContext->uavBarrier(mpBlas.get());

        //add the photon count for this iteration. geomDesc is saved as a pointer in blasInputs
        uint maxPhotons = kMaxFrameDimension * kMaxFrameDimensionY * mStaticParams.maxSurfaceBounces;
        aabbCount = std::min(aabbCount, maxPhotons);
        blas.geomDescs.AABBs.AABBCount = static_cast<UINT64>(aabbCount);

        //Fill the build desc struct
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC asDesc = {};
        asDesc.Inputs = blas.buildInputs;
        asDesc.ScratchAccelerationStructureData = mpBlasScratch->getGpuAddress();
        asDesc.DestAccelerationStructureData = mpBlas->getGpuAddress();

        //Build the acceleration structure
        FALCOR_GET_COM_INTERFACE(pContext->getLowLevelData()->getCommandList(), ID3D12GraphicsCommandList4, pList4);
        pList4->BuildRaytracingAccelerationStructure(&asDesc, 0, nullptr);

        //Barrier for the blas
        pContext->uavBarrier(mpBlas.get());
    }

    //
    {
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {};
        inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        inputs.NumDescs = 1;
        //Update Flag could be set for TLAS. This made no real time difference in our test so it is left out. Updating could reduce the memory of the TLAS scratch buffer a bit
        inputs.Flags = fastBuild ? D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD : D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC asDesc = {};
        asDesc.Inputs = inputs;
        asDesc.Inputs.InstanceDescs = mPhotonTlas.pInstanceDescs->getGpuAddress();
        asDesc.ScratchAccelerationStructureData = mpTlasScratch->getGpuAddress();
        asDesc.DestAccelerationStructureData = mPhotonTlas.pTlas->getGpuAddress();

        // Create TLAS
        FALCOR_GET_COM_INTERFACE(pContext->getLowLevelData()->getCommandList(), ID3D12GraphicsCommandList4, pList4);
        pContext->resourceBarrier(mPhotonTlas.pInstanceDescs.get(), Resource::State::NonPixelShader);
        pList4->BuildRaytracingAccelerationStructure(&asDesc, 0, nullptr);
        pContext->uavBarrier(mPhotonTlas.pTlas.get());

        //logWarning(std::to_string(mpBlas->getElementCount()));
    }
}

bool BDPT::renderRenderingUI(Gui::Widgets& widget)
{
    bool dirty = false;
    bool runtimeDirty = false;

    if (mFixedSampleCount)
    {
        dirty |= widget.var("Samples/pixel", mStaticParams.samplesPerPixel, 1u, kMaxSamplesPerPixel);
    }
    else widget.text("Samples/pixel: Variable");
    widget.tooltip("Number of samples per pixel. One path is traced for each sample.\n\n"
        "When the '" + kInputSampleCount + "' input is connected, the number of samples per pixel is loaded from the texture.");

    if (widget.var("Max surface bounces", mStaticParams.maxSurfaceBounces, 0u, 20u))
    {
        // Allow users to change the max surface bounce parameter in the UI to clamp all other surface bounce parameters.
        mStaticParams.maxDiffuseBounces = std::min(mStaticParams.maxDiffuseBounces, mStaticParams.maxSurfaceBounces);
        mStaticParams.maxSpecularBounces = std::min(mStaticParams.maxSpecularBounces, mStaticParams.maxSurfaceBounces);
        mStaticParams.maxTransmissionBounces = std::min(mStaticParams.maxTransmissionBounces, mStaticParams.maxSurfaceBounces);
        dirty = true;
    }
    widget.tooltip("Maximum number of surface bounces (diffuse + specular + transmission).\n"
        "Note that specular reflection events from a material with a roughness greater than specularRoughnessThreshold are also classified as diffuse events.");

    dirty |= widget.var("Max diffuse bounces", mStaticParams.maxDiffuseBounces, 0u, 20u);
    widget.tooltip("Maximum number of diffuse bounces.\n0 = direct only\n1 = one indirect bounce etc.");

    dirty |= widget.var("Max specular bounces", mStaticParams.maxSpecularBounces, 0u, 20u);
    widget.tooltip("Maximum number of specular bounces.\n0 = direct only\n1 = one indirect bounce etc.");

    dirty |= widget.var("Max transmission bounces", mStaticParams.maxTransmissionBounces, 0u, 20u);
    widget.tooltip("Maximum number of transmission bounces.\n0 = no transmission\n1 = one transmission bounce etc.");

    // Sampling options.

    if (widget.dropdown("Sample generator", SampleGenerator::getGuiDropdownList(), mStaticParams.sampleGenerator))
    {
        mpSampleGenerator = SampleGenerator::create(mStaticParams.sampleGenerator);
        dirty = true;
    }

    dirty |= widget.checkbox("BSDF importance sampling", mStaticParams.useBSDFSampling);
    widget.tooltip("BSDF importance sampling should normally be enabled.\n\n"
        "If disabled, cosine-weighted hemisphere sampling is used for debugging purposes");

    dirty |= widget.checkbox("Russian roulette", mStaticParams.useRussianRoulette);
    widget.tooltip("Use russian roulette to terminate low throughput paths.");

    dirty |= widget.checkbox("Next-event estimation (NEE)", mStaticParams.useNEE);
    widget.tooltip("Use next-event estimation.\nThis option enables direct illumination sampling at each path vertex.");

    if (mStaticParams.useNEE)
    {
        dirty |= widget.checkbox("Multiple importance sampling (MIS)", mStaticParams.useMIS);
        widget.tooltip("When enabled, BSDF sampling is combined with light sampling for the environment map and emissive lights.\n"
            "Note that MIS has currently no effect on analytic lights.");

        if (mStaticParams.useMIS)
        {
            dirty |= widget.dropdown("MIS heuristic", kMISHeuristicList, reinterpret_cast<uint32_t&>(mStaticParams.misHeuristic));

            if (mStaticParams.misHeuristic == MISHeuristic::PowerExp)
            {
                dirty |= widget.var("MIS power exponent", mStaticParams.misPowerExponent, 0.01f, 10.f);
            }
        }

        if (mpScene && mpScene->useEmissiveLights())
        {
            if (auto group = widget.group("Emissive sampler"))
            {
                if (widget.dropdown("Emissive sampler", kEmissiveSamplerList, (uint32_t&)mStaticParams.emissiveSampler))
                {
                    resetLighting();
                    dirty = true;
                }
                widget.tooltip("Selects which light sampler to use for importance sampling of emissive geometry.", true);

                if (mpEmissiveSampler)
                {
                    if (mpEmissiveSampler->renderUI(group)) mOptionsChanged = true;
                }
            }
        }
    }

    if (auto group = widget.group("BDPT connection"))
    {
        runtimeDirty |= widget.checkbox("s == 0 connection", s0);
        widget.tooltip("Add the contribution of s=0 type to the final result.");
        mParams.flag = s0 ? uint(BDPTFlags::s0) : 0;

        runtimeDirty |= widget.checkbox("s == 1 connection", s1);
        widget.tooltip("Add the contribution of s=1 type to the final result.");
        mParams.flag |= s1 ? uint(BDPTFlags::s1) : 0;

        runtimeDirty |= widget.checkbox("s > 1 connection", s2);
        widget.tooltip("Add the contribution of s>1 type to the final result.");
        mParams.flag |= s2 ? uint(BDPTFlags::s2) : 0;

        runtimeDirty |= widget.checkbox("use Vertex Merge", mUseVertexMerge);
        widget.tooltip("use VCM.");
        mParams.flag |= mUseVertexMerge ? uint(BDPTFlags::useVertexMerge) : 0;

        if (mUseVertexMerge) {
            runtimeDirty |= widget.checkbox("temproal reuse", t1);
            widget.tooltip("temproal reuse surfle");
            mParams.flag |= t1 ? 0 : uint(BDPTFlags::t1);
        }
        else {
            runtimeDirty |= widget.checkbox("use subspace", mUseSubspace);
            widget.tooltip("use subspace BDPT");
            mParams.flag |= mUseSubspace ? uint(BDPTFlags::useSubspaceBDPT) : 0;
        }

        runtimeDirty |= widget.checkbox("use subspace reservoir", mUseReservoir);
        widget.tooltip("reservoir");
        mParams.flag |= mUseReservoir ? uint(BDPTFlags::useReservoir) : 0;
    }

    
    

    if (auto group = widget.group("Spatiotemporal reuse"))
    {
        runtimeDirty |= widget.checkbox("Spatial reuse", spatialReuse);
        widget.tooltip("Enable spatial reuse.");
        mParams.flag |= spatialReuse ? uint(BDPTFlags::spatialReuse) : 0;
    }

    if (auto group = widget.group("RTXDI"))
    {
        dirty |= widget.checkbox("Enabled", mStaticParams.useRTXDI);
        widget.tooltip("Use RTXDI for direct illumination.");
        if (mpRTXDI) dirty |= mpRTXDI->renderUI(group);
    }

    if (auto group = widget.group("Material controls"))
    {
        dirty |= widget.checkbox("Alpha test", mStaticParams.useAlphaTest);
        widget.tooltip("Use alpha testing on non-opaque triangles.");

        dirty |= widget.checkbox("Adjust shading normals on secondary hits", mStaticParams.adjustShadingNormals);
        widget.tooltip("Enables adjustment of the shading normals to reduce the risk of black pixels due to back-facing vectors.\nDoes not apply to primary hits which is configured in GBuffer.", true);

        dirty |= widget.var("Max nested materials", mStaticParams.maxNestedMaterials, 2u, 4u);
        widget.tooltip("Maximum supported number of nested materials.");

        dirty |= widget.checkbox("Use lights in dielectric volumes", mStaticParams.useLightsInDielectricVolumes);
        widget.tooltip("Use lights inside of volumes (transmissive materials). We typically don't want this because lights are occluded by the interface.");

        dirty |= widget.checkbox("Disable caustics", mStaticParams.disableCaustics);
        widget.tooltip("Disable sampling of caustic light paths (i.e. specular events after diffuse events).");

        runtimeDirty |= widget.var("Specular roughness threshold", mParams.specularRoughnessThreshold, 0.f, 1.f);
        widget.tooltip("Specular reflection events are only classified as specular if the material's roughness value is equal or smaller than this threshold. Otherwise they are classified diffuse.");

        dirty |= widget.dropdown("Primary LOD Mode", kLODModeList, reinterpret_cast<uint32_t&>(mStaticParams.primaryLodMode));
        widget.tooltip("Texture LOD mode at primary hit");

        runtimeDirty |= widget.var("TexLOD bias", mParams.lodBias, -16.f, 16.f, 0.01f);
    }
    /*
    if (auto group = widget.group("Denoiser options"))
    {
        dirty |= widget.checkbox("Use NRD demodulation", mStaticParams.useNRDDemodulation);
        widget.tooltip("Global switch for NRD demodulation");
    }
    */
    if (auto group = widget.group("Output options"))
    {
        // Switch to enable/disable path tracer output.
        dirty |= widget.checkbox("Enable output", mEnabled);

        // Controls for output size.
        // When output size requirements change, we'll trigger a graph recompile to update the render pass I/O sizes.
        if (widget.dropdown("Output size", RenderPassHelpers::kIOSizeList, (uint32_t&)mOutputSizeSelection)) requestRecompile();
        if (mOutputSizeSelection == RenderPassHelpers::IOSize::Fixed)
        {
            if (widget.var("Size in pixels", mFixedOutputSize, 32u, 16384u)) requestRecompile();
        }

        dirty |= widget.dropdown("Color format", kColorFormatList, (uint32_t&)mStaticParams.colorFormat);
        widget.tooltip("Selects the color format used for internal per-sample color and denoiser buffers");
    }

    if (dirty) mRecompile = true;
    return dirty || runtimeDirty;
}

bool BDPT::renderDebugUI(Gui::Widgets& widget)
{
    bool dirty = false;

    if (auto group = widget.group("Debugging"))
    {
        dirty |= group.checkbox("Use fixed seed", mParams.useFixedSeed);
        group.tooltip("Forces a fixed random seed for each frame.\n\n"
            "This should produce exactly the same image each frame, which can be useful for debugging.");
        if (mParams.useFixedSeed)
        {
            dirty |= group.var("Seed", mParams.fixedSeed);
        }

        mpPixelDebug->renderUI(group);
    }

    return dirty;
}

void BDPT::renderStatsUI(Gui::Widgets& widget)
{
    if (auto g = widget.group("Statistics"))
    {
        // Show ray stats
        mpPixelStats->renderUI(g);
    }
}

void BDPT::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;

    // Rendering options.
    dirty |= renderRenderingUI(widget);

    // Stats and debug options.
    renderStatsUI(widget);
    dirty |= renderDebugUI(widget);

    if (dirty)
    {
        validateOptions();
        mOptionsChanged = true;
    }
}

bool BDPT::onMouseEvent(const MouseEvent& mouseEvent)
{
    return mpPixelDebug->onMouseEvent(mouseEvent);
}

bool BDPT::beginFrame(RenderContext* pRenderContext, const RenderData& renderData)
{
    const auto& pOutputColor = renderData.getTexture(kOutputColor);
    FALCOR_ASSERT(pOutputColor);

    // Set output frame dimension.
    setFrameDim(uint2(pOutputColor->getWidth(), pOutputColor->getHeight()));

    // Validate all I/O sizes match the expected size.
    // If not, we'll disable the path tracer to give the user a chance to fix the configuration before re-enabling it.
    bool resolutionMismatch = false;
    auto validateChannels = [&](const auto& channels) {
        for (const auto& channel : channels)
        {
            auto pTexture = renderData.getTexture(channel.name);
            if (pTexture && (pTexture->getWidth() != mParams.frameDim.x || pTexture->getHeight() != mParams.frameDim.y)) resolutionMismatch = true;
        }
    };
    validateChannels(kInputChannels);
    validateChannels(kOutputChannels);

    if (mEnabled && resolutionMismatch)
    {
        logError("BDPT I/O sizes don't match. The pass will be disabled.");
        mEnabled = false;
    }

    if (mpScene == nullptr || !mEnabled)
    {
        pRenderContext->clearUAV(pOutputColor->getUAV().get(), float4(0.f));

        // Set refresh flag if changes that affect the output have occured.
        // This is needed to ensure other passes get notified when the path tracer is enabled/disabled.
        if (mOptionsChanged)
        {
            auto& dict = renderData.getDictionary();
            auto flags = dict.getValue(kRenderPassRefreshFlags, Falcor::RenderPassRefreshFlags::None);
            if (mOptionsChanged) flags |= Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
            dict[Falcor::kRenderPassRefreshFlags] = flags;
        }

        return false;
    }

    // Update materials.
    prepareMaterials(pRenderContext);

    // Update the env map and emissive sampler to the current frame.
    bool lightingChanged = prepareLighting(pRenderContext);

    // Prepare RTXDI.
    prepareRTXDI(pRenderContext);
    if (mpRTXDI) mpRTXDI->beginFrame(pRenderContext, mParams.frameDim);

    // Update refresh flag if changes that affect the output have occured.
    auto& dict = renderData.getDictionary();
    if (mOptionsChanged || lightingChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, Falcor::RenderPassRefreshFlags::None);
        if (mOptionsChanged) flags |= Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        if (lightingChanged) flags |= Falcor::RenderPassRefreshFlags::LightingChanged;
        dict[Falcor::kRenderPassRefreshFlags] = flags;
        mOptionsChanged = false;
    }

    // Check if GBuffer has adjusted shading normals enabled.
    bool gbufferAdjustShadingNormals = dict.getValue(Falcor::kRenderPassGBufferAdjustShadingNormals, false);
    if (gbufferAdjustShadingNormals != mGBufferAdjustShadingNormals)
    {
        mGBufferAdjustShadingNormals = gbufferAdjustShadingNormals;
        mRecompile = true;
    }

    // Check if fixed sample count should be used. When the sample count input is connected we load the count from there instead.
    mFixedSampleCount = renderData[kInputSampleCount] == nullptr;
    /*
    // Check if guide data should be generated.
    mOutputGuideData = renderData[kOutputAlbedo] != nullptr || renderData[kOutputSpecularAlbedo] != nullptr
        || renderData[kOutputIndirectAlbedo] != nullptr || renderData[kOutputNormal] != nullptr
        || renderData[kOutputReflectionPosW] != nullptr;

    // Check if NRD data should be generated.
    mOutputNRDData =
        renderData[kOutputNRDDiffuseRadianceHitDist] != nullptr
        || renderData[kOutputNRDSpecularRadianceHitDist] != nullptr
        || renderData[kOutputNRDResidualRadianceHitDist] != nullptr
        || renderData[kOutputNRDEmission] != nullptr
        || renderData[kOutputNRDDiffuseReflectance] != nullptr
        || renderData[kOutputNRDSpecularReflectance] != nullptr;

    // Check if additional NRD data should be generated.
    bool prevOutputNRDAdditionalData = mOutputNRDAdditionalData;
    mOutputNRDAdditionalData =
        renderData[kOutputNRDDeltaReflectionRadianceHitDist] != nullptr
        || renderData[kOutputNRDDeltaTransmissionRadianceHitDist] != nullptr
        || renderData[kOutputNRDDeltaReflectionReflectance] != nullptr
        || renderData[kOutputNRDDeltaReflectionEmission] != nullptr
        || renderData[kOutputNRDDeltaReflectionNormWRoughMaterialID] != nullptr
        || renderData[kOutputNRDDeltaReflectionPathLength] != nullptr
        || renderData[kOutputNRDDeltaReflectionHitDist] != nullptr
        || renderData[kOutputNRDDeltaTransmissionReflectance] != nullptr
        || renderData[kOutputNRDDeltaTransmissionEmission] != nullptr
        || renderData[kOutputNRDDeltaTransmissionNormWRoughMaterialID] != nullptr
        || renderData[kOutputNRDDeltaTransmissionPathLength] != nullptr
        || renderData[kOutputNRDDeltaTransmissionPosW] != nullptr;
    if (mOutputNRDAdditionalData != prevOutputNRDAdditionalData) mRecompile = true;
    */
    // Enable pixel stats if rayCount or pathLength outputs are connected.
    if (renderData[kOutputRayCount] != nullptr || renderData[kOutputPathLength] != nullptr)
    {
        mpPixelStats->setEnabled(true);
    }

    mpPixelStats->beginFrame(pRenderContext, mParams.frameDim);
    mpPixelDebug->beginFrame(pRenderContext, mParams.frameDim);

    // Update the random seed.
    mParams.seed = mParams.useFixedSeed ? mParams.fixedSeed : mParams.frameCount;

    return true;
}

void BDPT::prepareMaterials(RenderContext* pRenderContext)
{
    // This functions checks for material changes and performs any necessary update.
    // For now all we need to do is to trigger a recompile so that the right defines get set.
    // In the future, we might want to do additional material-specific setup here.

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::MaterialsChanged))
    {
        mRecompile = true;
    }
}

bool BDPT::prepareLighting(RenderContext* pRenderContext)
{
    bool lightingChanged = false;

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::RenderSettingsChanged))
    {
        lightingChanged = true;
        mRecompile = true;
    }

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::SDFGridConfigChanged))
    {
        mRecompile = true;
    }

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::EnvMapChanged))
    {
        mpEnvMapSampler = nullptr;
        lightingChanged = true;
        mRecompile = true;
    }

    if (mpScene->useEnvLight())
    {
        if (!mpEnvMapSampler)
        {
            mpEnvMapSampler = EnvMapSampler::create(pRenderContext, mpScene->getEnvMap());
            lightingChanged = true;
            mRecompile = true;
        }
    }
    else
    {
        if (mpEnvMapSampler)
        {
            mpEnvMapSampler = nullptr;
            lightingChanged = true;
            mRecompile = true;
        }
    }

    // Request the light collection if emissive lights are enabled.
    if (mpScene->getRenderSettings().useEmissiveLights)
    {
        mpScene->getLightCollection(pRenderContext);
    }

    if (mpScene->useEmissiveLights())
    {
        if (!mpEmissiveSampler)
        {
            const auto& pLights = mpScene->getLightCollection(pRenderContext);
            FALCOR_ASSERT(pLights && pLights->getActiveLightCount() > 0);
            FALCOR_ASSERT(!mpEmissiveSampler);

            switch (mStaticParams.emissiveSampler)
            {
            case EmissiveLightSamplerType::Uniform:
                mpEmissiveSampler = EmissiveUniformSampler::create(pRenderContext, mpScene);
                break;
            case EmissiveLightSamplerType::LightBVH:
                mpEmissiveSampler = LightBVHSampler::create(pRenderContext, mpScene, mLightBVHOptions);
                break;
            case EmissiveLightSamplerType::Power:
                mpEmissiveSampler = EmissivePowerSampler::create(pRenderContext, mpScene);
                break;
            default:
                throw RuntimeError("Unknown emissive light sampler type");
            }
            lightingChanged = true;
            mRecompile = true;
        }
    }
    else
    {
        if (mpEmissiveSampler)
        {
            // Retain the options for the emissive sampler.
            if (auto lightBVHSampler = std::dynamic_pointer_cast<LightBVHSampler>(mpEmissiveSampler))
            {
                mLightBVHOptions = lightBVHSampler->getOptions();
            }

            mpEmissiveSampler = nullptr;
            lightingChanged = true;
            mRecompile = true;
        }
    }

    if (mpEmissiveSampler)
    {
        lightingChanged |= mpEmissiveSampler->update(pRenderContext);
        auto defines = mpEmissiveSampler->getDefines();
        //if (mpTracePass && mpTracePass->pProgram->addDefines(defines)) mRecompile = true;
        if (mpTraceLightPath && mpTraceLightPath->pProgram->addDefines(defines)) mRecompile = true;
        if (mpTraceCameraPath && mpTraceCameraPath->pProgram->addDefines(defines)) mRecompile = true;
    }

    return lightingChanged;
}

void BDPT::prepareRTXDI(RenderContext* pRenderContext)
{
    if (mStaticParams.useRTXDI)
    {
        if (!mpRTXDI) mpRTXDI = RTXDI::create(mpScene, mRTXDIOptions);

        // Emit warning if enabled while using spp != 1.
        if (!mFixedSampleCount || mStaticParams.samplesPerPixel != 1)
        {
            logWarning("Using RTXDI with samples/pixel != 1 will only generate one RTXDI sample reused for all pixel samples.");
        }
    }
    else
    {
        mpRTXDI = nullptr;
    }
}

void BDPT::updatePrograms()
{
    FALCOR_ASSERT(mpScene);

    if (mRecompile == false) return;

    auto defines = mStaticParams.getDefines(*this);
    auto globalTypeConformances = mpScene->getMaterialSystem()->getTypeConformances();

    // Create trace passes lazily.
    //if (!mpTracePass) mpTracePass = std::make_unique<TracePass>("tracePass", "", mpScene, kTracePassFilename, defines, globalTypeConformances);
    if (!mpTraceLightPath) mpTraceLightPath = std::make_unique<TracePass>("traceLightPath", "", mpScene, kTraceLightPathFilename, defines, globalTypeConformances);
    if (!mpTraceCameraPath) mpTraceCameraPath = std::make_unique<TracePass>("traceCameraPath", "", mpScene, kTraceCameraPathFilename, defines, globalTypeConformances);
    if (!mpRISReuse) mpRISReuse = std::make_unique<CollectPass>(kSpatiotemporalReuseFilename, mpScene);
    /*
    if (mOutputNRDAdditionalData)
    {
        if (!mpTraceDeltaReflectionPass) mpTraceDeltaReflectionPass = std::make_unique<TracePass>("traceDeltaReflectionPass", "DELTA_REFLECTION_PASS", mpScene, defines, globalTypeConformances);
        if (!mpTraceDeltaTransmissionPass) mpTraceDeltaTransmissionPass = std::make_unique<TracePass>("traceDeltaTransmissionPass", "DELTA_TRANSMISSION_PASS", mpScene, defines, globalTypeConformances);
    }
    */
    // Create program vars for trace programs.
    // We only need to set defines for program specialization here. Type conformances have already been setup on construction.
    //mpTracePass->prepareProgram(defines);
    mpTraceLightPath->prepareProgram(defines);
    mpTraceCameraPath->prepareProgram(defines);
    //if (mpTraceDeltaReflectionPass) mpTraceDeltaReflectionPass->prepareProgram(defines);
    //if (mpTraceDeltaTransmissionPass) mpTraceDeltaTransmissionPass->prepareProgram(defines);

    // Create compute passes.
    Program::Desc baseDesc;
    baseDesc.addShaderModules(mpScene->getShaderModules());
    baseDesc.addTypeConformances(globalTypeConformances);
    baseDesc.setShaderModel(kShaderModel);

    //if (!)

    if (!mpGeneratePaths)
    {
        Program::Desc desc = baseDesc;
        desc.addShaderLibrary(kGeneratePathsFilename).csEntry("main");
        mpGeneratePaths = ComputePass::create(desc, defines, false);
    }
    if (!mpReflectTypes)
    {
        Program::Desc desc = baseDesc;
        desc.addShaderLibrary(kReflectTypesFile).csEntry("main");
        mpReflectTypes = ComputePass::create(desc, defines, false);
    }
    
    if (!mpSpatiotemporalReuse)
    {
        Program::Desc desc = baseDesc;
        desc.addShaderLibrary(kSpatiotemporalReuseFilename).csEntry("main");
        mpSpatiotemporalReuse = ComputePass::create(desc, defines, false);
    }
/*
    if (!mpCullingPass)
    {
        Program::Desc desc = baseDesc;
        desc.addShaderLibrary(kCullingPassFilename).csEntry("main");
        mpCullingPass = ComputePass::create(desc, defines, false);
    }
*/
 
    // Perform program specialization.
    // Note that we must use set instead of add functions to replace any stale state.
    auto prepareProgram = [&](Program::SharedPtr program)
    {
        program->setDefines(defines);
    };
    prepareProgram(mpGeneratePaths->getProgram());
    prepareProgram(mpResolvePass->getProgram());
    prepareProgram(mpReflectTypes->getProgram());
    prepareProgram(mpSpatiotemporalReuse->getProgram());

    // Create program vars for the specialized programs.
    mpGeneratePaths->setVars(nullptr);
    mpResolvePass->setVars(nullptr);
    mpReflectTypes->setVars(nullptr);
    mpSpatiotemporalReuse->setVars(nullptr);
    //mpMergePass->setVars(nullptr);
    //mpCullingPass->setVars(nullptr);

    mVarsChanged = true;
    mRecompile = false;
}

Program::DefineList BDPT::StaticParams::getDefines(const BDPT& owner) const
{
    Program::DefineList defines;

    // Path tracer configuration.
    defines.add("SAMPLES_PER_PIXEL", (owner.mFixedSampleCount ? std::to_string(samplesPerPixel) : "0")); // 0 indicates a variable sample count
    defines.add("MAX_SURFACE_BOUNCES", std::to_string(maxSurfaceBounces));
    defines.add("MAX_DIFFUSE_BOUNCES", std::to_string(maxDiffuseBounces));
    defines.add("MAX_SPECULAR_BOUNCES", std::to_string(maxSpecularBounces));
    defines.add("MAX_TRANSMISSON_BOUNCES", std::to_string(maxTransmissionBounces));
    defines.add("ADJUST_SHADING_NORMALS", adjustShadingNormals ? "1" : "0");
    defines.add("USE_BSDF_SAMPLING", useBSDFSampling ? "1" : "0");
    defines.add("USE_NEE", useNEE ? "1" : "0");
    defines.add("USE_MIS", useMIS ? "1" : "0");
    defines.add("USE_RUSSIAN_ROULETTE", useRussianRoulette ? "1" : "0");
    defines.add("USE_RTXDI", useRTXDI ? "1" : "0");
    defines.add("USE_ALPHA_TEST", useAlphaTest ? "1" : "0");
    defines.add("USE_LIGHTS_IN_DIELECTRIC_VOLUMES", useLightsInDielectricVolumes ? "1" : "0");
    defines.add("DISABLE_CAUSTICS", disableCaustics ? "1" : "0");
    defines.add("PRIMARY_LOD_MODE", std::to_string((uint32_t)primaryLodMode));
    //defines.add("USE_NRD_DEMODULATION", useNRDDemodulation ? "1" : "0");
    defines.add("COLOR_FORMAT", std::to_string((uint32_t)colorFormat));
    defines.add("MIS_HEURISTIC", std::to_string((uint32_t)misHeuristic));
    defines.add("MIS_POWER_EXPONENT", std::to_string(misPowerExponent));
    defines.add("LIGHT_PASS_WIDTH", std::to_string(lightPassWidth));
    defines.add("LIGHT_PASS_HEIGHT", std::to_string(lightPassHeight));
    //defines.add("CANDIDATE_NUMBER", std::to_string(candidateNumber));
    defines.add("LOG_SUBSPACE_SIZE", std::to_string(logSubspaceSize));

    // Sampling utilities configuration.
    FALCOR_ASSERT(owner.mpSampleGenerator);
    defines.add(owner.mpSampleGenerator->getDefines());

    if (owner.mpEmissiveSampler) defines.add(owner.mpEmissiveSampler->getDefines());
    if (owner.mpRTXDI) defines.add(owner.mpRTXDI->getDefines());

    defines.add("INTERIOR_LIST_SLOT_COUNT", std::to_string(maxNestedMaterials));

    defines.add("GBUFFER_ADJUST_SHADING_NORMALS", owner.mGBufferAdjustShadingNormals ? "1" : "0");

    // Scene-specific configuration.
    const auto& scene = owner.mpScene;
    if (scene) defines.add(scene->getSceneDefines());
    defines.add("USE_ENV_LIGHT", scene && scene->useEnvLight() ? "1" : "0");
    defines.add("USE_ANALYTIC_LIGHTS", scene && scene->useAnalyticLights() ? "1" : "0");
    defines.add("USE_EMISSIVE_LIGHTS", scene && scene->useEmissiveLights() ? "1" : "0");
    defines.add("USE_CURVES", scene && (scene->hasGeometryType(Scene::GeometryType::Curve)) ? "1" : "0");
    defines.add("USE_SDF_GRIDS", scene && scene->hasGeometryType(Scene::GeometryType::SDFGrid) ? "1" : "0");
    defines.add("USE_HAIR_MATERIAL", scene && scene->getMaterialCountByType(MaterialType::Hair) > 0u ? "1" : "0");

    // Set default (off) values for additional features.
    defines.add("USE_VIEW_DIR", "0");
    defines.add("OUTPUT_GUIDE_DATA", "0");
    defines.add("OUTPUT_NRD_DATA", "0");
    defines.add("OUTPUT_NRD_ADDITIONAL_DATA", "0");

    return defines;
}

BDPT::TracePass::TracePass(const std::string& name, const std::string& passDefine, const Scene::SharedPtr& pScene, const std::string& shaderFile, const Program::DefineList& defines, const Program::TypeConformanceList& globalTypeConformances)
    : name(name)
    , passDefine(passDefine)
{
    const uint32_t kRayTypeScatter = 0;
    const uint32_t kMissScatter = 0;

    RtProgram::Desc desc;
    desc.addShaderModules(pScene->getShaderModules());
    desc.addShaderLibrary(shaderFile);
    desc.setShaderModel(kShaderModel);
    desc.setMaxPayloadSize(160); // This is conservative but the required minimum is 140 bytes.
    desc.setMaxAttributeSize(pScene->getRaytracingMaxAttributeSize());
    desc.setMaxTraceRecursionDepth(3);
    if (!pScene->hasProceduralGeometry()) desc.setPipelineFlags(RtPipelineFlags::SkipProceduralPrimitives);

    // Create ray tracing binding table.
    pBindingTable = RtBindingTable::create(1, 1, pScene->getGeometryCount());

    // Specify entry point for raygen and miss shaders.
    // The raygen shader needs type conformances for *all* materials in the scene.
    // The miss shader doesn't need type conformances as it doesn't access any materials.
    pBindingTable->setRayGen(desc.addRayGen("rayGen", globalTypeConformances));
    pBindingTable->setMiss(kMissScatter, desc.addMiss("scatterMiss"));
    //pBindingTable->setMiss(1, desc.addMiss("RISMiss"));

    // Specify hit group entry points for every combination of geometry and material type.
    // The code for each hit group gets specialized for the actual types it's operating on.
    // First query which material types the scene has.
    auto materialTypes = pScene->getMaterialSystem()->getMaterialTypes();

    for (const auto materialType : materialTypes)
    {
        auto typeConformances = pScene->getMaterialSystem()->getTypeConformances(materialType);

        // Add hit groups for triangles.
        if (auto geometryIDs = pScene->getGeometryIDs(Scene::GeometryType::TriangleMesh, materialType); !geometryIDs.empty())
        {
            auto shaderID = desc.addHitGroup("scatterTriangleClosestHit", "scatterTriangleAnyHit", "", typeConformances, to_string(materialType));
            //auto testRay = desc.addHitGroup("", "RISAnyHit", "", typeConformances, to_string(materialType));
            //pBindingTable->setHitGroup(1, geometryIDs, testRay);
            pBindingTable->setHitGroup(kRayTypeScatter, geometryIDs, shaderID);
        }

        // Add hit groups for displaced triangle meshes.
        if (auto geometryIDs = pScene->getGeometryIDs(Scene::GeometryType::DisplacedTriangleMesh, materialType); !geometryIDs.empty())
        {
            auto shaderID = desc.addHitGroup("scatterDisplacedTriangleMeshClosestHit", "", "displacedTriangleMeshIntersection", typeConformances, to_string(materialType));
            //auto testRay = desc.addHitGroup("", "RISAnyHit", "", typeConformances, to_string(materialType));
            //pBindingTable->setHitGroup(1, geometryIDs, testRay);
            pBindingTable->setHitGroup(kRayTypeScatter, geometryIDs, shaderID);
        }

        // Add hit groups for curves.
        if (auto geometryIDs = pScene->getGeometryIDs(Scene::GeometryType::Curve, materialType); !geometryIDs.empty())
        {
            auto shaderID = desc.addHitGroup("scatterCurveClosestHit", "", "curveIntersection", typeConformances, to_string(materialType));
            //auto testRay = desc.addHitGroup("", "RISAnyHit", "", typeConformances, to_string(materialType));
            //pBindingTable->setHitGroup(1, geometryIDs, testRay);
            pBindingTable->setHitGroup(kRayTypeScatter, geometryIDs, shaderID);
        }

        // Add hit groups for SDF grids.
        if (auto geometryIDs = pScene->getGeometryIDs(Scene::GeometryType::SDFGrid, materialType); !geometryIDs.empty())
        {
            auto shaderID = desc.addHitGroup("scatterSdfGridClosestHit", "", "sdfGridIntersection", typeConformances, to_string(materialType));
            //auto testRay = desc.addHitGroup("", "RISAnyHit", "", typeConformances, to_string(materialType));
            //pBindingTable->setHitGroup(1, geometryIDs, testRay);
            pBindingTable->setHitGroup(kRayTypeScatter, geometryIDs, shaderID);
        }
    }

    //auto shaderID = desc.addHitGroup("scatterTriangleClosestHit", "scatterTriangleAnyHit", "", typeConformances, to_string(materialType));
    
    pProgram = RtProgram::create(desc, defines);
    
}

void BDPT::TracePass::prepareProgram(const Program::DefineList& defines)
{
    FALCOR_ASSERT(pProgram != nullptr && pBindingTable != nullptr);
    pProgram->setDefines(defines);
    if (!passDefine.empty()) pProgram->addDefine(passDefine);
    pVars = RtProgramVars::create(pProgram, pBindingTable);
    //logWarning(std::to_string(pVars->getRayTypeCount()));
}

void BDPT::prepareResources(RenderContext* pRenderContext, const RenderData& renderData)
{
    /*
    // Compute allocation requirements for paths and output samples.
    // Note that the sample buffers are padded to whole tiles, while the max path count depends on actual frame dimension.
    // If we don't have a fixed sample count, assume the worst case.
    uint32_t spp = mFixedSampleCount ? mStaticParams.samplesPerPixel : kMaxSamplesPerPixel;
    uint32_t tileCount = mParams.screenTiles.x * mParams.screenTiles.y;
    const uint32_t sampleCount = tileCount * kScreenTileDim.x * kScreenTileDim.y * spp;
    const uint32_t screenPixelCount = mParams.frameDim.x * mParams.frameDim.y;
    const uint32_t pathCount = screenPixelCount * spp;


    // Allocate output sample offset buffer if needed.
    // This buffer stores the output offset to where the samples for each pixel are stored consecutively.
    // The offsets are local to the current tile, so 16-bit format is sufficient and reduces bandwidth usage.
    if (!mFixedSampleCount)
    {
        if (!mpSampleOffset || mpSampleOffset->getWidth() != mParams.frameDim.x || mpSampleOffset->getHeight() != mParams.frameDim.y)
        {
            FALCOR_ASSERT(kScreenTileDim.x * kScreenTileDim.y * kMaxSamplesPerPixel <= (1u << 16));
            mpSampleOffset = Texture::create2D(mParams.frameDim.x, mParams.frameDim.y, ResourceFormat::R16Uint, 1, 1, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
            mVarsChanged = true;
        }
    }
    */
    auto var = mpReflectTypes->getRootVar();
    /*
    // Allocate per-sample buffers.
    // For the special case of fixed 1 spp, the output is written out directly and this buffer is not needed.
    if (!mFixedSampleCount || mStaticParams.samplesPerPixel > 1)
    {
        if (!mpSampleColor || mpSampleColor->getElementCount() < sampleCount || mVarsChanged)
        {
            mpSampleColor = Buffer::createStructured(var["sampleColor"], sampleCount, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
            mVarsChanged = true;
        }
    }
    */
    //TODO: change height
    uint32_t lightVertexElementCount = mStaticParams.lightPassWidth * mStaticParams.lightPassHeight * mStaticParams.maxSurfaceBounces;
    uint32_t cameraVertexElementCount = kMaxFrameDimension * kMaxFrameDimensionY * mStaticParams.maxSurfaceBounces;
    if (!mpLightPathVertexBuffer) mpLightPathVertexBuffer = Buffer::createStructured(var["LightPathsVertexsBuffer"], lightVertexElementCount, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
    if (!mpLightPathsIndexBuffer) mpLightPathsIndexBuffer = Buffer::createStructured(var["LightPathsIndexBuffer"], lightVertexElementCount);
    if (!mpLightPathsVertexsPositionBuffer) mpLightPathsVertexsPositionBuffer = Buffer::createStructured(var["LightPathsVertexsPositionBuffer"], lightVertexElementCount, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
    //if (!mpCameraPathsVertexsReservoirBuffer) mpCameraPathsVertexsReservoirBuffer = Buffer::createStructured(var["CameraPathsVertexsReservoirBuffer"], cameraVertexElementCount, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
    //if (!mpDstCameraPathsVertexsReservoirBuffer) mpDstCameraPathsVertexsReservoirBuffer = Buffer::createStructured(var["DstCameraPathsVertexsReservoirBuffer"], cameraVertexElementCount, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
    if (!mpOutput) mpOutput = Texture::create2D(mParams.frameDim.x, mParams.frameDim.y, ResourceFormat::RGBA32Float, 1, 1, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);

    uint subspaceSize = 1 << mStaticParams.logSubspaceSize;
    if (!mpSubspaceWeight[0]) {
        mpSubspaceWeight[0] = Texture::create2D(subspaceSize, subspaceSize, ResourceFormat::R32Uint, 1, 1, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
        mpMergePass["subspaceWeight"] = mpSubspaceWeight[0];
    }
    if (!mpSubspaceWeight[1]) {
        mpSubspaceWeight[1] = Texture::create2D(subspaceSize, subspaceSize, ResourceFormat::R32Uint, 1, 1, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
        mpMergePass["nextSubspaceWeight"] = mpSubspaceWeight[1];
        mpPrefixSumPass["nextSubspaceWeight"] = mpSubspaceWeight[1];
    }
    if (!mpSubspaceCount[0]) {
        mpSubspaceCount[0] = Texture::create2D(subspaceSize, subspaceSize, ResourceFormat::R32Uint, 1, 1, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
        mpMergePass["subspaceCount"] = mpSubspaceCount[0];
    }
    if (!mpSubspaceCount[1]) {
        mpSubspaceCount[1] = Texture::create2D(subspaceSize, subspaceSize, ResourceFormat::R32Uint, 1, 1, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
        mpMergePass["totalCount"] = mpSubspaceCount[1];
        mpPrefixSumPass["totalCount"] = mpSubspaceCount[1];
    }
    
    if (!mpPrefixOfWeight) {
        mpPrefixOfWeight = Texture::create2D(subspaceSize, subspaceSize, ResourceFormat::R32Float, 1, 1, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
        mpPrefixSumPass["prefixSum"] = mpPrefixOfWeight;
        mpMergePass["prefixSum"] = mpPrefixOfWeight;
    }
    
    if (!mpSumOfWeight) {
        mpSumOfWeight = Texture::create1D(subspaceSize, ResourceFormat::R32Float, 1, 1, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
        mpPrefixSumPass["sumWeight"] = mpSumOfWeight;
        mpMergePass["sumWeight"] = mpSumOfWeight;
    }

    if (!mpPrefixOfCount) {
        mpPrefixOfCount = Texture::create2D(subspaceSize, subspaceSize, ResourceFormat::R32Uint, 1, 1, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
        mpPrefixSumPass["prefixSumOfCount"] = mpPrefixOfCount;
    }

    if (!mpSumOfCount) {
        mpSumOfCount = Texture::create1D(subspaceSize, ResourceFormat::R32Uint, 1, 1, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
        mpPrefixSumPass["sumCount"] = mpSumOfCount;
    }

    if (!mpSubspaceReservoir) {
        mpSubspaceReservoir = std::make_shared<subspaceReservoir>(subspaceSize);
        mpSubspaceReservoir->clear(pRenderContext);
    }
    if (!mpSubspaceSecondaryMoment) {
        mpSubspaceSecondaryMoment = Texture::create2D(subspaceSize, subspaceSize, ResourceFormat::R32Uint, 1, 1, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
        mpMergePass["subspaceSecondaryMoment"] = mpSubspaceSecondaryMoment;
        mpPrefixSumPass["subspaceSecondaryMoment"] = mpSubspaceSecondaryMoment;
    }
    if (!mpPrefixOfSecondaryMoment) {
        mpPrefixOfSecondaryMoment= Texture::create2D(subspaceSize, subspaceSize, ResourceFormat::R32Float, 1, 1, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
        mpPrefixSumPass["prefixSumOfSecondaryMoment"] = mpPrefixOfSecondaryMoment;
    }
    if (!mpSumOfSecondaryMoment) {
        mpSumOfSecondaryMoment = Texture::create1D(subspaceSize, ResourceFormat::R32Float, 1, 1, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
        mpPrefixSumPass["sumSecondaryMoment"] = mpSumOfSecondaryMoment;
    }
    if (!mpMaxVariance) {
        mpMaxVariance = Texture::create1D(subspaceSize, ResourceFormat::R32Float, 1, 1, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
        mpPrefixSumPass["maxVariance"] = mpMaxVariance;
    }
    /*
    if (!mpGatherPointVbuffer) mpGatherPointVbuffer = Texture::create2D(mParams.frameDim.x, mParams.frameDim.y, ResourceFormat::RGBA32Uint, 1, 1, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
    if (!mpGatherPointThp) mpGatherPointThp = Texture::create2D(mParams.frameDim.x, mParams.frameDim.y, ResourceFormat::RGBA32Float, 1, 1, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
    if (!mpGatherPointDir) mpGatherPointDir = Texture::create2D(mParams.frameDim.x, mParams.frameDim.y, ResourceFormat::RGBA32Float, 1, 1, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
    uint32_t hashSize = 1 << mStaticParams.cullingHashBufferSizeBytes;
    hashSize = static_cast<uint>(sqrt(hashSize));
    if (!mpHashBuffer) mpHashBuffer = Texture::create2D(hashSize, hashSize, ResourceFormat::R8Uint, 1, 1, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);

    // Caustic
    {
        mpCausticBuffers.maxSize = lightVertexElementCount;
        if (!mpCausticBuffers.infoFlux) mpCausticBuffers.infoFlux = Buffer::createStructured(sizeof(float4), mpCausticBuffers.maxSize);
        if (!mpCausticBuffers.infoDir) mpCausticBuffers.infoDir = Buffer::createStructured(sizeof(float4), mpCausticBuffers.maxSize, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
        if (!mpCausticBuffers.aabb) mpCausticBuffers.aabb = Buffer::createStructured(sizeof(D3D12_RAYTRACING_AABB), mpCausticBuffers.maxSize);
    }

    // Global
    {
        mpGlobalBuffers.maxSize = lightVertexElementCount;
        if (!mpGlobalBuffers.infoFlux) mpGlobalBuffers.infoFlux = Buffer::createStructured(sizeof(float4), mpGlobalBuffers.maxSize);
        if (!mpGlobalBuffers.infoDir) mpGlobalBuffers.infoDir = Buffer::createStructured(sizeof(float4), mpGlobalBuffers.maxSize, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
        if (!mpGlobalBuffers.aabb) mpGlobalBuffers.aabb = Buffer::createStructured(sizeof(D3D12_RAYTRACING_AABB), mpGlobalBuffers.maxSize);
    }
*/
    
    if (!mpSortPass) mpSortPass = std::make_unique<Bitonic64Sort>(mpLightPathsIndexBuffer, 0);
    if (!mpTreeBuilder) mpTreeBuilder = std::make_unique<VertexTreeBuilder>(mpLightPathsIndexBuffer, mpLightPathsVertexsPositionBuffer, lightVertexElementCount);
    
    if (!mpPrevGatherPoints) mpPrevGatherPoints = std::make_shared<GatherPointInfo>(mParams.frameDim.x, mParams.frameDim.y);
    if (!mpGatherPoints) mpGatherPoints = std::make_shared<GatherPointInfo>(mParams.frameDim.x, mParams.frameDim.y);
    
    if (!mpPairs) mpPairs = std::make_shared<SamplePairs>(cameraVertexElementCount);
    if (!mpPrevPairs) mpPrevPairs = std::make_shared<SamplePairs>(cameraVertexElementCount);
    //if (!mpAABB) mpAABB = Buffer::createStructured(sizeof(D3D12_RAYTRACING_AABB), cameraVertexElementCount);

    //if (!mpPathPos) mpPathPos = Texture::create3D(mParams.frameDim.x, mParams.frameDim.y, mStaticParams.maxSurfaceBounces, ResourceFormat::RGBA32Uint, 1, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
    //if (!mpPathRadiance) mpPathRadiance = Texture::create3D(mParams.frameDim.x, mParams.frameDim.y, mStaticParams.maxSurfaceBounces, ResourceFormat::RGBA32Float, 1, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
    //if (!mpPathHitInfo) mpPathHitInfo = Texture::create3D(mParams.frameDim.x, mParams.frameDim.y, mStaticParams.maxSurfaceBounces, ResourceFormat::RGBA32Uint, 1, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);

    //if (!mpCameraPathsIndexBuffer) mpCameraPathsIndexBuffer = Buffer::createStructured(var["CameraPathsIndexBuffer"], cameraVertexElementCount);
    //if (!mpMCounter) mpMCounter = Buffer::createStructured(var["MCounter"], mStaticParams.maxSurfaceBounces + 2, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
    //pRenderContext->clearUAV(mpMCounter->getUAV().get(), zero4);
    /*
    if (mOutputGuideData && (!mpSampleGuideData || mpSampleGuideData->getElementCount() < sampleCount || mVarsChanged))
    {
        mpSampleGuideData = Buffer::createStructured(var["sampleGuideData"], sampleCount, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
        mVarsChanged = true;
    }

    if (mOutputNRDData && (!mpSampleNRDRadiance || mpSampleNRDRadiance->getElementCount() < sampleCount || mVarsChanged))
    {
        mpSampleNRDRadiance = Buffer::createStructured(var["sampleNRDRadiance"], sampleCount, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
        mpSampleNRDHitDist = Buffer::createStructured(var["sampleNRDHitDist"], sampleCount, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
        mpSampleNRDEmission = Buffer::createStructured(var["sampleNRDEmission"], sampleCount, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
        mpSampleNRDReflectance = Buffer::createStructured(var["sampleNRDReflectance"], sampleCount, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
        mVarsChanged = true;
    }
    */
}

void BDPT::preparePathTracer(const RenderData& renderData)
{
    // Create path tracer parameter block if needed.
    if (!mpPathTracerBlock || mVarsChanged)
    {
        auto reflector = mpReflectTypes->getProgram()->getReflector()->getParameterBlock("pathTracer");
        mpPathTracerBlock = ParameterBlock::create(reflector);
        FALCOR_ASSERT(mpPathTracerBlock);
        mVarsChanged = true;
    }

    // Bind resources.
    auto var = mpPathTracerBlock->getRootVar();
    float3 dimension = mpScene->getSceneBounds().maxPoint - mpScene->getSceneBounds().minPoint;
    var["corner"] = mpScene->getSceneBounds().minPoint;
    var["dimension"] = dimension;
    radius = 0.005 * sqrt(dimension.x * dimension.x + dimension.y * dimension.y + dimension.z * dimension.z);
    var["globalRadius"] = radius;

    if (mUseVertexMerge) {
        var["InputFluxAndNumber"] = mpPrevGatherPoints->mpFluxAndNumber;
        var["InputNormalAndRadii"] = mpPrevGatherPoints->mpNormalAndRadii;
        var["InputPosAndNewRadii"] = mpPrevGatherPoints->mpPosAndNewRadii;
        var["InputIteration"] = mpPrevGatherPoints->mpIteration;

        var["OutputFluxAndNumber"] = mpGatherPoints->mpFluxAndNumber;
        var["OutputNormalAndRadii"] = mpGatherPoints->mpNormalAndRadii;
        var["OutputPosAndNewRadii"] = mpGatherPoints->mpPosAndNewRadii;
        var["OutputIteration"] = mpGatherPoints->mpIteration;
    }
    
    //mpScene->getParameterBlock()->acc
    
    if (mVarsChanged) {
        var["SubspaceWeight"] = mpSubspaceWeight[0];
        var["prevSubspaceWeight"] = mpSubspaceWeight[1];
        var["SubspaceCount"] = mpSubspaceCount[0];
        var["prefixSum"] = mpPrefixOfWeight;
        var["weightSum"] = mpSumOfWeight;
        var["countSum"] = mpSumOfCount;
        var["prefixSumOfCount"] = mpPrefixOfCount;

        var["SubspaceSecondaryMoment"] = mpSubspaceSecondaryMoment;
        var["prefixSumOfSecondaryMoment"] = mpPrefixOfSecondaryMoment;
        var["secondaryMomentSum"] = mpSumOfSecondaryMoment;

        var["maxVariance"] = mpMaxVariance;

        mpSubspaceReservoir->bind(var);
    }

    
    //var["motionVec"] = renderData.getTexture(kInputMotionVectors);
    //var["hitPointAABB"] = mpAABB;

    /*
    if (mVarsChanged) {
        var["gatherPointVbuffer"] = mpGatherPointVbuffer;
        var["gatherPointThp"] = mpGatherPointThp;
        var["gatherPointDir"] = mpGatherPointDir;
        var["hashBuffer"] = mpHashBuffer;
    }
    
    radius = 0.005 * sqrt(dimension.x * dimension.x + dimension.y * dimension.y + dimension.z * dimension.z);
    // 

    uint hashSize = 1 << mStaticParams.cullingHashBufferSizeBytes;
    var["gHashScaleFactor"] = 1.0f / (radius * 2);
    var["gHashSize"] = hashSize;
    var["gYExtend"] = static_cast<uint>(sqrt(hashSize));
    var["gGlobalRadius"] = radius;
    */

    /*
    var["samplePositionI"] = mpPrevPairs->samplePosition;
    var["hitPointPositionI"] = mpPrevPairs->hitPointPosition;
    var["normalI"] = mpPrevPairs->normal;
    var["radianceI"] = mpPrevPairs->radiance;
    var["ReservoirI"] = mpPrevPairs->reservoir;

    var["samplePositionO"] = mpPairs->samplePosition;
    var["hitPointPositionO"] = mpPairs->hitPointPosition;
    var["normalO"] = mpPairs->normal;
    var["radianceO"] = mpPairs->radiance;
    var["ReservoirO"] = mpPairs->reservoir;
    */
    /*
    if (mVarsChanged) {
        var["pathPos"] = mpPathPos;
        var["pathRadiance"] = mpPathRadiance;
        var["pathHitInfo"] = mpPathHitInfo;
    }
    */
    setShaderData(var, renderData);
}

void BDPT::setShaderData(const ShaderVar& var, const RenderData& renderData, bool useLightSampling) const
{
    // Bind static resources that don't change per frame.
    if (mVarsChanged)
    {
        if (useLightSampling && mpEnvMapSampler) mpEnvMapSampler->setShaderData(var["envMapSampler"]);

        //var["sampleOffset"] = mpSampleOffset; // Can be nullptr
        //var["sampleColor"] = mpSampleColor;
        var["LightPathsVertexsBuffer"] = mpLightPathVertexBuffer;
        var["LightPathsIndexBuffer"] = mpLightPathsIndexBuffer;
        var["LightPathsVertexsPositionBuffer"] = mpLightPathsVertexsPositionBuffer;
        
        var["nodes"] = mpTreeBuilder->getTree();
        //var["CameraPathsVertexsReservoirBuffer"] = mpCameraPathsVertexsReservoirBuffer;
        //var["CameraPathsIndexBuffer"] = mpCameraPathsIndexBuffer;
        //var["MCounter"] = mpMCounter;
        //var["sampleGuideData"] = mpSampleGuideData;
    }

    // Bind runtime data.
    //setNRDData(var["outputNRD"], renderData);

    Texture::SharedPtr pViewDir;
    if (mpScene->getCamera()->getApertureRadius() > 0.f)
    {
        pViewDir = renderData.getTexture(kInputViewDir);
        if (!pViewDir) logWarning("Depth-of-field requires the '{}' input. Expect incorrect rendering.", kInputViewDir);
    }

    Texture::SharedPtr pSampleCount;
    if (!mFixedSampleCount)
    {
        pSampleCount = renderData.getTexture(kInputSampleCount);
        if (!pSampleCount) throw RuntimeError("PathTracer: Missing sample count input texture");
    }

    var["params"].setBlob(mParams);
    var["vbuffer"] = renderData.getTexture(kInputVBuffer);
    var["viewDir"] = pViewDir; // Can be nullptr
    var["sampleCount"] = pSampleCount; // Can be nullptr
    var["outputColor"] = mpOutput;//renderData.getTexture(kOutputColor);

    if (useLightSampling && mpEmissiveSampler)
    {
        // TODO: Do we have to bind this every frame?
        mpEmissiveSampler->setShaderData(var["emissiveSampler"]);
    }
}

void BDPT::generatePaths(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE("generatePaths");

    // Check shader assumptions.
    // We launch one thread group per screen tile, with threads linearly indexed.
    const uint32_t tileSize = kScreenTileDim.x * kScreenTileDim.y;
    FALCOR_ASSERT(kScreenTileDim.x == 16 && kScreenTileDim.y == 16); // TODO: Remove this temporary limitation when Slang bug has been fixed, see comments in shader.
    FALCOR_ASSERT(kScreenTileBits.x <= 4 && kScreenTileBits.y <= 4); // Since we use 8-bit deinterleave.
    FALCOR_ASSERT(mpGeneratePaths->getThreadGroupSize().x == tileSize);
    FALCOR_ASSERT(mpGeneratePaths->getThreadGroupSize().y == 1 && mpGeneratePaths->getThreadGroupSize().z == 1);

    // Additional specialization. This shouldn't change resource declarations.
    mpGeneratePaths->addDefine("USE_VIEW_DIR", (mpScene->getCamera()->getApertureRadius() > 0 && renderData[kInputViewDir] != nullptr) ? "1" : "0");
    //mpGeneratePaths->addDefine("OUTPUT_GUIDE_DATA", mOutputGuideData ? "1" : "0");
    //mpGeneratePaths->addDefine("OUTPUT_NRD_DATA", mOutputNRDData ? "1" : "0");
    //mpGeneratePaths->addDefine("OUTPUT_NRD_ADDITIONAL_DATA", mOutputNRDAdditionalData ? "1" : "0");

    // Bind resources.
    auto var = mpGeneratePaths->getRootVar()["CB"]["gPathGenerator"];
    setShaderData(var, renderData, false);

    mpGeneratePaths["gScene"] = mpScene->getParameterBlock();

    if (mpRTXDI) mpRTXDI->setShaderData(mpGeneratePaths->getRootVar());

    // Launch one thread per pixel.
    // The dimensions are padded to whole tiles to allow re-indexing the threads in the shader.
    mpGeneratePaths->execute(pRenderContext, { mParams.screenTiles.x * tileSize, mParams.screenTiles.y, 1u });
}

void BDPT::spatiotemporalReuse(RenderContext* pRenderContext, const RenderData& renderData) {
    FALCOR_PROFILE("spatiotemporalReuse");

    auto var = mpSpatiotemporalReuse->getRootVar()["CB"]["gSpatiotemproalReuse"];
    //var["LightPathsVertexsBuffer"] = mpLightPathVertexBuffer;
    //var["SrcCameraPathsVertexsReservoirBuffer"] = mpCameraPathsVertexsReservoirBuffer;
    //var["DstCameraPathsVertexsReservoirBuffer"] = mpDstCameraPathsVertexsReservoirBuffer;
    var["output"] = mpOutput;

    //var["params"].setBlob(mParams);
    //Texture3D<uint4> pathPos;       // xyz := pos, w := index
    //Texture3D<float4> pathRadiance; // xyz := radiance, w := de;
    //Texture3D<uint4> pathHitInfo;
   // var["pathPos"] =
    if (mVarsChanged) {
        var["pathPos"] = mpPathPos;
        var["pathRadiance"] = mpPathRadiance;
        var["pathHitInfo"] = mpPathHitInfo;
    }

    var["outputColor"] = renderData.getTexture(kOutputColor);

    mpSpatiotemporalReuse["gScene"] = mpScene->getParameterBlock();

    mpSpatiotemporalReuse->execute(pRenderContext, { mParams.frameDim, 1u });
}

void BDPT::buildSubspaceWeightMatrix(RenderContext* pRenderContext, const RenderData& renderData) {
    //mpMergePass["output"] = renderData.getTexture(kOutputColor);
    mpMergePass["CB"]["frameCount"] = 0;// mParams.frameCount;
    uint groupSize = 1 << (mStaticParams.logSubspaceSize - 1);
    uint texSize = groupSize << 1;
    mpMergePass->execute(pRenderContext, { texSize,texSize,1u });
    pRenderContext->uavBarrier(mpSubspaceWeight[1].get());
    pRenderContext->uavBarrier(mpSubspaceCount[1].get());
    pRenderContext->uavBarrier(mpSubspaceSecondaryMoment.get());

    //mpSubspaceWeightComputeState->setProgram(mpPrefixSumPass);
    //pRenderContext->dispatch(mpSubspaceWeightComputeState.get(), mpPrefixSumVar.get(), { 1024u,1u,1u });
    mpPrefixSumPass["CB"]["frameCount"] = mParams.frameCount;
    mpPrefixSumPass->execute(pRenderContext, uint3(groupSize, texSize, 1u));
    pRenderContext->uavBarrier(mpPrefixOfWeight.get());
    pRenderContext->uavBarrier(mpSumOfWeight.get());
    pRenderContext->uavBarrier(mpPrefixOfCount.get());
    pRenderContext->uavBarrier(mpSumOfCount.get());
    pRenderContext->uavBarrier(mpSubspaceSecondaryMoment.get());
    pRenderContext->uavBarrier(mpPrefixOfSecondaryMoment.get());
    pRenderContext->uavBarrier(mpSumOfSecondaryMoment.get());
    pRenderContext->uavBarrier(mpMaxVariance.get());
}

void BDPT::tracePass(RenderContext* pRenderContext, const RenderData& renderData, TracePass& tracePass, uint2 dim)
{
    FALCOR_PROFILE(tracePass.name);

    FALCOR_ASSERT(tracePass.pProgram != nullptr && tracePass.pBindingTable != nullptr && tracePass.pVars != nullptr);

    // Additional specialization. This shouldn't change resource declarations.
    tracePass.pProgram->addDefine("USE_VIEW_DIR", (mpScene->getCamera()->getApertureRadius() > 0 && renderData[kInputViewDir] != nullptr) ? "1" : "0");
    //tracePass.pProgram->addDefine("OUTPUT_GUIDE_DATA", mOutputGuideData ? "1" : "0");
    //tracePass.pProgram->addDefine("OUTPUT_NRD_DATA", mOutputNRDData ? "1" : "0");
    //tracePass.pProgram->addDefine("OUTPUT_NRD_ADDITIONAL_DATA", mOutputNRDAdditionalData ? "1" : "0");

    // Bind global resources.
    auto var = tracePass.pVars->getRootVar();
    mpScene->setRaytracingShaderData(pRenderContext, var);

    if (mVarsChanged) mpSampleGenerator->setShaderData(var);
    if (mpRTXDI) mpRTXDI->setShaderData(var);

    mpPixelStats->prepareProgram(tracePass.pProgram, var);
    mpPixelDebug->prepareProgram(tracePass.pProgram, var);

    // Bind the path tracer.
    var["gPathTracer"] = mpPathTracerBlock;
    bool tlasValid = var["gHitPointAS"].setSrv(mPhotonTlas.pSrv);
    //mPhotonTlas.pTlas->
    //logWarning(std::to_string(tlasValid));

    // Full screen dispatch.
    mpScene->raytrace(pRenderContext, tracePass.pProgram.get(), tracePass.pVars, uint3(dim, 1));
}

void BDPT::resolvePass(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (/*!mOutputGuideData && !mOutputNRDData && mFixedSampleCount && */mStaticParams.samplesPerPixel == 1) return;

    FALCOR_PROFILE("resolvePass");

    // This pass is executed when multiple samples per pixel are used.
    // We launch one thread per pixel that computes the resolved color by iterating over the samples.
    // The samples are arranged in tiles with pixels in Morton order, with samples stored consecutively for each pixel.
    // With adaptive sampling, an extra sample offset lookup table computed by the path generation pass is used to
    // locate the samples for each pixel.

    // Additional specialization. This shouldn't change resource declarations.
    //mpResolvePass->addDefine("OUTPUT_GUIDE_DATA", mOutputGuideData ? "1" : "0");
    //mpResolvePass->addDefine("OUTPUT_NRD_DATA", mOutputNRDData ? "1" : "0");

    // Bind resources.
    auto var = mpResolvePass->getRootVar()["CB"]["gResolvePass"];
    var["params"].setBlob(mParams);
    var["sampleCount"] = renderData.getTexture(kInputSampleCount); // Can be nullptr
    var["outputColor"] = renderData.getTexture(kOutputColor);
    var["outputAlbedo"] = renderData.getTexture(kOutputAlbedo);
    var["outputSpecularAlbedo"] = renderData.getTexture(kOutputSpecularAlbedo);
    var["outputIndirectAlbedo"] = renderData.getTexture(kOutputIndirectAlbedo);
    var["outputNormal"] = renderData.getTexture(kOutputNormal);
    var["outputReflectionPosW"] = renderData.getTexture(kOutputReflectionPosW);
    /*
    var["outputNRDDiffuseRadianceHitDist"] = renderData.getTexture(kOutputNRDDiffuseRadianceHitDist);
    var["outputNRDSpecularRadianceHitDist"] = renderData.getTexture(kOutputNRDSpecularRadianceHitDist);
    var["outputNRDDeltaReflectionRadianceHitDist"] = renderData.getTexture(kOutputNRDDeltaReflectionRadianceHitDist);
    var["outputNRDDeltaTransmissionRadianceHitDist"] = renderData.getTexture(kOutputNRDDeltaTransmissionRadianceHitDist);
    var["outputNRDResidualRadianceHitDist"] = renderData.getTexture(kOutputNRDResidualRadianceHitDist);
    */
    if (mVarsChanged)
    {
        var["sampleOffset"] = mpSampleOffset; // Can be nullptr
        var["sampleColor"] = mpSampleColor;
        //var["LightPathsVertexsBuffer"] = mpLightPathVertexBuffer;
        /*
        var["sampleGuideData"] = mpSampleGuideData;
        var["sampleNRDRadiance"] = mpSampleNRDRadiance;
        var["sampleNRDHitDist"] = mpSampleNRDHitDist;
        var["sampleNRDEmission"] = mpSampleNRDEmission;
        var["sampleNRDReflectance"] = mpSampleNRDReflectance;
        */
    }

    // Launch one thread per pixel.
    mpResolvePass->execute(pRenderContext, { mParams.frameDim, 1u });
}

void BDPT::endFrame(RenderContext* pRenderContext, const RenderData& renderData)
{
    mpPixelStats->endFrame(pRenderContext);
    mpPixelDebug->endFrame(pRenderContext);

    auto copyTexture = [pRenderContext](Texture* pDst, const Texture* pSrc)
    {
        if (pDst && pSrc)
        {
            FALCOR_ASSERT(pDst && pSrc);
            FALCOR_ASSERT(pDst->getFormat() == pSrc->getFormat());
            FALCOR_ASSERT(pDst->getWidth() == pSrc->getWidth() && pDst->getHeight() == pSrc->getHeight());
            pRenderContext->copyResource(pDst, pSrc);
        }
        else if (pDst)
        {
            pRenderContext->clearUAV(pDst->getUAV().get(), uint4(0, 0, 0, 0));
        }
    };

    // Copy pixel stats to outputs if available.
    copyTexture(renderData.getTexture(kOutputRayCount).get(), mpPixelStats->getRayCountTexture(pRenderContext).get());
    copyTexture(renderData.getTexture(kOutputPathLength).get(), mpPixelStats->getPathLengthTexture().get());

    if (mpRTXDI) mpRTXDI->endFrame(pRenderContext);
    //mpLightPathsIndexBuffer->getUAVCounter()->
    //uint32_t zero = 0;
    //pRenderContext->clearUAV(mpCameraPathsVertexsReservoirBuffer->getUAV().get(), zero4);
    pRenderContext->clearUAV(mpOutput->getUAV().get(), zero4);
    //std::swap(mpPairs, mpPrevPairs);
    //mpPairs->clear(pRenderContext);
    if (mUseVertexMerge) {
        std::swap(mpPrevGatherPoints, mpGatherPoints);
    }
    
    pRenderContext->clearUAV(mpSubspaceWeight[0]->getUAV().get(), zero4);
    pRenderContext->clearUAV(mpSubspaceCount[0]->getUAV().get(), zero4);
    //mpGatherPoints->clear(pRenderContext);
    //pRenderContext->clearUAV(mpHashBuffer->getUAV().get(), zero4);

    mVarsChanged = false;
    //mpScene->getMesh(0).getTriangleCount
    //logWarning(std::to_string(mpScene-> ));
    mParams.frameCount++;
}
