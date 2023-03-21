import Utils.Geometry.GeometryHelpers;
import Scene.RaytracingInline;
import Rendering.Lights.LightHelpers;
import LoadShadingData;
//import NRDHelpers;
import PathTracer;
import PathState;
import ColorType;
import BDPTParams;
import PathData;

struct SpatiotemproalReuse
{
    PathTracerParams params;

    StructuredBuffer<VertexInfo> LightPathsVertexsBuffer; ///< Output the Vertex of all the light paths.
    StructuredBuffer<CameraVertex> SrcCameraPathsVertexsReservoirBuffer;
    Texture2D<float4> output;

    RWStructuredBuffer<CameraVertex> DstCameraPathsVertexsReservoirBuffer;

    RWTexture2D<float4> outputColor; ///< Output color.

    const int screenSpaceReuseDomain = 3;

    inline uint cameraPathOffset(uint2 pixel, int offset = 0) {
        uint base = (pixel.y * params.frameDim.x + pixel.x) * kMaxSurfaceBounces;
        return base + offset;
    }

    float3 RISWeight(Vertex cameraPathEnd, Vertex sample, inout SampleGenerator sg, bool fr0 = true, bool g = true, bool fr1 = true) {
        // RISBDPT
        let lod = ExplicitLodTextureSampler(0);

        float3 result = sample.beta;

        result *= g ? G(cameraPathEnd, sample) : 1;

        const IBSDF bsdf0 = gScene.materials.getBSDF(cameraPathEnd.sd, lod);
        const IBSDF bsdf1 = gScene.materials.getBSDF(sample.sd, lod);
        float3 toSample = normalize(sample.sd.posW - cameraPathEnd.sd.posW);

        const uint lobes0 = bsdf0.getLobes(cameraPathEnd.sd);
        const uint lobes1 = bsdf1.getLobes(sample.sd);

        result *= fr0 ? bsdf0.eval(cameraPathEnd.sd, toSample, sg) : 1;

        result *= fr1 ? bsdf1.eval(sample.sd, -toSample, sg) : 1;
        // result *= abs(dot(sample.sd.N, wo));

        return result;
    }

    inline float3 G(Vertex v0, Vertex v1) {
        float3 d = v0.sd.posW - v1.sd.posW;
        float g = 1 / dot(d, d);
        // d *= sqrt(g);
        // g *= v0.type == VertexType::SurfaceOrMedium ? abs(dot(v0.sd.N, d)) : 1;
        // g *= v1.type == VertexType::SurfaceOrMedium ? abs(dot(v1.sd.N, d)) : 1;
        return g;
    }

    Ray getVisibiliyTestRay(Vertex v, Vertex sample) {
        float3 cameraSideOrigin = computeRayOrigin(v.sd.posW, dot(v.sd.faceN, sample.sd.posW - v.sd.posW) >= 0 ? v.sd.faceN : -v.sd.faceN);
        float3 lightSideOrigin = computeRayOrigin(sample.sd.posW, dot(sample.sd.faceN, v.sd.posW - sample.sd.posW) >= 0 ? sample.sd.faceN : -sample.sd.faceN);

        float3 toLight = lightSideOrigin - cameraSideOrigin;
        float distSqr = max(dot(toLight, toLight), kMinLightDistSqr);
        float distance = sqrt(distSqr);
        float3 dir = toLight / distance;

        Ray ray = Ray(cameraSideOrigin, dir, 0, distance);
        return ray;
    }

    float BDPTMIS(uint2 pixel, int offset, uint sampleIndex) {
        const bool isPrimaryHit = offset == 0;

        let lod = ExplicitLodTextureSampler(0);

        int t = offset + 2;
        int s = int(sampleIndex % kMaxSurfaceBounces) + 1;
        float sumRi = 0;
        float ri = 1;

        Vertex pt = SrcCameraPathsVertexsReservoirBuffer[cameraPathOffset(pixel,offset)].v.unpack(isPrimaryHit);
        Vertex qs = LightPathsVertexsBuffer[sampleIndex].unpack(false);
        Vertex qsMinus = LightPathsVertexsBuffer[sampleIndex - 1].unpack(false);
        Vertex ptMinus;
        if (t - 2 > 0) {
            ptMinus = SrcCameraPathsVertexsReservoirBuffer[cameraPathOffset(pixel, offset - 1)].v.unpack(offset - 1 == 1);
        }
        else {
            ptMinus = Vertex::createCamera(gScene.camera.getPosition(), gScene.camera.data.cameraW);
        }
        for (int i = t - 2; i >= 0; --i) {
            VertexInfo vi = SrcCameraPathsVertexsReservoirBuffer[cameraPathOffset(pixel, i)].v;
            float pdfRev = vi.pdfRev;
            bool delta = vi.isDelta();
            bool deltaMinus = (i == 0) ? false : SrcCameraPathsVertexsReservoirBuffer[cameraPathOffset(pixel, i)].v.isDelta();
            if (i == t - 2) {
                float3 wo = normalize(pt.sd.posW - qs.sd.posW);
                const IBSDF bsdf = gScene.materials.getBSDF(qs.sd, lod);
                // pdfRev = bsdf.evalPdf(qs.sd, wo);
                pdfRev = qs.ConvertDensity(bsdf.evalPdf(qs.sd, wo), pt);
                delta = false;
            }
            else if (i == t - 3) {
                ShadingData sdRev = pt.sd;
                sdRev.V = normalize(pt.sd.posW - qs.sd.posW);
                float3 wo = normalize(ptMinus.sd.posW - pt.sd.posW);
                const IBSDF bsdf = gScene.materials.getBSDF(sdRev, lod);
                pdfRev = pt.ConvertDensity(bsdf.evalPdf(sdRev, wo), ptMinus);
            }
            ri *= remap0(pdfRev) / remap0(vi.pdfFwd);
            if (!delta && !deltaMinus) {
                sumRi += ri;
            }
        }

        ri = 1;
        for (int i = s - 1; i >= 0; --i) {
            VertexInfo vi = LightPathsVertexsBuffer[sampleIndex - (s - 1) + i];
            float pdfRev = vi.pdfRev;
            bool delta = vi.isDelta();
            bool deltaMinus = (i == 0) ? false : LightPathsVertexsBuffer[sampleIndex - (s - 1) + (i - 1)].isDelta();
            if (i == s - 1) {
                float3 wo = normalize(qs.sd.posW - pt.sd.posW);
                const IBSDF bsdf = gScene.materials.getBSDF(pt.sd, lod);
                pdfRev = pt.ConvertDensity(bsdf.evalPdf(pt.sd, wo), qs);
                delta = false;
            }
            if (i == s - 2) {
                ShadingData sdRev = qs.sd;
                sdRev.V = normalize(qs.sd.posW - pt.sd.posW);
                float3 wo = normalize(qsMinus.sd.posW - qs.sd.posW);
                const IBSDF bsdf = gScene.materials.getBSDF(sdRev, lod);
                pdfRev = qs.ConvertDensity(bsdf.evalPdf(sdRev, wo), qsMinus);
            }
            ri *= remap0(pdfRev) / remap0(vi.pdfFwd);
            if (!delta && !deltaMinus) {
                sumRi += ri;
            }
        }

        return 1.f / (1 + sumRi);
    }

    inline float remap0(float f) {
        return f != 0.0 ? f : 1.0;
    }

    float3 SpatialReuse(const uint2 pixel, uint offset, out bool isValid) {
        SampleGenerator sg = SampleGenerator(pixel, params.seed + offset);

        CameraVertex cv = SrcCameraPathsVertexsReservoirBuffer[cameraPathOffset(pixel, int(offset))];
        isValid = cv.v.isValid();
        if (!isValid) return 0;
        bool visible = cv.v.isVisible();
        Vertex v = cv.v.unpack(offset == 0);
        Reservoir r = cv.r;
        if (!visible) r.weightSum = 0;

        int2 ld = max(int2(pixel) - screenSpaceReuseDomain, 0);
        int2 rt = min(int2(pixel) + screenSpaceReuseDomain, int2(params.frameDim) - 1);
        int2 dom = rt - ld;
        float total = float((dom.x + 1) * (dom.y + 1));

        Vertex resultSample = LightPathsVertexsBuffer[r.sampleIndex].unpack(false);
        float3 result_q = RISWeight(v, resultSample, sg, false);
        float result_weight = result_q.x + result_q.y + result_q.z;
        bool isChange = false;
        uint sum = 0;

        for (int i = ld.x; i <= rt.x; ++i) {
            for (int j = ld.y; j <= rt.y; ++j) {
                CameraVertex neighbor = SrcCameraPathsVertexsReservoirBuffer[cameraPathOffset(uint2(i, j), int(offset))];
                if (!neighbor.v.isValid()) continue;
                
                Reservoir dstR = neighbor.r;
                Vertex lightSample = LightPathsVertexsBuffer[dstR.sampleIndex].unpack(false);
                //Vertex neighborVertex = neighbor.v.unpack(offset == 0);
                float3 q = RISWeight(v, lightSample, sg, false);
                float weight = q.x + q.y + q.z;
                float weightSum = weight * dstR.W * dstR.M;
                if (r.combine(weightSum, dstR, sampleNext1D(sg))) {
                    sum++;
                    resultSample = lightSample;
                    result_q = q;
                    result_weight = weight;
                    isChange = true;
                }
            }
        }
        if (r.weightSum == 0.f) {
            cv.r = r;
            cv.v.setVisible(false);
            DstCameraPathsVertexsReservoirBuffer[cameraPathOffset(pixel, int(offset))] = cv;
            return 0;
        }
        //float weight_s = result_q.x + result_q.y + result_q.z;
        float3 f = RISWeight(v, resultSample, sg);
        // float weight_s = f.x + f.y + f.z;
        r.W = (r.weightSum / r.M) / result_weight;

        float inv_M = 1.0 / float(kLightPassHeight * kLightPassWidth);
        float3 Lr = inv_M * v.beta * f * r.W;
        if (any(Lr > 0) && isChange) {
            Ray ray = getVisibiliyTestRay(v, resultSample);
            SceneRayQuery<false> sceneRayQuery;
            visible = sceneRayQuery.traceVisibilityRay(ray, RAY_FLAG_NONE, 0xff);
        }

        Lr *= (visible) ? BDPTMIS(pixel, int(offset), r.sampleIndex) : 0;
        cv.r = r;
        cv.v.setVisible(visible);
        DstCameraPathsVertexsReservoirBuffer[cameraPathOffset(pixel, int(offset))] = cv;

        return Lr;
    }

    void execute(const uint2 pixel) {
        float3 Lr = 0;
        bool isValid = true;
        /*
        if(params.hasFlag(BDPTFlags::s2) && params.hasFlag(BDPTFlags::spatialReuse))
        {
            for (uint i = 0; (i < kMaxSurfaceBounces) && isValid; ++i) {
                Lr += SpatialReuse(pixel, i, isValid);
            }
        }
        */
        outputColor[pixel] = output[pixel];// + (float4(Lr, 0));
    }
};

cbuffer CB
{
    SpatiotemproalReuse gSpatiotemproalReuse;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gSpatiotemproalReuse.execute(dispatchThreadId.xy);
}
