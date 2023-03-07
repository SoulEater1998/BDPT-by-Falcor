
__exported import LoadShadingData;
__exported import Utils.Sampling.SampleGenerator;
__exported import Scene.HitInfo;
import PathState;

enum class VertexType
{
    Camera = 0,
    Light,
    SurfaceOrMedium,
};

struct Vertex
{
    ShadingData sd;
    float3 beta = float3(1, 1, 1);
    bool valid = false;

    float pdfFwd = 0.f;
    float pdfRev = 0.f;
    bool delta = false;
    VertexType type = VertexType::SurfaceOrMedium;

    static Vertex init() {
        Vertex v = {};
        v.sd = {};
        v.valid = true;
        return v;
    }

    static Vertex create(ShadingData sd, float3 beta, float pdfFwd, float pdfRev, VertexType type) {
        Vertex v = {};
        v.beta = beta;
        v.sd = sd;
        v.valid = true;
        v.pdfFwd = pdfFwd;
        v.pdfRev = pdfRev;
        v.type = type;
        return v;
    };

    float ConvertDensity(float pdf, const Vertex next) {
        float3 w = next.sd.posW - sd.posW;
        if (dot(w, w) == 0) return 0.f;
        float invDist2 = 1 / dot(w, w);
        if(next.type == VertexType::SurfaceOrMedium) pdf *= abs(dot(next.sd.N, w * sqrt(invDist2)));
        return pdf * invDist2;
    };

    static Vertex createCamera(float3 posW, float3 N) {
        Vertex cv = {};
        cv.sd.posW = posW;
        cv.sd.N = N;
        cv.beta = 1.f;
        cv.pdfFwd = 1.f;
        return cv;
    };

    static Vertex createLight(float3 posW, float3 beta, float pdfFwd) {
        Vertex lv = {};
        lv.sd.posW = posW;
        lv.beta = beta;
        lv.pdfFwd = pdfFwd;
        return lv;
    };
};

struct Reservoir
{
    uint sampleIndex = 0;
    float weightSum = 0;
    uint M = 0; // The number of samples seen so far
    float W = 0; // The total weight

    [mutating] void update(uint dstSample, float weight, float rnd) {
        weightSum += weight;
        M += 1;
        sampleIndex = (rnd < weight / weightSum) ? dstSample : sampleIndex;
    }

};

struct CameraVertex
{
    VertexInfo v;
    Reservoir r;
};

enum class VertexFlag {
    typeMask = 0x0003,
    valid = 0x0004,
    delta = 0x0008,
};

struct VertexInfo
{
    HitInfo hit;
    float3 origin;
    float pdfFwd = 0;
    float3 dir;
    float pdfRev = 0;
    float3 beta;
    uint flagAndType = 0;

    static VertexInfo create(PathState path, Vertex v) {
        VertexInfo vinfo = {};
        vinfo.hit = path.hit;
        vinfo.origin = path.origin;
        vinfo.dir = path.dir;
        vinfo.pdfFwd = v.pdfFwd;
        vinfo.pdfRev = v.pdfRev;
        vinfo.beta = v.beta;

        uint flagAndType = uint(v.type);
        flagAndType |= v.valid ? uint(VertexFlag::valid) : 0;
        flagAndType |= v.delta ? uint(VertexFlag::delta) : 0;
        vinfo.flagAndType = flagAndType;

        return vinfo;
    }

    Vertex unpack(bool isPrimaryHit) {
        const HitType hitType = hit.getType();
        const bool isTriangleHit = hitType == HitType::Triangle;

        let lod = ExplicitLodTextureSampler(0);
        ShadingData sd = loadShadingData(hit, origin, dir, isPrimaryHit, lod);

        Vertex result = Vertex::create(sd, beta, pdfFwd, pdfRev, VertexType(flagAndType & uint(VertexFlag::typeMask)));
        result.delta = (flagAndType & uint(VertexFlag::delta)) != 0;
        result.valid = (flagAndType & uint(VertexFlag::valid)) != 0;

        return result;
    }

    bool isDelta() {
        return (flagAndType & uint(VertexFlag::delta)) != 0;
    }
};