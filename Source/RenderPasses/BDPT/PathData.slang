
import LoadShadingData;
import Utils.Sampling.SampleGenerator;
import Scene.HitInfo;
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

    static Vertex init(bool valid = true) {
        Vertex v = {};
        v.sd = {};
        v.valid = valid;
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
        /*if(next.type == VertexType::SurfaceOrMedium)*/ pdf *= abs(dot(next.sd.N, w * sqrt(invDist2)));
        return pdf * invDist2;
    };

    float ConvertDensity(float pdf, const float3 nextPos, const float3 nextNormal) {
        float3 w = nextPos - sd.posW;
        if (dot(w, w) == 0) return 0.f;
        float invDist2 = 1 / dot(w, w);
        pdf *= abs(dot(nextNormal, w * sqrt(invDist2)));
        return pdf * invDist2;
    }

    float ConvertDensity(PathState path) {
        float3 w = sd.posW - path.origin;
        if (dot(w, w) == 0) return 0.f;
        float invDist2 = 1 / dot(w, w);
        float pdf = path.pdf;
        pdf *= abs(dot(sd.N, w * sqrt(invDist2)));
        return pdf * invDist2;
    }

    float ConvertRevDensity(PathState path, float pdfRev) {
        float3 w = path.origin - sd.posW;
        if (dot(w, w) == 0) return 0.f;
        float invDist2 = 1 / dot(w, w);
        pdfRev *= abs(dot(path.normal, w * sqrt(invDist2)));
        return pdfRev * invDist2;
    }

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

    static Reservoir init() {
        Reservoir r = {};
        r.sampleIndex = 0;
        r.weightSum = 0;
        r.M = 0; // The number of samples seen so far
        r.W = 0;
        return r;
    }

    [mutating] bool update(uint dstSample, float weight, float rnd) {
        M += 1;
        weightSum += weight;
        if (weightSum == 0) return false;
        if (rnd < (weight / weightSum)) {
            sampleIndex = dstSample;
            return true;
        }
        else {
            return false;
        }
    }

    [mutating] bool combine(float dstWeight, Reservoir dstR, float rnd) {
        weightSum += dstWeight;
        M += dstR.M;
        if (weightSum == 0) return false;
        if (rnd < (dstWeight / weightSum)) {
            sampleIndex = dstR.sampleIndex;
            return true;
        }
        else {
            return false;
        }
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
    visible = 0x0010,
    diffuseHit = 0x0020
};

struct VertexInfo
{
    uint4 hitData;
    float3 origin;
    float pdfFwd = 0;
    float3 dir;
    float pdfRev = 0;
    float3 beta;
    uint flagAndType = 0;
    float3 originN;
    float de = 0;
    float pe = 0;
    uint prevIndex;

    static VertexInfo create(PathState path, Vertex v) {
        VertexInfo vinfo = {};
        vinfo.hitData = path.hit.getData();
        vinfo.origin = path.origin;
        vinfo.originN = path.normal;
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

    static VertexInfo createLight(Vertex v) {
        VertexInfo vinfo = {};
        vinfo.origin = v.sd.posW;   // Light pos
        vinfo.dir = v.sd.N;         // Light normal
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
        VertexType type = getType();

        if (type == VertexType::Light) {
            Vertex v = Vertex::init((flagAndType & uint(VertexFlag::valid)) != 0);
            v.sd.posW = origin;
            v.sd.N = dir;
            v.pdfFwd = pdfFwd;
            v.pdfRev = pdfRev;
            v.beta = beta;
            v.delta = (flagAndType & uint(VertexFlag::delta)) != 0;
            v.type = type;
            return v;
        }

        HitInfo hit = HitInfo(hitData);
        const HitType hitType = hit.getType();
        const bool isTriangleHit = hitType == HitType::Triangle;

        let lod = ExplicitLodTextureSampler(0);
        ShadingData sd = loadShadingData(hit, origin, dir, isPrimaryHit, lod);

        Vertex result = Vertex::create(sd, beta, pdfFwd, pdfRev, type);
        result.delta = (flagAndType & uint(VertexFlag::delta)) != 0;
        result.valid = (flagAndType & uint(VertexFlag::valid)) != 0;

        return result;
    }

    bool isDelta() {
        return (flagAndType & uint(VertexFlag::delta)) != 0;
    }

    bool isValid() {
        return (flagAndType & uint(VertexFlag::valid)) != 0;
    }

    bool isCausticHit() {
        return (flagAndType & uint(VertexFlag::diffuseHit)) != 0;
    }

    bool isLight() {
        uint type = flagAndType & uint(VertexFlag::typeMask);
        return type == uint(VertexType::Light);
    }

    bool isVisible() {
        return (flagAndType & uint(VertexFlag::visible)) != 0;
    }

    VertexType getType() {
        return VertexType(flagAndType & uint(VertexFlag::typeMask));
    }

    [mutating] void setFlag(bool tOrf, VertexFlag flag) {
        if (tOrf) {
            flagAndType |= uint(flag);
        }
        else {
            flagAndType &= ~uint(flag);
        }
    }

    [mutating] void setDelta(bool delta) {
        if (delta) {
            flagAndType |= uint(VertexFlag::delta);
        }
        else {
            flagAndType &= ~uint(VertexFlag::delta);
        }
    }

    [mutating] void setVisible(bool visible) {
        if (visible) {
            flagAndType |= uint(VertexFlag::visible);
        }
        else {
            flagAndType &= ~uint(VertexFlag::visible);
        }
    }
};

inline float getIntensity(float3 beta)
{
    return beta.x + beta.y + beta.z;
}

inline float MaxDistAlong(float3 p, float3 dir, float3 boundMin, float3 boundMax)
{
    float3 dir_p = dir * p;
    float3 mx0 = dir * boundMin - dir_p;
    float3 mx1 = dir * boundMax - dir_p;
    return max(mx0[0], mx1[0]) + max(mx0[1], mx1[1]) + max(mx0[2], mx1[2]);
}

inline void CoordinateSystem_(float3 v1, out float3 v2, out float3 v3)
{
    if (abs(v1.x) > abs(v1.y))
        v2 = float3(-v1.z, 0, v1.x) / sqrt(v1.x * v1.x + v1.z * v1.z);
    else
        v2 = float3(0, v1.z, -v1.y) / sqrt(v1.y * v1.y + v1.z * v1.z);
    v3 = normalize(cross(v1, v2));
}

inline float AbsMinDistAlong(float3 p, float3 dir, float3 boundMin, float3 boundMax)
{
    bool hasPositive = false;
    bool hasNegative = false;
    float a = dot(dir, float3(boundMin.x, boundMin.y, boundMin.z) - p);
    float b = dot(dir, float3(boundMin.x, boundMin.y, boundMax.z) - p);
    float c = dot(dir, float3(boundMin.x, boundMax.y, boundMin.z) - p);
    float d = dot(dir, float3(boundMin.x, boundMax.y, boundMax.z) - p);
    float e = dot(dir, float3(boundMax.x, boundMin.y, boundMin.z) - p);
    float f = dot(dir, float3(boundMax.x, boundMin.y, boundMax.z) - p);
    float g = dot(dir, float3(boundMax.x, boundMax.y, boundMin.z) - p);
    float h = dot(dir, float3(boundMax.x, boundMax.y, boundMax.z) - p);
    hasPositive = a > 0 || b > 0 || c > 0 || d > 0 || e > 0 || f > 0 || g > 0 || h > 0;
    hasNegative = a < 0 || b < 0 || c < 0 || d < 0 || e < 0 || f < 0 || g < 0 || h < 0;
    if (hasPositive && hasNegative)
        return 0.f;
    else
        return min(min(min(abs(a), abs(b)), min(abs(c), abs(d))), min(min(abs(e), abs(f)), min(abs(g), abs(h))));
}

inline float AbsMinDistAlong(float3 p, float3 dir, float3 boundMin, float3 boundMax, out bool hasPositive)
{
    hasPositive = false;
    bool hasNegative = false;
    float a = dot(dir, float3(boundMin.x, boundMin.y, boundMin.z) - p);
    float b = dot(dir, float3(boundMin.x, boundMin.y, boundMax.z) - p);
    float c = dot(dir, float3(boundMin.x, boundMax.y, boundMin.z) - p);
    float d = dot(dir, float3(boundMin.x, boundMax.y, boundMax.z) - p);
    float e = dot(dir, float3(boundMax.x, boundMin.y, boundMin.z) - p);
    float f = dot(dir, float3(boundMax.x, boundMin.y, boundMax.z) - p);
    float g = dot(dir, float3(boundMax.x, boundMax.y, boundMin.z) - p);
    float h = dot(dir, float3(boundMax.x, boundMax.y, boundMax.z) - p);
    hasPositive = a > 0 || b > 0 || c > 0 || d > 0 || e > 0 || f > 0 || g > 0 || h > 0;
    hasNegative = a < 0 || b < 0 || c < 0 || d < 0 || e < 0 || f < 0 || g < 0 || h < 0;
    if (hasPositive && hasNegative) {
        return 0.f;
    }
    else {
        return min(min(min(abs(a), abs(b)), min(abs(c), abs(d))), min(min(abs(e), abs(f)), min(abs(g), abs(h))));
    }     
}

inline float GeomTermBound(float3 p, float3 N, float3 boundMin, float3 boundMax)
{
    float nrm_max = MaxDistAlong(p, N, boundMin, boundMax);
    if (nrm_max <= 0)
        return 0.0f;
    float3 T, B;
    CoordinateSystem_(N, T, B);
    float y_amin = AbsMinDistAlong(p, T, boundMin, boundMax);
    float z_amin = AbsMinDistAlong(p, B, boundMin, boundMax);
    float hyp2 = y_amin * y_amin + z_amin * z_amin + nrm_max * nrm_max;
    return nrm_max * rsqrt(hyp2);
}

inline float SquaredDistanceToClosestPoint(float3 p, float3 boundMin, float3 boundMax)
{
    float3 d = min(max(p, boundMin), boundMax) - p;
    return dot(d, d);
}

inline float SquaredDistanceToFarthestPoint(float3 p, float3 boundMin, float3 boundMax)
{
    float3 d = max(abs(boundMin - p), abs(boundMax - p));
    return dot(d, d);
}

inline float normalizedWeights(float l2_0, float l2_1, float intensGeom0, float intensGeom1)
{
    float ww0 = l2_1 * intensGeom0;
    float ww1 = l2_0 * intensGeom1;
    return ww0 / (ww0 + ww1);
};

float InterlockedAddFloat(RWTexture2D<uint> buf, uint2 addr, float value)
{
    uint i_val = asuint(value);
    uint tmp0 = 0;
    uint tmp1;
    while (true)
    {
        InterlockedCompareExchange(buf[addr], tmp0, i_val, tmp1);
        if (tmp1 == tmp0) {
            return asfloat(tmp1);
        }
        tmp0 = tmp1;
        i_val = asuint(value + asfloat(tmp1));
    }
}

#define AVAILABLE 0xffffffff

void myLock(RWTexture2D<uint> signal, uint2 addr, uint threadSymbol)
{
    uint tmp1;
    while (true)
    {
        InterlockedCompareExchange(signal[addr], AVAILABLE, threadSymbol, tmp1);
        if (tmp1 == AVAILABLE) return;
    }
}

uint myUnlock(RWTexture2D<uint> signal, uint2 addr)
{
    uint tmp1;
    InterlockedExchange(signal[addr], AVAILABLE, tmp1);
    return tmp1;
}