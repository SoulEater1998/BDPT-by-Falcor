
__exported import LoadShadingData;
__exported import Utils.Sampling.SampleGenerator;
__exported import Scene.HitInfo;

enum class VertexType
{
    Camera = 0,
    Light,
    SurfaceOrMedium,
};

struct Vertex
{
    ShadingData sd;
    float3 beta;
    bool valid = false;

    float pdfFwd = 0;
    float pdfRev = 0;
    bool delta = false;
    VertexType type;

    static Vertex init() {
        Vertex v;
        v.valid = true;
        return v;
    }

    static Vertex create(ShadingData sd, float3 beta, float pdfFwd, float pdfRev, VertexType type) {
        Vertex v;
        v.beta = beta;
        v.sd = sd;
        v.valid = true;
        v.pdfFwd = pdfFwd;
        v.pdfRev = pdfRev;
        v.type = type;
    }

    float ConvertDensity(float pdf, const Vertex next) {
        float3 w = next.sd.posW - sd.posW;
        if (dot(w, w) == 0) return 0;
        float invDist2 = 1 / dot(w, w);
        pdf *= abs(dot(next.sd.N, w * sqrt(invDist2)));
        return pdf * invDist2;
    }
}

struct BDPathVertex
{
    float3 color;
    bool valid;
    float3 posW;
    float3 N;
    float3 V;

    float3 dif;
    float3 spec;
    float rough;
    bool isSpecular;

    float pdfForward;

    [mutating] void update(Vertex lv) {
        color = lv.beta;
        valid = true;
        posW = lv.sd.posW;
        N = lv.sd.N;
        pdfForward = lv.pdfFwd;
    }

    static BDPathVertex init()
    {
        BDPathVertex v;
        v.color = float3(0, 0, 0);
        v.valid = false;
        v.posW = float3(0, 0, 0);
        v.N = float3(0, 0, 0);
        v.V = float3(0, 0, 0);
        v.dif = float3(0, 0, 0);
        v.spec = float3(0, 0, 0);
        v.rough = 0.0f;
        v.isSpecular = false;
        v.pdfForward = 0.0f;
        return v;
    }

    static BDPathVertex create(float3 color, float3 posW, float3 N, float3 V, float3 dif, float3 spec, float rough, bool isSpecular, float pdfForward)
    {
        BDPathVertex v;
        v.color = color;
        v.valid = true;
        v.posW = posW;
        v.N = N;
        v.V = V;
        v.dif = dif;
        v.spec = spec;
        v.rough = rough;
        v.isSpecular = isSpecular;
        v.pdfForward = pdfForward;
        return v;
    }
};

// The payload structure for our indirect rays
struct RayPayload
{
    float3 color;
    SampleGenerator sg;

    float3 posW;
    float3 N;
    float3 V;
    float3 dif;
    float3 spec;
    float rough;
    bool isSpecular;

    float pdfForward;

    float3 rayOrigin;
    float3 rayDir;

    bool terminated;
};

RayPayload initPayload(float3 rayOrigin, float3 rayDir, float3 N,float3 color, SampleGenerator sg)
{
    RayPayload payload;
    payload.rayOrigin = rayOrigin;
    payload.rayDir = rayDir;
    payload.sg = sg;
    payload.color = color;
    payload.posW = rayOrigin;
    payload.N = N;
    payload.V = float3(0.0f);
    payload.dif = float3(0.0f);
    payload.spec = float3(0.0f);
    payload.rough = 0.0;
    payload.isSpecular = false;
    payload.pdfForward = 0.0;
    payload.terminated = false;
    return payload;
}

void updateRayData(inout RayPayload rayData, float3 color, float3 hit, float3 L, float3 N, float3 V, float3 dif, float3 spec, float rough, bool isSpecular, float pdfF)
{
    // update brdf
    rayData.color *= color;

    // update next ray segment
    rayData.rayOrigin = hit;
    rayData.rayDir = L;

    // update geometry data
    rayData.posW = hit;
    rayData.N = N;
    rayData.V = V;

    // update material data
    rayData.dif = dif;
    rayData.spec = spec;
    rayData.rough = rough;
    rayData.isSpecular = isSpecular;

    // update pdf
    rayData.pdfForward = pdfF;
}
