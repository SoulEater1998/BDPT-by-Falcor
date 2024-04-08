
#include "Scene/SceneDefines.slangh"

import Scene.Raytracing;
import Scene.RaytracingInline; // For visibility queries.
import Scene.Intersection;
import Rendering.Utils.PixelStats;
import RenderPasses.Shared.Denoising.NRDConstants;
import Utils.Math.PackedFormats;
import Utils.Math.FormatConversion;
import PathTracer;

ParameterBlock<PathTracer> gPathTracer;
RaytracingAccelerationStructure gHitPointAS;
Scheduler gScheduler;

// Define ray indices.
static const uint kRayTypeScatter = 0;
static const uint kMissScatter = 0;

struct PathPayload
{
    uint4 packed[5];

    PackedHitInfo hit;
    InteriorList interiorList;  ///< Interior list. Keeping track of a stack of materials with medium properties. The size depends on the config (4B/item).
    SampleGenerator sg;         ///< Sample generator state. Typically 4-16B.

    static PathPayload pack(const PathState path)
    {
        PathPayload p = {};

        p.packed[0].xyz = asuint(path.origin);
        p.packed[0].w = path.id;

        p.packed[1].xyz = asuint(path.dir);
        p.packed[1].w = path.flagsAndVertexIndex;

        p.packed[2].xyz = asuint(path.normal);
        p.packed[2].w = uint(path.rejectedHits) | ((f32tof16(path.sceneLength) & 0xffff) << 16);

        p.packed[3].xyz = asuint(path.thp);
        p.packed[3].w = path.bounceCounters;

        p.packed[4].xyz = asuint(path.L);
        p.packed[4].w = asuint(path.pdf);

        p.hit = path.hit.getData();
        p.interiorList = path.interiorList;
        p.sg = path.sg;

        return p;
    }

    static PathState unpack(const PathPayload p)
    {
        PathState path = {};

        path.origin = asfloat(p.packed[0].xyz);
        path.id = p.packed[0].w;

        path.dir = asfloat(p.packed[1].xyz);
        path.flagsAndVertexIndex = p.packed[1].w;

        path.normal = asfloat(p.packed[2].xyz);
        path.rejectedHits = uint16_t(p.packed[2].w & 0xffff);
        path.sceneLength = float16_t(f16tof32(p.packed[2].w >> 16));

        path.thp = asfloat(p.packed[3].xyz);
        path.bounceCounters = p.packed[3].w;

        path.L = asfloat(p.packed[4].xyz);
        path.pdf = asfloat(p.packed[4].w);

        path.hit = HitInfo(p.hit);
        path.interiorList = p.interiorList;
        path.sg = p.sg;

        return path;
    }
};

struct RISPayload
{
    uint4 packed; // x := sample weight(float), y := total weight(float), z := sample count, w := index
    SampleGenerator sg;
}

struct SphereAttribs
{
    float2 pad;
};

struct VisibilityQuery : PathTracer::IVisibilityQuery
{
    bool traceVisibilityRay(const Ray ray)
    {
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        return sceneRayQuery.traceVisibilityRay(ray, RAY_FLAG_NONE, 0xff);
    }
};

/** Simple scheduler using a fullscreen raytracing dispatch.
*/
struct Scheduler
{
    bool handleCollectIntersection(const float3 origin, const uint primIndex) {
        return false;//gPathTracer.handleCollectIntersection(origin, primIndex);
    }

    void handleCollectAnyHit(inout RISPayload payload, const float3 origin, const float3 normal, const uint primIndex) {
        // uint4 reservoir = payload.packed;
        float rnd = sampleNext1D(payload.sg);
        //gPathTracer.handleCollectAnyHit(payload.packed, origin, normal, primIndex, rnd);
        // payload.packed = reservoir;
    }
    
    /** Handle a miss.
        \param[in,out] payload Payload data.
    */

    void handleMiss(inout PathPayload payload)
    {
        PathState path = PathPayload::unpack(payload);
        path.clearHit();
        path.sceneLength = float16_t(kNRDInvalidPathLength);

        gPathTracer.setupPathLogging(path);
        gPathTracer.BDPTHandleMiss(path);

        payload = PathPayload::pack(path);
    }

    /** Handle a hit.
        \param[in,out] payload Payload data.
        \param[in] hit Hit info.
        \param[in] hitT Hit distance.
    */
    void handleHit(inout PathPayload payload, const HitInfo hit, const float hitT)
    {
        PathState path = PathPayload::unpack(payload);
        path.setHit(hit);
        path.sceneLength += float16_t(hitT);

        gPathTracer.setupPathLogging(path);

        //VisibilityQuery vq;
        //gPathTracer.lightPathHandleHit(path, vq);

        payload = PathPayload::pack(path);
    }

    /** Trace a ray to find the next hit.
        This replaces `PathTracer::nextHit` but without support for volume sampling.
        \param[in,out] path Path state.
    */
    void nextHit(inout PathState path, inout ExtraPayload exPayload)
    {
        // Advance to next path vertex.
        path.incrementVertexIndex();

        // Trace ray.
        logTraceRay(PixelStatsRayType::ClosestHit);
        const Ray ray = path.getScatterRay();
        PathPayload payload = PathPayload::pack(path);
        uint rayFlags = RAY_FLAG_NONE;
        if (!kUseAlphaTest) rayFlags |= RAY_FLAG_FORCE_OPAQUE;
        TraceRay(gScene.rtAccel, rayFlags, 0xff /* instanceInclusionMask */, kRayTypeScatter /* hitIdx */, rayTypeCount, kMissScatter /* missIdx */, ray.toRayDesc(), payload);

        path = PathPayload::unpack(payload);
        if (path.isActive()) {
            VisibilityQuery vq;
            gPathTracer.lightPathHandleHit(path, vq, exPayload);
        }
    }

    /** Traces a path and write result to output.
        \param[in] pathID Path ID.
    */
    void tracePath(uint pathID)
    {
        PathState path = {};
        ExtraPayload exPayload = {};
        //sample initial ray for light subpath
        gPathTracer.generateLightPath(pathID, path, exPayload);
        gPathTracer.setupPathLogging(path);

        // Trace path.
        
        while (path.isActive())
        {
            nextHit(path, exPayload);
        }
        
        //gPathTracer.pushInBuffer(path, dim.x);
    }

    /** Runs the path tracer for a single pixel.
        \param[in] pixel Pixel index.
    */
    void run(uint2 pixel)
    {
        // Determine number of samples to take.
        uint samplesRemaining = 1; // kLightSubpathPerThread;
        
        // Loop over samples.
        while (samplesRemaining > 0)
        {
            samplesRemaining -= 1;
            uint pathID = pixel.x | (pixel.y << 12) | (samplesRemaining << 24);
            tracePath(pathID);
        }
    }
}

//
// Shader entry point for miss shaders.
//

[shader("miss")]
void scatterMiss(inout PathPayload payload : SV_RayPayload)
{
    gScheduler.handleMiss(payload);
}

//
// Shader entry points for TriangleMesh hit groups.
//

[shader("anyhit")]
void scatterTriangleAnyHit(inout PathPayload payload : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
#if USE_ALPHA_TEST
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f)) IgnoreHit();
#endif
}

[shader("closesthit")]
void scatterTriangleClosestHit(inout PathPayload payload : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    TriangleHit triangleHit;
    triangleHit.instanceID = getGeometryInstanceID();
    triangleHit.primitiveIndex = PrimitiveIndex();
    triangleHit.barycentrics = attribs.barycentrics;
    gScheduler.handleHit(payload, HitInfo(triangleHit), RayTCurrent());
}

//
// Shader entry points for DisplacedTriangleMesh hit groups.
//

[shader("intersection")]
void displacedTriangleMeshIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    DisplacedTriangleMeshIntersector::Attribs attribs;
    float t;
    if (DisplacedTriangleMeshIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), attribs, t))
    {
        ReportHit(t, 0, attribs);
    }
}

[shader("closesthit")]
void scatterDisplacedTriangleMeshClosestHit(inout PathPayload payload, DisplacedTriangleMeshIntersector::Attribs attribs)
{
    DisplacedTriangleHit displacedTriangleHit;
    displacedTriangleHit.instanceID = getGeometryInstanceID();
    displacedTriangleHit.primitiveIndex = PrimitiveIndex();
    displacedTriangleHit.barycentrics = attribs.barycentrics;
    displacedTriangleHit.displacement = attribs.displacement;
    gScheduler.handleHit(payload, HitInfo(displacedTriangleHit), RayTCurrent());
}

//
// Shader entry points for Curve hit groups.
//

[shader("intersection")]
void curveIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    CurveIntersector::Attribs attribs;
    float t;
    if (CurveIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), attribs, t))
    {
        ReportHit(t, 0, attribs);
    }
}

[shader("closesthit")]
void scatterCurveClosestHit(inout PathPayload payload, CurveIntersector::Attribs attribs)
{
    CurveHit curveHit;
    curveHit.instanceID = getGeometryInstanceID();
    curveHit.primitiveIndex = PrimitiveIndex();
    curveHit.barycentrics = attribs.barycentrics;
    gScheduler.handleHit(payload, HitInfo(curveHit), RayTCurrent());
}


//
// Shader entry points for SDFGrid hit groups.
//

[shader("intersection")]
void sdfGridIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    SDFGridHitData sdfGridHitData;
    float t;
    if (SDFGridIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), sdfGridHitData, t))
    {
        ReportHit(t, 0, sdfGridHitData);
    }
}

[shader("closesthit")]
void scatterSdfGridClosestHit(inout PathPayload payload, SDFGridHitData sdfGridHitData)
{
    SDFGridHit sdfGridHit;
    sdfGridHit.instanceID = getGeometryInstanceID();
    sdfGridHit.hitData = sdfGridHitData;
    gScheduler.handleHit(payload, HitInfo(sdfGridHit), RayTCurrent());
}

//
// Shader entry points for neighbor collect groups.
//
[shader("intersection")]
void RISIntersection()
{
    const float3 origin = ObjectRayOrigin();
    const uint primIndex = PrimitiveIndex();

    bool tHit = gScheduler.handleCollectIntersection(origin, primIndex);
    SphereAttribs attribs;
    attribs.pad = float2(1);
    tHit = true;
    if (tHit) {
        ReportHit(RayTCurrent(), 0, attribs);
    }
}

[shader("anyhit")]
void RISAnyHit(inout RISPayload payload: SV_RayPayload, SphereAttribs pad: SV_IntersectionAttributes)
{
    const float3 origin = ObjectRayOrigin();
    const float3 normal = ObjectRayDirection();
    const uint primIndex = PrimitiveIndex();
    payload.packed.z = 1;
    // payload.packed.z += pad.pad.x > 0 ? 1 : 0;
    // gScheduler.handleCollectAnyHit(payload, origin, normal, primIndex);
}

[shader("miss")]
void RISMiss(inout RISPayload payload)
{
    // payload.packed.z = 10;
}

//
// Shader entry point for ray generation shader.
//

[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 frameDim = DispatchRaysDimensions().xy;
    if (all(pixel >= frameDim)) return;

    gScheduler.run(pixel);
}
