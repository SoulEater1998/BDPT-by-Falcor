import LoadShadingData;
//import NRDHelpers;
import PathTracer;
import PathState;
import ColorType;
import BDPTParams;
import PathData;

struct SpatiotemproalResue
{
    PathTracerParams params;

    StructuredBuffer<Vertex> LightPathsVertexsBuffer; ///< Output the Vertex of all the light paths.
    StructuredBuffer<CameraVertex> SrcCameraPathsVertexsReservoirBuffer;
    RWStructuredBuffer<CameraVertex> DstCameraPathsVertexsReservoirBuffer;

    const int screenSpaceReuseDomain = 3;

    uint cameraPathOffset(uint2 pixel, int offset = 0) {
        uint base = (pixel.y * params.frameDim.x + pixel.x) * kMaxSurfaceBounces;
        return base + offset;
    }

    void execute(const uint2 pixel, uint offset) {
        CameraVertex cv = SrcCameraPathsVertexsReservoirBuffer[cameraPathOffset(pixel, int(offset))];
        int2 ld = max(int2(pixel) - screenSpaceReuseDomain, 0);
        int2 rt = min(int2(pixel) + screenSpaceReuseDomain, int2(params.frameDim) - 1);
        for (int i = ld.x; i <= rt.x; ++i) {
            for (int j = ld.y; j <= rt.y; ++j) {
                Reservoir r = SrcCameraPathsVertexsReservoirBuffer[cameraPathOffset(uint2(i, j), int(offset))].r;
                Vertex
            }
        }
    }
}