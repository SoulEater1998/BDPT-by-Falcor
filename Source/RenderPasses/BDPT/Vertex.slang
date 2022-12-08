import LoadShadingData;

enum class VertexType
{
    Camera = 0,
    Light,
    Surface,
    Medium,
};

struct Vertex
{
    ShadingData sd;
    float3 beta;
    
    float pdfFwd = 0;
    float pdfRev = 0;
    bool delta = false;
    VertexType tpye;
};

