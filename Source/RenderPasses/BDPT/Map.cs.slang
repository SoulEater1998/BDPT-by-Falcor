//import PathTracer;
//import PathData;
StructuredBuffer<uint2> srcBuffer;
RWStructuredBuffer<uint2> dstBuffer;

#define DEFAULT_BLOCK_SIZE 1024
cbuffer CSConstants
{
    uint num;
    uint reduction;
};
/*
float Random1DTo1D(float value, float a, float b) {
    // make value more random by making it bigger
    float random = frac(sin(value + b) * a);
    return random;
}
*/
static const int base = 2;

float IntegerRadicalInverse(int Base, int i)
{
    int numPoints, inverse;
    numPoints = 1;
    // 此循环将i在"Base"进制下的数字左右Flip
    for (inverse = 0; i > 0; i /= Base)
    {
        inverse = inverse * Base + (i % Base);
        numPoints = numPoints * Base;
    }

    // 除以Digit将这个数镜像到小数点右边
    return inverse / (float)numPoints;
}

[numthreads(DEFAULT_BLOCK_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    
    
    // float3 corner = Bound[0].xyz;
    // float3 dimension = Bound[1].xyz;

    //uint i = min(uint(num * IntegerRadicalInverse(base, int(DTid.x))), num - 1);
    //keyIndexList.Store2(8 * DTid.x, srcBuffer[i]);
    uint i = DTid.x;
    //for()
    
    if (i < num)
    {
        keyIndexList.Store2(8 * DTid.x, srcBuffer[i]);
    }
    
}
