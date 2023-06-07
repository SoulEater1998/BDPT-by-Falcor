
ByteAddressBuffer keyIndexList : register(u0);
RWTexture2D<float4> gOutput;
//StructuredBuffer<float4> vplBound;
cbuffer CSConstants
{
    uint totalNum;
};

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    uint offset = pixel.y * 1920 + pixel.x;
    if (totalNum == 0) {
        gOutput[pixel] = float4(0, 1, 0, 0);
    }
    else if (offset >= totalNum) {
        gOutput[pixel] = float4(1, 0, 0, 0);
    }
    else {
        uint2 KeyIndexPair = keyIndexList.Load2(8 * offset);
        //gOutput[pixel] = float4(0, 1, 0, 0);
        gOutput[pixel] = float4(float(KeyIndexPair.y)/0x4fffffff);
    }
}
