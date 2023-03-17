
// Copyright (c) 2022, Fengqi Liu <M202173624@hust.edu.cn>
// All rights reserved.
// This code is licensed under the MIT License (MIT).

StructuredBuffer<float4> Positions : register(t0);
StructuredBuffer<float4> Bound;
RWByteAddressBuffer keyIndexList : register(u0);
#define DEFAULT_BLOCK_SIZE 512
cbuffer CSConstants
{
    uint num;
    uint quantLevels;
};

inline uint3 BitExpansion(uint3 x)
{
	//https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
    x = (x | x << 16) & 0x30000ff;
    x = (x | x << 8) & 0x300f00f;
    x = (x | x << 4) & 0x30c30c3;
    x = (x | x << 2) & 0x9249249;
    return x;
}

[numthreads(DEFAULT_BLOCK_SIZE, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    float3 corner = Bound[0].xyz;
    float3 dimension = Bound[1].xyz;
    if (DTid.x < num)
    {
		//normalize position to [0,1]
        float3 normPos = (Positions[DTid.x].xyz - corner) / dimension;
        uint3 quantPos = min(max(0, uint3(normPos * quantLevels)), quantLevels - 1);
        quantPos = BitExpansion(quantPos);
        uint mortonCode = quantPos.x * 4 + quantPos.y * 2 + quantPos.z;

        uint2 KeyIndexPair = uint2(DTid.x, mortonCode);
        keyIndexList.Store2(8 * DTid.x, KeyIndexPair);
    }
}
