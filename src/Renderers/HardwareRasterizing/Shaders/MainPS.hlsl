#include "cbuff.hlsli"
float4 main(float3 worldPos : WorldPos, float2 uv : TEXCOORD0, float3 normals : TEXCOORD1) : SV_Target
{
    return saturate(length(worldPos - camPos) / 1500);
}