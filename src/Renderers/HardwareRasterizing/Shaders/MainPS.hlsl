#include "cbuff.hlsli"

Texture2D tex;
SamplerState sam;
float4 main(float3 worldPos : WorldPos, float2 uv : TEXCOORD0, float3 normals : TEXCOORD1) : SV_Target
{
    float4 s = tex.Sample(sam, uv);
    clip(s.a - 0.1);
    return s;
    //return saturate(length(worldPos - camPos) / 1500);
}