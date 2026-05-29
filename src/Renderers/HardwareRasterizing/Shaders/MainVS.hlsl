#include "cbuff.hlsli"

struct VSOut
{
    float3 originalWorldPos : WorldPos;
    float2 uv : TEXCOORD0;
    float3 normals : TEXCOORD1;
    float4 transformedPos : SV_Position;
};
VSOut main(float3 pos : Pos, float2 uv : UV, float3 normals : Normals)
{
    VSOut ret;
    ret.originalWorldPos = float3(pos.x, pos.y, pos.z);
    ret.transformedPos = mul(float4(ret.originalWorldPos, 1.f), viewProjection);
    ret.uv = uv;
    ret.normals = normals;
    return ret;
}