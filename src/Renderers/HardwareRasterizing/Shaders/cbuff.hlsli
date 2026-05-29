#pragma once
cbuffer CBuf
{
    matrix view, projection, viewProjection;
    float4 time;
    float4 camPos, lightDir;
}