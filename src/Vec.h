#pragma once
#include <bob/Vec2.h>
#include <bob/SSE_Vec4.h>
#include <bob/VectorPack.h>
#include <AVXxy/AVXxy.h>

//typedef bob::_Vec2<double, double> Vec2d;
typedef bob::_SSE_Vec4_float Vec4f;
using namespace bob;
using namespace AVXXY_NAMESPACE;

typedef f32x16 float32x16;
typedef f32x8 float32x8;
typedef i32x8 int32x8;
typedef i32x16 int32x16;
typedef SIMD_Mask<16> Mask16;
typedef SIMD_Mask<8> Mask8;