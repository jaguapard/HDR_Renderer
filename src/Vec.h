#pragma once
#include <bob/Vec2.h>
#include <bob/SSE_Vec4.h>
#undef min
#undef max
#include <avxxy/avxxy.h>

//typedef bob::_Vec2<double, double> Vec2d;
typedef bob::_SSE_Vec4_float Vec4f;
using namespace bob;
using namespace AVXXY_NAMESPACE;

typedef f32x16 float32x16;
typedef f32x8 float32x8;
typedef i32x8 int32x8;
typedef i32x16 int32x16;

typedef SIMD_VectorPack<f32x16, 4> Vec4_f32x16;
typedef SIMD_VectorPack<f32x16, 3> Vec3_f32x16;
typedef SIMD_VectorPack<f32x8, 4> Vec4_f32x8;
typedef SIMD_VectorPack<f32x8, 3> Vec3_f32x8;