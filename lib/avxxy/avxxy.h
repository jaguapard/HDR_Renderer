#pragma once
#include "namespace.h"
#include "SIMD_Vector.h"
#include "Dispatcher.h" //TODO: remove this from releases?
#include "funcs.h"
#include "funcs_impl.h"
#include "operators.h"
#include "SIMD_BitMask_impl.h"

namespace AVXXY_NAMESPACE
{
	typedef SIMD_Vector<int8_t, 2> i8x2;
	typedef SIMD_Vector<int8_t, 4> i8x4;
	typedef SIMD_Vector<int8_t, 8> i8x8;
	typedef SIMD_Vector<int8_t, 16> i8x16;
	typedef SIMD_Vector<int8_t, 32> i8x32;
	typedef SIMD_Vector<int8_t, 64> i8x64;

	typedef SIMD_Vector<int16_t, 2> i16x2;
	typedef SIMD_Vector<int16_t, 4> i16x4;
	typedef SIMD_Vector<int16_t, 8> i16x8;
	typedef SIMD_Vector<int16_t, 16> i16x16;
	typedef SIMD_Vector<int16_t, 32> i16x32;
	typedef SIMD_Vector<int16_t, 64> i16x64;

	typedef SIMD_Vector<int32_t, 2> i32x2;
	typedef SIMD_Vector<int32_t, 4> i32x4;
	typedef SIMD_Vector<int32_t, 8> i32x8;
	typedef SIMD_Vector<int32_t, 16> i32x16;
	typedef SIMD_Vector<int32_t, 32> i32x32;
	typedef SIMD_Vector<int32_t, 64> i32x64;

	typedef SIMD_Vector<int64_t, 2> i64x2;
	typedef SIMD_Vector<int64_t, 4> i64x4;
	typedef SIMD_Vector<int64_t, 8> i64x8;
	typedef SIMD_Vector<int64_t, 16> i64x16;
	typedef SIMD_Vector<int64_t, 32> i64x32;
	typedef SIMD_Vector<int64_t, 64> i64x64;

	typedef SIMD_Vector<uint8_t, 2> u8x2;
	typedef SIMD_Vector<uint8_t, 4> u8x4;
	typedef SIMD_Vector<uint8_t, 8> u8x8;
	typedef SIMD_Vector<uint8_t, 16> u8x16;
	typedef SIMD_Vector<uint8_t, 32> u8x32;
	typedef SIMD_Vector<uint8_t, 64> u8x64;

	typedef SIMD_Vector<uint16_t, 2> u16x2;
	typedef SIMD_Vector<uint16_t, 4> u16x4;
	typedef SIMD_Vector<uint16_t, 8> u16x8;
	typedef SIMD_Vector<uint16_t, 16> u16x16;
	typedef SIMD_Vector<uint16_t, 32> u16x32;
	typedef SIMD_Vector<uint16_t, 64> u16x64;

	typedef SIMD_Vector<uint32_t, 2> u32x2;
	typedef SIMD_Vector<uint32_t, 4> u32x4;
	typedef SIMD_Vector<uint32_t, 8> u32x8;
	typedef SIMD_Vector<uint32_t, 16> u32x16;
	typedef SIMD_Vector<uint32_t, 32> u32x32;
	typedef SIMD_Vector<uint32_t, 64> u32x64;

	typedef SIMD_Vector<uint64_t, 2> u64x2;
	typedef SIMD_Vector<uint64_t, 4> u64x4;
	typedef SIMD_Vector<uint64_t, 8> u64x8;
	typedef SIMD_Vector<uint64_t, 16> u64x16;
	typedef SIMD_Vector<uint64_t, 32> u64x32;
	typedef SIMD_Vector<uint64_t, 64> u64x64;

	typedef SIMD_Vector<float, 2> f32x2;
	typedef SIMD_Vector<float, 4> f32x4;
	typedef SIMD_Vector<float, 8> f32x8;
	typedef SIMD_Vector<float, 16> f32x16;
	typedef SIMD_Vector<float, 32> f32x32;
	typedef SIMD_Vector<float, 64> f32x64;

	typedef SIMD_Vector<double, 2> f64x2;
	typedef SIMD_Vector<double, 4> f64x4;
	typedef SIMD_Vector<double, 8> f64x8;
	typedef SIMD_Vector<double, 16> f64x16;
	typedef SIMD_Vector<double, 32> f64x32;
	typedef SIMD_Vector<double, 64> f64x64;


	typedef SIMD_Vector<int8_t, 16> xmm_i8;
	typedef SIMD_Vector<int16_t, 8> xmm_i16;
	typedef SIMD_Vector<int32_t, 4> xmm_i32;
	typedef SIMD_Vector<int64_t, 2> xmm_i64;
	typedef SIMD_Vector<uint8_t, 16> xmm_u8;
	typedef SIMD_Vector<uint16_t, 8> xmm_u16;
	typedef SIMD_Vector<uint32_t, 4> xmm_u32;
	typedef SIMD_Vector<uint64_t, 2> xmm_u64;
	typedef SIMD_Vector<float, 4> xmm_f32;
	typedef SIMD_Vector<double, 2> xmm_f64;

	typedef SIMD_Vector<int8_t, 32> ymm_i8;
	typedef SIMD_Vector<int16_t, 16> ymm_i16;
	typedef SIMD_Vector<int32_t, 8> ymm_i32;
	typedef SIMD_Vector<int64_t, 4> ymm_i64;
	typedef SIMD_Vector<uint8_t, 32> ymm_u8;
	typedef SIMD_Vector<uint16_t, 16> ymm_u16;
	typedef SIMD_Vector<uint32_t, 8> ymm_u32;
	typedef SIMD_Vector<uint64_t, 4> ymm_u64;
	typedef SIMD_Vector<float, 8> ymm_f32;
	typedef SIMD_Vector<double, 4> ymm_f64;

	typedef SIMD_Vector<int8_t, 64> zmm_i8;
	typedef SIMD_Vector<int16_t, 32> zmm_i16;
	typedef SIMD_Vector<int32_t, 16> zmm_i32;
	typedef SIMD_Vector<int64_t, 8> zmm_i64;
	typedef SIMD_Vector<uint8_t, 64> zmm_u8;
	typedef SIMD_Vector<uint16_t, 32> zmm_u16;
	typedef SIMD_Vector<uint32_t, 16> zmm_u32;
	typedef SIMD_Vector<uint64_t, 8> zmm_u64;
	typedef SIMD_Vector<float, 16> zmm_f32;
	typedef SIMD_Vector<double, 8> zmm_f64;
}