#pragma once
#include "namespace.h"
#include "SIMD_Mask.h"
#include "SIMD_Vector.h"
#include "meta/meta.h"

namespace AVXXY_NAMESPACE
{
	typedef SIMD_Vector<int8_t, 1> i8x1;
	typedef SIMD_Vector<int8_t, 2> i8x2;
	typedef SIMD_Vector<int8_t, 4> i8x4;
	typedef SIMD_Vector<int8_t, 8> i8x8;
	typedef SIMD_Vector<int8_t, 16> i8x16;
	typedef SIMD_Vector<int8_t, 32> i8x32;
	typedef SIMD_Vector<int8_t, 64> i8x64;

	typedef SIMD_Vector<int16_t, 1> i16x1;
	typedef SIMD_Vector<int16_t, 2> i16x2;
	typedef SIMD_Vector<int16_t, 4> i16x4;
	typedef SIMD_Vector<int16_t, 8> i16x8;
	typedef SIMD_Vector<int16_t, 16> i16x16;
	typedef SIMD_Vector<int16_t, 32> i16x32;
	typedef SIMD_Vector<int16_t, 64> i16x64;

	typedef SIMD_Vector<int32_t, 1> i32x1;
	typedef SIMD_Vector<int32_t, 2> i32x2;
	typedef SIMD_Vector<int32_t, 4> i32x4;
	typedef SIMD_Vector<int32_t, 8> i32x8;
	typedef SIMD_Vector<int32_t, 16> i32x16;
	typedef SIMD_Vector<int32_t, 32> i32x32;
	typedef SIMD_Vector<int32_t, 64> i32x64;

	typedef SIMD_Vector<int64_t, 1> i64x1;
	typedef SIMD_Vector<int64_t, 2> i64x2;
	typedef SIMD_Vector<int64_t, 4> i64x4;
	typedef SIMD_Vector<int64_t, 8> i64x8;
	typedef SIMD_Vector<int64_t, 16> i64x16;
	typedef SIMD_Vector<int64_t, 32> i64x32;
	typedef SIMD_Vector<int64_t, 64> i64x64;

	typedef SIMD_Vector<uint8_t, 1> u8x1;
	typedef SIMD_Vector<uint8_t, 2> u8x2;
	typedef SIMD_Vector<uint8_t, 4> u8x4;
	typedef SIMD_Vector<uint8_t, 8> u8x8;
	typedef SIMD_Vector<uint8_t, 16> u8x16;
	typedef SIMD_Vector<uint8_t, 32> u8x32;
	typedef SIMD_Vector<uint8_t, 64> u8x64;

	typedef SIMD_Vector<uint16_t, 1> u16x1;
	typedef SIMD_Vector<uint16_t, 2> u16x2;
	typedef SIMD_Vector<uint16_t, 4> u16x4;
	typedef SIMD_Vector<uint16_t, 8> u16x8;
	typedef SIMD_Vector<uint16_t, 16> u16x16;
	typedef SIMD_Vector<uint16_t, 32> u16x32;
	typedef SIMD_Vector<uint16_t, 64> u16x64;

	typedef SIMD_Vector<fp16_t, 1> fp16x1;
	typedef SIMD_Vector<fp16_t, 2> fp16x2;
	typedef SIMD_Vector<fp16_t, 4> fp16x4;
	typedef SIMD_Vector<fp16_t, 8> fp16x8;
	typedef SIMD_Vector<fp16_t, 16> fp16x16;
	typedef SIMD_Vector<fp16_t, 32> fp16x32;
	typedef SIMD_Vector<fp16_t, 64> fp16x64;

	typedef SIMD_Vector<bf16_t, 1> bf16x1;
	typedef SIMD_Vector<bf16_t, 2> bf16x2;
	typedef SIMD_Vector<bf16_t, 4> bf16x4;
	typedef SIMD_Vector<bf16_t, 8> bf16x8;
	typedef SIMD_Vector<bf16_t, 16> bf16x16;
	typedef SIMD_Vector<bf16_t, 32> bf16x32;
	typedef SIMD_Vector<bf16_t, 64> bf16x64;

	typedef SIMD_Vector<uint32_t, 1> u32x1;
	typedef SIMD_Vector<uint32_t, 2> u32x2;
	typedef SIMD_Vector<uint32_t, 4> u32x4;
	typedef SIMD_Vector<uint32_t, 8> u32x8;
	typedef SIMD_Vector<uint32_t, 16> u32x16;
	typedef SIMD_Vector<uint32_t, 32> u32x32;
	typedef SIMD_Vector<uint32_t, 64> u32x64;

	typedef SIMD_Vector<uint64_t, 1> u64x1;
	typedef SIMD_Vector<uint64_t, 2> u64x2;
	typedef SIMD_Vector<uint64_t, 4> u64x4;
	typedef SIMD_Vector<uint64_t, 8> u64x8;
	typedef SIMD_Vector<uint64_t, 16> u64x16;
	typedef SIMD_Vector<uint64_t, 32> u64x32;
	typedef SIMD_Vector<uint64_t, 64> u64x64;

	typedef SIMD_Vector<float, 1> f32x1;
	typedef SIMD_Vector<float, 2> f32x2;
	typedef SIMD_Vector<float, 4> f32x4;
	typedef SIMD_Vector<float, 8> f32x8;
	typedef SIMD_Vector<float, 16> f32x16;
	typedef SIMD_Vector<float, 32> f32x32;
	typedef SIMD_Vector<float, 64> f32x64;

	typedef SIMD_Vector<double, 1> f64x1;
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
	typedef SIMD_Vector<fp16_t, 8> xmm_fp16;
	typedef SIMD_Vector<bf16_t, 8> xmm_bf16;
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
	typedef SIMD_Vector<fp16_t, 16> ymm_fp16;
	typedef SIMD_Vector<bf16_t, 16> ymm_bf16;
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
	typedef SIMD_Vector<fp16_t, 32> zmm_fp16;
	typedef SIMD_Vector<bf16_t, 32> zmm_bf16;
	typedef SIMD_Vector<uint32_t, 16> zmm_u32;
	typedef SIMD_Vector<uint64_t, 8> zmm_u64;
	typedef SIMD_Vector<float, 16> zmm_f32;
	typedef SIMD_Vector<double, 8> zmm_f64;


	typedef SIMD_Vector<int8_t, meta::REG_LANE_COUNT_FOR<int8_t>> i8xn;
	typedef SIMD_Vector<int16_t, meta::REG_LANE_COUNT_FOR<int16_t>> i16xn;
	typedef SIMD_Vector<int32_t, meta::REG_LANE_COUNT_FOR<int32_t>> i32xn;
	typedef SIMD_Vector<int64_t, meta::REG_LANE_COUNT_FOR<int64_t>> i64xn;
	typedef SIMD_Vector<uint64_t, meta::REG_LANE_COUNT_FOR<uint64_t>> u64xn;
	typedef SIMD_Vector<uint32_t, meta::REG_LANE_COUNT_FOR<uint32_t>> u32xn;
	typedef SIMD_Vector<uint16_t, meta::REG_LANE_COUNT_FOR<uint16_t>> u16xn;
	typedef SIMD_Vector<uint8_t, meta::REG_LANE_COUNT_FOR<uint8_t>> u8xn;
	typedef SIMD_Vector<float, meta::REG_LANE_COUNT_FOR<float>> f32xn;
	typedef SIMD_Vector<double, meta::REG_LANE_COUNT_FOR<double>> f64xn;
	typedef SIMD_Vector<fp16_t, meta::REG_LANE_COUNT_FOR<fp16_t>> fp16xn;
	typedef SIMD_Vector<bf16_t, meta::REG_LANE_COUNT_FOR<bf16_t>> bf16xn;



	typedef SIMD_Mask<meta::ScalarSizeClassEnum::byte, meta::REG_LANE_COUNT_FOR<int8_t>> maskn_b;
	typedef SIMD_Mask<meta::ScalarSizeClassEnum::word, meta::REG_LANE_COUNT_FOR<int16_t>> maskn_w;
	typedef SIMD_Mask<meta::ScalarSizeClassEnum::dword, meta::REG_LANE_COUNT_FOR<int32_t>> maskn_d;
	typedef SIMD_Mask<meta::ScalarSizeClassEnum::qword, meta::REG_LANE_COUNT_FOR<int64_t>> maskn_q;

	typedef SIMD_Mask<meta::ScalarSizeClassEnum::byte, 1> mask1b;
	typedef SIMD_Mask<meta::ScalarSizeClassEnum::word, 1> mask1w;
	typedef SIMD_Mask<meta::ScalarSizeClassEnum::dword, 1> mask1d;
	typedef SIMD_Mask<meta::ScalarSizeClassEnum::qword, 1> mask1q;

	typedef SIMD_Mask<meta::ScalarSizeClassEnum::byte, 2> mask2b;
	typedef SIMD_Mask<meta::ScalarSizeClassEnum::word, 2> mask2w;
	typedef SIMD_Mask<meta::ScalarSizeClassEnum::dword, 2> mask2d;
	typedef SIMD_Mask<meta::ScalarSizeClassEnum::qword, 2> mask2q;

	typedef SIMD_Mask<meta::ScalarSizeClassEnum::byte, 4> mask4b;
	typedef SIMD_Mask<meta::ScalarSizeClassEnum::word, 4> mask4w;
	typedef SIMD_Mask<meta::ScalarSizeClassEnum::dword, 4> mask4d;
	typedef SIMD_Mask<meta::ScalarSizeClassEnum::qword, 4> mask4q;

	typedef SIMD_Mask<meta::ScalarSizeClassEnum::byte, 8> mask8b;
	typedef SIMD_Mask<meta::ScalarSizeClassEnum::word, 8> mask8w;
	typedef SIMD_Mask<meta::ScalarSizeClassEnum::dword, 8> mask8d;
	typedef SIMD_Mask<meta::ScalarSizeClassEnum::qword, 8> mask8q;

	typedef SIMD_Mask<meta::ScalarSizeClassEnum::byte, 16> mask16b;
	typedef SIMD_Mask<meta::ScalarSizeClassEnum::word, 16> mask16w;
	typedef SIMD_Mask<meta::ScalarSizeClassEnum::dword, 16> mask16d;
	typedef SIMD_Mask<meta::ScalarSizeClassEnum::qword, 16> mask16q;

	typedef SIMD_Mask<meta::ScalarSizeClassEnum::byte, 32> mask32b;
	typedef SIMD_Mask<meta::ScalarSizeClassEnum::word, 32> mask32w;
	typedef SIMD_Mask<meta::ScalarSizeClassEnum::dword, 32> mask32d;
	typedef SIMD_Mask<meta::ScalarSizeClassEnum::qword, 32> mask32q;

	typedef SIMD_Mask<meta::ScalarSizeClassEnum::byte, 64> mask64b;
	typedef SIMD_Mask<meta::ScalarSizeClassEnum::word, 64> mask64w;
	typedef SIMD_Mask<meta::ScalarSizeClassEnum::dword, 64> mask64d;
	typedef SIMD_Mask<meta::ScalarSizeClassEnum::qword, 64> mask64q;
}