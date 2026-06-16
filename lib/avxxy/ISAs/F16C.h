#pragma once
#include "../namespace.h"
#include "../tags.h"
#include "../SIMD_BitMask.h"
#include "../SIMD_Vector.h"
#include "../FeatureSet.h"
#include "../funcs.h"

namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		namespace ISA
		{
			using namespace concepts;
			using namespace utils;
			template<internals::FeatureSet FS>
			struct F16C
			{
				template <size_t N>
				static SIMD_Vector<uint16_t, N> eval(op_fp32_to_fp16, const SIMD_Vector<float, N>& a)
				{
					using T = SIMD_Vector<float, N>;
					if constexpr (sizeof(T) > 32) return { vcvt_fp32_fp16(a.lo()), vcvt_fp32_fp16(a.hi()) };
					else if constexpr (ymm_sized<T>) return _mm256_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT);
					else if constexpr (xmm_sized<T>) return _mm_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT);
					else static_assert(always_false_v<T>);
				}
				template <size_t N>
				static SIMD_Vector<float, N> eval(op_fp16_to_fp32, const SIMD_Vector<uint16_t, N>& a)
				{
					using T = SIMD_Vector<float, N>;
					if constexpr (sizeof(T) > 32) return { vcvt_fp16_fp32(a.lo()), vcvt_fp16_fp32(a.hi()) };
					else if constexpr (ymm_sized<T>) return _mm256_cvtph_ps(a);
					else if constexpr (xmm_sized<T>) return _mm_cvtph_ps(a);
					else static_assert(always_false_v<T>);
				}
			};
		}
	}
}