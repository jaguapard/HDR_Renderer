#pragma once
#include "shared.h"

namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		using namespace meta;
		struct ISA_F16C
		{
			template <typename Op, size_t N>
			requires (IsCvtOp<Op> && std::same_as<typename Op::cvt_to_t, fp16_t>)
			static SIMD_Vector<fp16_t, N> eval(const SIMD_Vector<float, N>& a)
			{
				using T = SIMD_Vector<float, N>;
				if constexpr (sizeof(T) > 32) return { vcvt<fp16_t>(a.lo()), vcvt<fp16_t>(a.hi()) };
				else if constexpr (ymm_sized<T>) return _mm256_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT);
				else if constexpr (xmm_sized<T>) return _mm_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT);
				else static_assert(always_false_v<T>);
			}
			template <typename Op, size_t N>
				requires (IsCvtOp<Op> && std::same_as<typename Op::cvt_to_t, float>)
			static SIMD_Vector<float, N> eval(const SIMD_Vector<fp16_t, N>& a)
			{
				using T = SIMD_Vector<float, N>;
				if constexpr (sizeof(T) > 32) return { vcvt<float>(a.lo()), vcvt<float>(a.hi()) };
				else if constexpr (ymm_sized<T>) return _mm256_cvtph_ps(a);
				else if constexpr (xmm_sized<T>) return _mm_cvtph_ps(a);
				else static_assert(always_false_v<T>);
			}
		};
	}
}