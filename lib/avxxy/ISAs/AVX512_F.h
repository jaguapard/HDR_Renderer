#pragma once
#include "shared.h"
namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		using namespace meta;
		struct ISA_AVX512_F
		{
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_add>)
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return T{ add(a.lo(), b.lo()), add(a.hi(), b.hi()) };
				else if constexpr (zmm_sized<T>)
				{
					if constexpr (is_f64<S>) return _mm512_add_pd(a, b);
					else if constexpr (is_f32<S>) return _mm512_add_ps(a, b);
					else if constexpr (any_i64<S>) return _mm512_add_epi64(a, b);
					else if constexpr (any_i32<S>) return _mm512_add_epi32(a, b);
					else return fail_ack_t{};
				}
				//TODO: check these!
				//else if constexpr (!FS.has(Feature::AVX2) && std::is_signed_v<S>) return vcvt<S>(add(vcvt<int32_t>(a), vcvt<int32_t>(b)));
				//else if constexpr (!FS.has(Feature::AVX2) && std::is_unsigned_v<S>) return vcvt<S>(add(vcvt<uint32_t>(a), vcvt<uint32_t>(b)));
				else return fail_ack_t{};
			}
		};
	}
}