#pragma once
#include "shared.h"
namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		struct ISA_AVX512_VBMI
		{
			//static inline constexpr FeatureSet FS = internals::FS_current;
			//TODO: allow same sized non-int
			template<typename Op, typename S, size_t N, typename I>
				requires (std::same_as<Op,op_permx> && meta::any_i8<S>&& meta::any_int<I> && sizeof(SIMD_Vector<S, N>) >= (FS.has(AVX512_VL) ? 0 : 33))
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& ind)
			{
				using namespace meta;
				using canon_t = typename ScalarTraits<S>::UintT;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(I) != sizeof(S)) return permx(a, vcvt<canon_t>(ind));
				else if constexpr (sizeof(T) > 64)
				{
					auto alo = a.lo();
					auto ahi = a.hi();
					return T{ permx2(alo, ahi, ind.lo()), permx2(alo, ahi, ind.hi()) };
				}
				else if constexpr (zmm_sized<T> && any_i8<S>) return _mm512_permutexvar_epi8(ind, a);
				else if constexpr (FS.has(AVX512_VL) & ymm_sized<T> && any_i8<S>) return _mm256_permutexvar_epi8(ind, a);
				else if constexpr (FS.has(AVX512_VL) & xmm_sized<T> && any_i8<S>) return _mm_permutexvar_epi8(ind, a);
				else return fail_ack_t{};
			}
			template<typename Op, typename S, size_t N, typename I>
				requires (std::same_as<Op, op_permx2> && meta::any_i8<S>&& meta::any_int<I> && sizeof(SIMD_Vector<S, N>) >= (FS.has(AVX512_VL) ? 0 : 33))
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b, const SIMD_Vector<I, N>& ind)
			{
				using namespace meta;
				using canon_t = typename ScalarTraits<S>::UintT;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(I) != sizeof(S)) return permx(a, vcvt<canon_t>(ind));
				else if constexpr (sizeof(T) > 64)
				{
					T pa = permx(a, ind);
					T pb = permx(b, ind);
					return mask_mov(pb, (ind & (2 * N - 1)) < N, pa);
				}
				else if constexpr (zmm_sized<T> && any_i8<S>) return _mm512_permutex2var_epi8(a, ind, b);
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && any_i8<S>) return _mm256_permutex2var_epi8(a, ind, b);
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && any_i8<S>) return _mm_permutex2var_epi8(a, ind, b);
				else return fail_ack_t{};
			}
		};
	}
}