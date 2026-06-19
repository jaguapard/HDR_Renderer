#pragma once
#include "../namespace.h"
#include "../tags.h"
#include "../SIMD_BitMask.h"
#include "../SIMD_Vector.h"
#include "../FeatureSet.h"
namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		namespace ISA
		{
			struct AVX512VBMI
			{
				static inline constexpr FeatureSet FS = internals::FS_current;
				template<typename S, size_t N, typename I>
					requires (concepts::any_i8<S> && concepts::any_int<I> && sizeof(SIMD_Vector<S,N>) >= (FS.has(AVX512_VL) ? 0 : 33))
				static SIMD_Vector<S, N> eval(op_permx, const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& ind)
				{
					using namespace concepts;
					using canon_t = typename same_size_uint_t<S>::type;
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(I) != sizeof(S)) return permx(a, vcvt<canon_t>(ind));
					else if constexpr (sizeof(T) > 64)
					{
						auto alo = a.lo();
						auto ahi = a.hi();
						return { permx2(alo, ahi, ind.lo()), permx2(alo, ahi, ind.hi()) };
					}
					else if constexpr (zmm_sized<T> && any_i8<S>) return _mm512_permutexvar_epi8(ind, a);
					else if constexpr (FS.has(AVX512_VL) & ymm_sized<T> && any_i8<S>) return _mm256_permutexvar_epi8(ind, a);
					else if constexpr (FS.has(AVX512_VL) & xmm_sized<T> && any_i8<S>) return _mm_permutexvar_epi8(ind, a);
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N, typename I>
					requires (concepts::any_i8<S>&& concepts::any_int<I> && sizeof(SIMD_Vector<S, N>) >= (FS.has(AVX512_VL) ? 0 : 33))
				static SIMD_Vector<S, N> eval(op_permx2, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b, const SIMD_Vector<I, N>& ind)
				{
					using namespace concepts;
					using canon_t = typename same_size_uint_t<S>::type;
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
					else static_assert(always_false_v<S>);
				}
			};
		}
	}
}