#pragma once
#include "../namespace.h"
#include "../tags.h"
#include "../SIMD_BitMask.h"
#include "../SIMD_Vector.h"
#include "../FeatureSet.h"
#include "../funcs.h"
#include "../tables.h"

namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		namespace ISA
		{
			using namespace concepts;
			using namespace utils;
			template<internals::FeatureSet FS>
			struct AVX
			{
				template<typename S, size_t N>
					requires ((is_f32<S> || is_f64<S>) && sizeof(SIMD_Vector<S, N>) > 16)
				static SIMD_Vector<S, N> eval(op_abs, const SIMD_Vector<S, N>& a)
				{
					using U = same_size_uint_t<S>::type;
					constexpr U sb = U(1) << (sizeof(S) * 8 - 1);
					return logic_xor(a, SIMD_Vector<S, N>(std::bit_cast<S>(~sb))); //force sign bit to 0
				}

				template<typename S, size_t N>
					requires ((is_f32<S> || is_f64<S>) && sizeof(SIMD_Vector<S, N>) > 16)
				static SIMD_Vector<S, N> eval(op_add, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32) return { add(a.lo(), b.lo()), add(a.hi(),b.hi()) };
					else if constexpr (ymm_sized<T> && is_f64<S>) return _mm256_add_pd(a, b);
					else if constexpr (ymm_sized<T> && is_f32<S>) return _mm256_add_ps(a, b);
					else static_assert(always_false_v<T>);
				}
				template<typename S, size_t N>
					requires ((is_f32<S> || is_f64<S>) && sizeof(SIMD_Vector<S, N>) > 16)
				static SIMD_Vector<S, N> eval(op_sub, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32) return { sub(a.lo(), b.lo()), sub(a.hi(),b.hi()) };
					else if constexpr (ymm_sized<T> && is_f64<S>) return _mm256_sub_pd(a, b);
					else if constexpr (ymm_sized<T> && is_f32<S>) return _mm256_sub_ps(a, b);
					else static_assert(always_false_v<T>);
				}

				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 16)
				static SIMD_Vector<S, N> eval(op_and, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32) return { logic_and(a.lo(), b.lo()), logic_and(a.hi(), b.hi()) };
					else if constexpr (ymm_sized<T>) return _mm256_and_ps(vreinterpret<__m256>(a), vreinterpret<__m256>(b));
					else static_assert(always_false_v<T>);
				}
				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 16)
				static SIMD_Vector<S, N> eval(op_or, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32) return { logic_or(a.lo(), b.lo()), logic_or(a.hi(), b.hi()) };
					else if constexpr (ymm_sized<T>) return _mm256_or_ps(vreinterpret<__m256>(a), vreinterpret<__m256>(b));
					else static_assert(always_false_v<T>);
				}
				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 16)
				static SIMD_Vector<S, N> eval(op_xor, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32) return { logic_xor(a.lo(), b.lo()), logic_xor(a.hi(), b.hi()) };
					else if constexpr (ymm_sized<T>) return _mm256_xor_ps(vreinterpret<__m256>(a), vreinterpret<__m256>(b));
					else static_assert(always_false_v<T>);
				}
				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 16)
				static SIMD_Vector<S, N> eval(op_not, const SIMD_Vector<S, N>& a)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32) return { logic_not(a.lo()), logic_not(a.hi()) };
					//can't use compare random vectors trick, since FP comparisons are wonky with NaNs and signed 0, thus, have to explicitly set all ones
					else if (ymm_sized<T>) return _mm256_xor_ps(vreinterpret<__m256>(a), _mm256_set1_ps(std::bit_cast<float>(0xFFFFFFFF)));
					else static_assert(always_false_v<T>);
				}

				template<typename S, size_t N, typename I>
					requires (sizeof(S) >= 4 && concepts::any_int<I>)
				static SIMD_Vector<S, N> eval(op_permx, const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& ind)
				{
					using namespace concepts;
					using canon_t = typename same_size_uint_t<S>::type;
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(I) != sizeof(S)) return permx(a, vcvt<canon_t>(ind));
					else if constexpr (sizeof(T) > 16)
					{
						auto alo = a.lo();
						auto ahi = a.hi();
						return { permx2(alo, ahi, ind.lo()), permx2(alo, ahi, ind.hi()) };
					}
					else
					{
						//these are only good for 128 bits, 256 are intra-lanes
						if constexpr (xmm_sized<T> && sizeof(S) == 8) return vreinterpret<T>(_mm_permutevar_pd(vreinterpret<__m128d>(a), ind));
						else if constexpr (xmm_sized<T> && sizeof(S) == 4) return vreinterpret<T>(_mm_permutevar_ps(vreinterpret<__m128d>(a), ind));
						else static_assert(always_false_v<T>);
					}
				}

				template<typename S, size_t N>
					requires (sizeof(S) >= 4)
				static SIMD_Vector<S, N> eval(op_load<S, N>, const void* p, const SIMD_BitMask<N>& mask = SIMD_BitMask<N>::AllOnes, const SIMD_Vector<S, N>& src = 0)
				{
					using T = SIMD_Vector<S, N>;
					using F = std::conditional_t<(sizeof(S) > 4), double, float>;
					using I = std::conditional_t<(sizeof(S) > 4), int64_t, int32_t>;
					const F* sp = reinterpret_cast<const F*>(p);
					if constexpr (sizeof(T) > 32) return { load<S,N / 2>(sp, mask.lo(), src.lo()), load<S,N / 2>(sp + N / 2, mask.hi(),src.hi()) };
					else
					{
						auto vec_mask = mask2vec<I, N>(mask);
						if constexpr (ymm_sized<T> && sizeof(S) == 8) return _mm256_maskload_pd(sp, vec_mask);
						else if constexpr (ymm_sized<T> && sizeof(S) == 4) return _mm256_maskload_ps(sp, vec_mask);
						else if constexpr (xmm_sized<T> && sizeof(S) == 8) return _mm_maskload_pd(sp, vec_mask);
						else if constexpr (xmm_sized<T> && sizeof(S) == 4) return _mm_maskload_ps(sp, vec_mask);
						else static_assert(always_false_v<T>);
					}
				}

				template<typename S, size_t N>
					requires (sizeof(S) >= 4)
				static void eval(op_store, SIMD_Vector<S, N> vec, void* p, const SIMD_BitMask<N>& mask = SIMD_BitMask<N>::AllOnes)
				{
					using T = SIMD_Vector<S, N>;
					using F = std::conditional_t<(sizeof(S) > 4), double, float>;
					using I = std::conditional_t<(sizeof(S) > 4), int64_t, int32_t>;
					F* sp = reinterpret_cast<F*>(p);
					if constexpr (sizeof(T) > 32)
					{
						store(vec.lo(), sp, mask.lo());
						store(vec.hi(), sp + N / 2, mask.hi());
					}
					else
					{
						auto vec_mask = mask2vec<I, N>(mask);
						if constexpr (ymm_sized<T> && sizeof(S) == 8) _mm256_maskstore_pd(sp, vec_mask, vec);
						else if constexpr (ymm_sized<T> && sizeof(S) == 4) _mm256_maskstore_ps(sp, vec_mask, vec);
						else if constexpr (xmm_sized<T> && sizeof(S) == 8) _mm_maskstore_pd(sp, vec_mask, vec);
						else if constexpr (xmm_sized<T> && sizeof(S) == 4) _mm_maskstore_ps(sp, vec_mask, vec);
						else static_assert(always_false_v<T>);
					}
				}

			};
		}
	}
}