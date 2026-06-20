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
			struct AVX
			{
				static inline constexpr FeatureSet FS = internals::FS_current;
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
				static SIMD_Vector<S, N> eval(op_load<S, N>, const void* p, const SIMD_Mask<S, N>& mask = SIMD_Mask<S, N>::AllOnes(), const SIMD_Vector<S, N>& src = 0)
				{
					using T = SIMD_Vector<S, N>;
					using F = std::conditional_t<(sizeof(S) > 4), double, float>;
					using I = std::conditional_t<(sizeof(S) > 4), int64_t, int32_t>;
					const F* sp = reinterpret_cast<const F*>(p);
					if constexpr (sizeof(T) > 32) return { load<S,N / 2>(sp, mask.lo(), src.lo()), load<S,N / 2>(sp + N / 2, mask.hi(),src.hi()) };
					else
					{
						auto vec_mask = vcast<SIMD_Vector<I, N>>(mask.as_vector());
						if constexpr (ymm_sized<T> && sizeof(S) == 8) return _mm256_maskload_pd(sp, vec_mask);
						else if constexpr (ymm_sized<T> && sizeof(S) == 4) return _mm256_maskload_ps(sp, vec_mask);
						else if constexpr (xmm_sized<T> && sizeof(S) == 8) return _mm_maskload_pd(sp, vec_mask);
						else if constexpr (xmm_sized<T> && sizeof(S) == 4) return _mm_maskload_ps(sp, vec_mask);
						else static_assert(always_false_v<T>);
					}
				}

				template<typename S, size_t N>
					requires (sizeof(S) >= 4)
				static void eval(op_store, SIMD_Vector<S, N> vec, void* p, const SIMD_Mask<S, N>& mask = SIMD_Mask<S, N>::AllOnes())
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
						auto vec_mask = SIMD_Mask<I, N>(mask).as_vector();
						if constexpr (ymm_sized<T> && sizeof(S) == 8) _mm256_maskstore_pd(sp, vec_mask, vec);
						else if constexpr (ymm_sized<T> && sizeof(S) == 4) _mm256_maskstore_ps(sp, vec_mask, vec);
						else if constexpr (xmm_sized<T> && sizeof(S) == 8) _mm_maskstore_pd(sp, vec_mask, vec);
						else if constexpr (xmm_sized<T> && sizeof(S) == 4) _mm_maskstore_ps(sp, vec_mask, vec);
						else static_assert(always_false_v<T>);
					}
				}

				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 16 && (is_f32<S> || is_f64<S>))
				static SIMD_Mask<S, N> eval(op_cmpeq, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32)  return { cmp_equal(a.lo(),b.lo()), cmp_equal(a.hi(),b.hi()) };
					else if constexpr (ymm_sized<T> && is_f64<S>) return _mm256_cmp_pd(a, b, _CMP_EQ_OQ);
					else if constexpr (ymm_sized<T> && is_f32<S>) return _mm256_cmp_ps(a, b, _CMP_EQ_OQ);
					else static_assert(always_false_v<T>);
				}
				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 16 && (is_f32<S> || is_f64<S>))
				static SIMD_Mask<S, N> eval(op_cmpneq, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32)  return { cmp_not_equal(a.lo(),b.lo()), cmp_not_equal(a.hi(),b.hi()) };
					else if constexpr (ymm_sized<T> && is_f64<S>) return _mm256_cmp_pd(a, b, _CMP_NEQ_OQ);
					else if constexpr (ymm_sized<T> && is_f32<S>) return _mm256_cmp_ps(a, b, _CMP_NEQ_OQ);
					else static_assert(always_false_v<T>);
				}
				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 16 && (is_f32<S> || is_f64<S>))
				static SIMD_Mask<S, N> eval(op_cmpgt, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32)  return { cmp_greater(a.lo(),b.lo()), cmp_greater(a.hi(),b.hi()) };
					else if constexpr (ymm_sized<T> && is_f64<S>) return _mm256_cmp_pd(a, b, _CMP_GT_OQ);
					else if constexpr (ymm_sized<T> && is_f32<S>) return _mm256_cmp_ps(a, b, _CMP_GT_OQ);
					else static_assert(always_false_v<T>);
				}
				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 16 && (is_f32<S> || is_f64<S>))
				static SIMD_Mask<S, N> eval(op_cmpge, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32)  return { cmp_greater_or_equal(a.lo(),b.lo()), cmp_greater_or_equal(a.hi(),b.hi()) };
					else if constexpr (ymm_sized<T> && is_f64<S>) return _mm256_cmp_pd(a, b, _CMP_GE_OQ);
					else if constexpr (ymm_sized<T> && is_f32<S>) return _mm256_cmp_ps(a, b, _CMP_GE_OQ);
					else static_assert(always_false_v<T>);
				}
				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 16 && (is_f32<S> || is_f64<S>))
				static SIMD_Mask<S, N> eval(op_cmplt, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32)  return { cmp_less(a.lo(),b.lo()), cmp_less(a.hi(),b.hi()) };
					else if constexpr (ymm_sized<T> && is_f64<S>) return _mm256_cmp_pd(a, b, _CMP_LT_OQ);
					else if constexpr (ymm_sized<T> && is_f32<S>) return _mm256_cmp_ps(a, b, _CMP_LT_OQ);
					else static_assert(always_false_v<T>);
				}
				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 16 && (is_f32<S> || is_f64<S>))
				static SIMD_Mask<S, N> eval(op_cmple, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32)  return { cmp_less_or_equal(a.lo(),b.lo()), cmp_less_or_equal(a.hi(),b.hi()) };
					else if constexpr (ymm_sized<T> && is_f64<S>) return _mm256_cmp_pd(a, b, _CMP_LE_OQ);
					else if constexpr (ymm_sized<T> && is_f32<S>) return _mm256_cmp_ps(a, b, _CMP_LE_OQ);
					else static_assert(always_false_v<T>);
				}

				template<typename S, size_t N>
					requires (sizeof(S) >= 4 && sizeof(SIMD_Vector<S, N>) >= 17)
				static SIMD_Mask<S, N>::UintT eval(op_movemask, const SIMD_Vector<S, N>& a)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32) return concat_bitmasks<N / 2>(movemask(a.lo()), movemask(a.hi()));
					else if constexpr (ymm_sized<T> && sizeof(S) == 8) return _mm256_movemask_pd(vreinterpret<__m256d>(a));
					else if constexpr (ymm_sized<T> && sizeof(S) == 4) return _mm256_movemask_ps(vreinterpret<__m256>(a));
					else static_assert(always_false_v<T>);
				}

				/* TODO: low bits will result in tiny values, potentially crushing comparison with 0. May play around it by oring to get something into exponent.
				* Watch out of signed 0 and NaN!
				template<typename S, size_t N>
					requires (sizeof(S) >= 4 && sizeof(SIMD_Vector<S, N>) >= 17)
				static SIMD_Vector<S, N> eval(op_mask2vec<S, N>, const SIMD_Mask<S,N>& a)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32) return { mask2vec<S>(a.lo()),mask2vec<S>(a.hi()) };
					else if constexpr (ymm_sized<T> && sizeof(S) == 8)
					{
						__m256i broadcasted = _mm256_set1_epi64x(a);
						__m256d x = _mm256_andnot_pd(_mm256_castsi256_pd(broadcasted), _mm256_castsi256_pd(_mm256_setr_epi64x(1, 2, 4, 8)));
						return _mm256_cmp_pd(x, _mm256_setzero_pd());
					}
					else if constexpr (ymm_sized<T> && any_i32<S>)
					{
						__m256i broadcasted = _mm256_set1_epi32(a);
						__m256i x = _mm256_andnot_si256(broadcasted, _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128));
						return _mm256_cmpeq_epi32(x, _mm256_set1_epi32(0));
					}
					else if constexpr (ymm_sized<T> && any_i16<S>)
					{
						__m256i broadcasted = _mm256_set1_epi16(a);
						__m256i x = _mm256_andnot_si256(broadcasted, _mm256_setr_epi16(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768));
						return _mm256_cmpeq_epi16(x, _mm256_set1_epi16(0));
					}
				}*/
			};
		}
	}
}