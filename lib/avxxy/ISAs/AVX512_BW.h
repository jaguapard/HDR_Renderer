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
			template<internals::FeatureSet FS>
			struct AVX512BW
			{
				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_mask_mov, const SIMD_Vector<S, N>& ifBitClear, const SIMD_BitMask<N>& mask, const SIMD_Vector<S, N>& ifBitSet)
					requires (sizeof(S) < 4 && concepts::zmm_sized<SIMD_Vector<S, N>>)
				{
					using namespace concepts;
					if constexpr (any_i16<S>) return _mm512_mask_mov_epi16(ifBitClear, mask, ifBitSet);
					else if constexpr (any_i8<S>) return _mm512_mask_mov_epi8(ifBitClear, mask, ifBitSet);
					else static_assert(always_false_v<S>);
				}

				template<typename S, size_t N>
				requires (sizeof(SIMD_Vector<S,N>) > 32 && sizeof(S) < 4)
				static SIMD_BitMask<N> eval(op_cmpeq, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					if constexpr (sizeof(SIMD_Vector<S, N>) > 64) return { cmp_equal(a.lo(), b.lo()), cmp_equal(a.hi(), b.hi()) };
					else if constexpr (is_i16<S>) return _mm512_cmpeq_epi16_mask(a, b);
					else if constexpr (is_u16<S>) return _mm512_cmpeq_epu16_mask(a, b);
					else if constexpr (is_i8<S>) return _mm512_cmpeq_epi8_mask(a, b);
					else if constexpr (is_u8<S>) return _mm512_cmpeq_epu8_mask(a, b);
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 32 && sizeof(S) < 4)
				static SIMD_BitMask<N> eval(op_cmpneq, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					if constexpr (sizeof(SIMD_Vector<S, N>) > 64) return { cmp_not_equal(a.lo(), b.lo()), cmp_not_equal(a.hi(), b.hi()) };
					else if constexpr (is_i16<S>) return _mm512_cmpneq_epi16_mask(a, b);
					else if constexpr (is_u16<S>) return _mm512_cmpneq_epu16_mask(a, b);
					else if constexpr (is_i8<S>) return _mm512_cmpneq_epi8_mask(a, b);
					else if constexpr (is_u8<S>) return _mm512_cmpneq_epu8_mask(a, b);
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 32 && sizeof(S) < 4)
				static SIMD_BitMask<N> eval(op_cmpgt, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					if constexpr (sizeof(SIMD_Vector<S, N>) > 64) return { cmp_greater(a.lo(), b.lo()), cmp_greater(a.hi(), b.hi()) };
					else if constexpr (is_i16<S>) return _mm512_cmpgt_epi16_mask(a, b);
					else if constexpr (is_u16<S>) return _mm512_cmpgt_epu16_mask(a, b);
					else if constexpr (is_i8<S>) return _mm512_cmpgt_epi8_mask(a, b);
					else if constexpr (is_u8<S>) return _mm512_cmpgt_epu8_mask(a, b);
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 32 && sizeof(S) < 4)
				static SIMD_BitMask<N> eval(op_cmpge, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					if constexpr (sizeof(SIMD_Vector<S, N>) > 64) return { cmp_greater_or_equal(a.lo(), b.lo()), cmp_greater_or_equal(a.hi(), b.hi()) };
					else if constexpr (is_i16<S>) return _mm512_cmpge_epi16_mask(a, b);
					else if constexpr (is_u16<S>) return _mm512_cmpge_epu16_mask(a, b);
					else if constexpr (is_i8<S>) return _mm512_cmpge_epi8_mask(a, b);
					else if constexpr (is_u8<S>) return _mm512_cmpge_epu8_mask(a, b);
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 32 && sizeof(S) < 4)
				static SIMD_BitMask<N> eval(op_cmplt, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					if constexpr (sizeof(SIMD_Vector<S, N>) > 64) return { cmp_less(a.lo(), b.lo()), cmp_less(a.hi(), b.hi()) };
					else if constexpr (is_i16<S>) return _mm512_cmplt_epi16_mask(a, b);
					else if constexpr (is_u16<S>) return _mm512_cmplt_epu16_mask(a, b);
					else if constexpr (is_i8<S>) return _mm512_cmplt_epi8_mask(a, b);
					else if constexpr (is_u8<S>) return _mm512_cmplt_epu8_mask(a, b);
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 32 && sizeof(S) < 4)
				static SIMD_BitMask<N> eval(op_cmple, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					if constexpr (sizeof(SIMD_Vector<S, N>) > 64) return { cmp_less_or_equal(a.lo(), b.lo()), cmp_less_or_equal(a.hi(), b.hi()) };
					else if constexpr (is_i16<S>) return _mm512_cmple_epi16_mask(a, b);
					else if constexpr (is_u16<S>) return _mm512_cmple_epu16_mask(a, b);
					else if constexpr (is_i8<S>) return _mm512_cmple_epi8_mask(a, b);
					else if constexpr (is_u8<S>) return _mm512_cmple_epu8_mask(a, b);
					else static_assert(always_false_v<S>);
				}

				template<typename S, size_t N, typename I>
					requires (concepts::any_i16<S> && concepts::any_int<I> && concepts::zmm_sized<SIMD_Vector<S, N>>)//sizeof(SIMD_Vector<S,N>& > 32))
				static SIMD_Vector<S, N> eval(op_permx, const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& ind)
				{
					using namespace concepts;
					using canon_t = typename same_size_uint_t<S>::type;
					if constexpr (sizeof(I) != sizeof(S)) return permx(a, vcvt<canon_t>(ind));
					//TODO: add > 64 byte permutex!
					else if constexpr (any_i16<S>) return _mm512_permutexvar_epi16(ind, a);
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N, typename I>
					requires (concepts::any_i16<S> && concepts::any_int<I> && concepts::zmm_sized<SIMD_Vector<S, N>>)//sizeof(SIMD_Vector<S,N>& > 32))
				static SIMD_Vector<S, N> eval(op_permx2, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b, const SIMD_Vector<I, N>& ind)
				{
					using namespace concepts;
					using canon_t = typename same_size_uint_t<S>::type;
					if constexpr (sizeof(I) != sizeof(S)) return permx(a, vcvt<canon_t>(ind));
					//TODO: add > 64 byte permutex2!
					else if constexpr (any_i16<S>) return _mm512_permutex2var_epi16(a, ind, b);
					else static_assert(always_false_v<S>);
				}

				template<typename S, size_t N>
					requires (sizeof(S) < 4 && sizeof(SIMD_Vector<S, N>) > 32)
				static SIMD_Vector<S, N> eval(op_load<S, N>, const void* p, const SIMD_BitMask<N>& mask = SIMD_BitMask<N>::AllOnes, const SIMD_Vector<S, N>& src = 0)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					SIMD_Vector<S, N> ret;
					const S* sp = (const S*)p;
					if constexpr (sizeof(T) > 64) return { load<S,N / 2>(sp,mask.lo(), src.lo()), load<S,N / 2>(sp + N / 2, mask.hi(), src.hi()) };
					else if constexpr (any_i16<S>) return _mm512_mask_loadu_epi16(src, mask, p);
					else if constexpr (any_i8<S>) return _mm512_mask_loadu_epi8(src, mask, p);
					else static_assert(always_false_v<S>);
				}

				template<typename S, size_t N>
					requires (sizeof(S) < 4 && sizeof(SIMD_Vector<S, N>) > 32)
				static void eval(op_store, SIMD_Vector<S, N> vec, void* p, const SIMD_BitMask<N>& mask = SIMD_BitMask<N>::AllOnes)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					SIMD_Vector<S, N> ret;
					const S* sp = (const S*)p;
					if constexpr (sizeof(T) > 64) {
						store(vec.lo(), p, mask.lo());
						store(vec.hi(), sp + N / 2, mask.hi());
					}
					else if constexpr (any_i16<S>) return _mm512_mask_storeu_epi16(p, mask, vec);
					else if constexpr (any_i8<S>) return _mm512_mask_storeu_epi8(p, mask, vec);
					else static_assert(always_false_v<S>);
				}
			};
		}
	}
};