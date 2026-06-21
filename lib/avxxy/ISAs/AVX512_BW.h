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
			struct AVX512BW
			{
				static inline constexpr FeatureSet FS = internals::FS_current;
				template<typename S, size_t N>
					requires (sizeof(S) < 4 && sizeof(SIMD_Vector<S, N>) > 32)
				static SIMD_Vector<S, N> eval(op_abs, const SIMD_Vector<S, N>& a)
				{
					using namespace concepts;
					if constexpr (sizeof(SIMD_Vector<S, N>) > 64) return { abs(a.lo()), abs(a.hi()) };
					else if constexpr (is_i16<S>) return _mm512_abs_epi16(a);
					else if constexpr (is_i8<S>) return _mm512_abs_epi8(a);
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_add, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
					requires (sizeof(SIMD_Vector<S, N>) > 32 && sizeof(S) < 4)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 64) return { add(a.lo(), b.lo()), add(a.hi(), b.hi()) };
					else if constexpr (any_i16<S>) return _mm512_add_epi16(a, b);
					else if constexpr (any_i8<S>) return _mm512_add_epi8(a, b);
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_sub, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
					requires (sizeof(SIMD_Vector<S, N>) > 32 && sizeof(S) < 4)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 64) return { sub(a.lo(), b.lo()), sub(a.hi(), b.hi()) };
					else if constexpr (any_i16<S>) return _mm512_sub_epi16(a, b);
					else if constexpr (any_i8<S>) return _mm512_sub_epi8(a, b);
					else static_assert(always_false_v<S>);
				}

				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_mul, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
					requires (sizeof(SIMD_Vector<S, N>) > 32 && any_small_int<S>)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					using canon_t = std::conditional_t<(std::is_signed_v<S>), int16_t, uint16_t>;
					if constexpr (sizeof(T) > 64) return { mul(a.lo(), b.lo()), mul(a.hi(), b.hi()) };
					else if constexpr (any_i16<S>) return _mm512_mullo_epi16(a, b);
					else return vcvt<S>(mul(vcvt<canon_t>(a), vcvt<canon_t>(b)));
				}

				template<typename S, size_t N, typename I>
					requires (concepts::any_small_int<S>&& concepts::any_int<I> && sizeof(SIMD_Vector<S, N>) >= (FS.has(AVX512_VL) ? 0 : 33))
				static SIMD_Vector<S, N> eval(op_shl, const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& b)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (!std::is_same_v<I, uint16_t>) return shift_left(a, vcvt<uint16_t>(b));
					else if constexpr (any_i8<S>) return vcvt<S>(shift_left(vcvt<uint16_t>(a), b));
					else if constexpr (sizeof(T) > 64) return { shift_left(a.lo(),b.lo()), shift_left(a.hi(),b.hi()) };
					else if constexpr (zmm_sized<T> && any_i16<S>) return _mm512_sllv_epi16(a, b);
					else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && any_i16<S>) return _mm256_sllv_epi16(a, b);
					else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && any_i16<S>) return _mm_sllv_epi16(a, b);
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N, typename I>
					requires (concepts::any_small_int<S>&& concepts::any_int<I> && sizeof(SIMD_Vector<S, N>) >= (FS.has(AVX512_VL) ? 0 : 33))
				static SIMD_Vector<S, N> eval(op_shr, const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& b)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (!std::is_same_v<I, uint16_t>) return shift_right(a, vcvt<uint16_t>(b));
					else if constexpr (any_i8<S>) return vcvt<S>(shift_right(vcvt<uint16_t>(a), b));
					else if constexpr (sizeof(T) > 64) return { shift_right(a.lo(),b.lo()), shift_right(a.hi(),b.hi()) };
					else if constexpr (zmm_sized<T> && any_i16<S>) return _mm512_srlv_epi16(a, b);
					else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && any_i16<S>) return _mm256_srlv_epi16(a, b);
					else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && any_i16<S>) return _mm_srlv_epi16(a, b);
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N>
					requires (any_small_int<S> && sizeof(SIMD_Vector<S, N>) >= (FS.has(AVX512_VL) ? 0 : 33))
				static typename SIMD_Vector<S, N>::MaskT eval(op_cmpeq, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 64) return { cmp_equal(a.lo(),b.lo()), cmp_equal(a.hi(),b.hi()) };
					else if constexpr (zmm_sized<T>)
					{
						if constexpr (is_i16<S>) return _mm512_cmpeq_epi16_mask(a, b);
						else if constexpr (is_u16<S>) return _mm512_cmpeq_epu16_mask(a, b);
						else if constexpr (is_i8<S>) return _mm512_cmpeq_epi8_mask(a, b);
						else if constexpr (is_u8<S>) return _mm512_cmpeq_epu8_mask(a, b);
						else static_assert(always_false_v<S>);
					}
					else if constexpr (FS.has(AVX512_VL))
					{
						if constexpr (ymm_sized<T>)
						{
							if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm256_cmpeq_epi16_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm256_cmpeq_epu16_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm256_cmpeq_epi8_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm256_cmpeq_epu8_mask(a, b);
							else static_assert(always_false_v<S>);
						}
						else if constexpr (xmm_sized<T>)
						{
							if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm_cmpeq_epi16_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm_cmpeq_epu16_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm_cmpeq_epi8_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm_cmpeq_epu8_mask(a, b);
							else static_assert(always_false_v<S>);
						}
						else static_assert(always_false_v<S>);
					}
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N>
					requires (any_small_int<S> && sizeof(SIMD_Vector<S, N>) >= (FS.has(AVX512_VL) ? 0 : 33))
				static typename SIMD_Vector<S, N>::MaskT eval(op_cmpneq, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 64) return { cmp_not_equal(a.lo(),b.lo()), cmp_not_equal(a.hi(),b.hi()) };
					else if constexpr (zmm_sized<T>)
					{
						if constexpr (is_i16<S>) return _mm512_cmpneq_epi16_mask(a, b);
						else if constexpr (is_u16<S>) return _mm512_cmpneq_epu16_mask(a, b);
						else if constexpr (is_i8<S>) return _mm512_cmpneq_epi8_mask(a, b);
						else if constexpr (is_u8<S>) return _mm512_cmpneq_epu8_mask(a, b);
						else static_assert(always_false_v<S>);
					}
					else if constexpr (FS.has(AVX512_VL))
					{
						if constexpr (ymm_sized<T>)
						{
							if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm256_cmpneq_epi16_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm256_cmpneq_epu16_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm256_cmpneq_epi8_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm256_cmpneq_epu8_mask(a, b);
							else static_assert(always_false_v<S>);
						}
						else if constexpr (xmm_sized<T>)
						{
							if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm_cmpneq_epi16_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm_cmpneq_epu16_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm_cmpneq_epi8_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm_cmpneq_epu8_mask(a, b);
							else static_assert(always_false_v<S>);
						}
						else static_assert(always_false_v<S>);
					}
					else static_assert(always_false_v<S>);
				}

				template<typename S, size_t N>
					requires (any_small_int<S> && sizeof(SIMD_Vector<S, N>) >= (FS.has(AVX512_VL) ? 0 : 33))
				static typename SIMD_Vector<S, N>::MaskT eval(op_cmplt, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 64) return { cmp_less(a.lo(),b.lo()), cmp_less(a.hi(),b.hi()) };
					else if constexpr (zmm_sized<T>)
					{
						if constexpr (is_i16<S>) return _mm512_cmplt_epi16_mask(a, b);
						else if constexpr (is_u16<S>) return _mm512_cmplt_epu16_mask(a, b);
						else if constexpr (is_i8<S>) return _mm512_cmplt_epi8_mask(a, b);
						else if constexpr (is_u8<S>) return _mm512_cmplt_epu8_mask(a, b);
						else static_assert(always_false_v<S>);
					}
					else if constexpr (FS.has(AVX512_VL))
					{
						if constexpr (ymm_sized<T>)
						{
							if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm256_cmplt_epi16_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm256_cmplt_epu16_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm256_cmplt_epi8_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm256_cmplt_epu8_mask(a, b);
							else static_assert(always_false_v<S>);
						}
						else if constexpr (xmm_sized<T>)
						{
							if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm_cmplt_epi16_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm_cmplt_epu16_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm_cmplt_epi8_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm_cmplt_epu8_mask(a, b);
							else static_assert(always_false_v<S>);
						}
						else static_assert(always_false_v<S>);
					}
					else static_assert(always_false_v<S>);
				}

				template<typename S, size_t N>
					requires (any_small_int<S> && sizeof(SIMD_Vector<S, N>) >= (FS.has(AVX512_VL) ? 0 : 33))
				static typename SIMD_Vector<S, N>::MaskT eval(op_cmple, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 64) return { cmp_less_or_equal(a.lo(),b.lo()), cmp_less_or_equal(a.hi(),b.hi()) };
					else if constexpr (zmm_sized<T>)
					{
						if constexpr (is_i16<S>) return _mm512_cmple_epi16_mask(a, b);
						else if constexpr (is_u16<S>) return _mm512_cmple_epu16_mask(a, b);
						else if constexpr (is_i8<S>) return _mm512_cmple_epi8_mask(a, b);
						else if constexpr (is_u8<S>) return _mm512_cmple_epu8_mask(a, b);
						else static_assert(always_false_v<S>);
					}
					else if constexpr (FS.has(AVX512_VL))
					{
						if constexpr (ymm_sized<T>)
						{
							if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm256_cmple_epi16_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm256_cmple_epu16_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm256_cmple_epi8_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm256_cmple_epu8_mask(a, b);
							else static_assert(always_false_v<S>);
						}
						else if constexpr (xmm_sized<T>)
						{
							if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm_cmple_epi16_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm_cmple_epu16_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm_cmple_epi8_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm_cmple_epu8_mask(a, b);
							else static_assert(always_false_v<S>);
						}
						else static_assert(always_false_v<S>);
					}
					else static_assert(always_false_v<S>);
				}

				template<typename S, size_t N>
					requires (any_small_int<S> && sizeof(SIMD_Vector<S, N>) >= (FS.has(AVX512_VL) ? 0 : 33))
				static typename SIMD_Vector<S, N>::MaskT eval(op_cmpgt, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 64) return { cmp_greater(a.lo(),b.lo()), cmp_greater(a.hi(),b.hi()) };
					else if constexpr (zmm_sized<T>)
					{
						if constexpr (is_i16<S>) return _mm512_cmpgt_epi16_mask(a, b);
						else if constexpr (is_u16<S>) return _mm512_cmpgt_epu16_mask(a, b);
						else if constexpr (is_i8<S>) return _mm512_cmpgt_epi8_mask(a, b);
						else if constexpr (is_u8<S>) return _mm512_cmpgt_epu8_mask(a, b);
						else static_assert(always_false_v<S>);
					}
					else if constexpr (FS.has(AVX512_VL))
					{
						if constexpr (ymm_sized<T>)
						{
							if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm256_cmpgt_epi16_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm256_cmpgt_epu16_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm256_cmpgt_epi8_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm256_cmpgt_epu8_mask(a, b);
							else static_assert(always_false_v<S>);
						}
						else if constexpr (xmm_sized<T>)
						{
							if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm_cmpgt_epi16_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm_cmpgt_epu16_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm_cmpgt_epi8_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm_cmpgt_epu8_mask(a, b);
							else static_assert(always_false_v<S>);
						}
						else static_assert(always_false_v<S>);
					}
					else static_assert(always_false_v<S>);
				}

				template<typename S, size_t N>
					requires (any_small_int<S> && sizeof(SIMD_Vector<S, N>) >= (FS.has(AVX512_VL) ? 0 : 33))
				static typename SIMD_Vector<S, N>::MaskT eval(op_cmpge, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 64) return { cmp_greater_or_equal(a.lo(),b.lo()), cmp_greater_or_equal(a.hi(),b.hi()) };
					else if constexpr (zmm_sized<T>)
					{
						if constexpr (is_i16<S>) return _mm512_cmpge_epi16_mask(a, b);
						else if constexpr (is_u16<S>) return _mm512_cmpge_epu16_mask(a, b);
						else if constexpr (is_i8<S>) return _mm512_cmpge_epi8_mask(a, b);
						else if constexpr (is_u8<S>) return _mm512_cmpge_epu8_mask(a, b);
						else static_assert(always_false_v<S>);
					}
					else if constexpr (FS.has(AVX512_VL))
					{
						if constexpr (ymm_sized<T>)
						{
							if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm256_cmpge_epi16_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm256_cmpge_epu16_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm256_cmpge_epi8_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm256_cmpge_epu8_mask(a, b);
							else static_assert(always_false_v<S>);
						}
						else if constexpr (xmm_sized<T>)
						{
							if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm_cmpge_epi16_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm_cmpge_epu16_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm_cmpge_epi8_mask(a, b);
							else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm_cmpge_epu8_mask(a, b);
							else static_assert(always_false_v<S>);
						}
						else static_assert(always_false_v<S>);
					}
					else static_assert(always_false_v<S>);
				}

				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_mask_mov, const SIMD_Vector<S, N>& ifBitClear, const typename SIMD_Vector<S, N>::MaskT& mask, const SIMD_Vector<S, N>& ifBitSet)
					requires (sizeof(S) < 4 && sizeof(SIMD_Vector<S, N>) >= (FS.has(AVX512_VL) ? 0 : 33))
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 64) return { mask_mov(ifBitClear.lo(), mask.lo(), ifBitSet.lo()), mask_mov(ifBitClear.hi(), mask.hi(), ifBitSet.hi()) };
					else if constexpr (zmm_sized<T> && any_i16<S>) return _mm512_mask_mov_epi16(ifBitClear, mask, ifBitSet);
					else if constexpr (zmm_sized<T> && any_i8<S>) return _mm512_mask_mov_epi8(ifBitClear, mask, ifBitSet);
					else if constexpr (ymm_sized<T> && FS.has(AVX512_VL) && any_i16<S>) return _mm256_mask_mov_epi16(ifBitClear, mask, ifBitSet);
					else if constexpr (ymm_sized<T> && FS.has(AVX512_VL) && any_i8<S>) return _mm256_mask_mov_epi8(ifBitClear, mask, ifBitSet);
					else if constexpr (xmm_sized<T> && FS.has(AVX512_VL) && any_i16<S>) return _mm_mask_mov_epi16(ifBitClear, mask, ifBitSet);
					else if constexpr (xmm_sized<T> && FS.has(AVX512_VL) && any_i8<S>) return _mm_mask_mov_epi8(ifBitClear, mask, ifBitSet);
					else static_assert(always_false_v<S>);
				}

				template<typename To, size_t N, typename From>
					requires ((std::max(sizeof(SIMD_Vector<To, N>), sizeof(SIMD_Vector<From, N>)) > 32 && any_small_int<From> && any_small_int<To>) || (FS.has(AVX512_VL) && any_i16<From> && any_i8<To> && std::max(sizeof(SIMD_Vector<To, N>), sizeof(SIMD_Vector<From, N>)) <= 32))
				static SIMD_Vector<To, N> eval(op_cvt<To>, const SIMD_Vector<From, N>& a)
				{
					using namespace concepts;
					using namespace utils;
					using TV = SIMD_Vector<To, N>;
					using FV = SIMD_Vector<From, N>;
					constexpr size_t MaxSize = std::max(sizeof(TV), sizeof(FV));

					if constexpr (MaxSize > 64) return { vcvt<To>(a.lo()), vcvt<To>(a.hi()) };
					else if constexpr (is_zmm_size(MaxSize) && any_i16<From> && any_i8<To>) return _mm512_cvtepi16_epi8(a);
					else if constexpr (is_zmm_size(MaxSize) && is_i8<From> && any_i16<To>) return _mm512_cvtepi8_epi16(a);
					else if constexpr (is_zmm_size(MaxSize) && is_u8<From> && any_i16<To>) return _mm512_cvtepu8_epi16(a);
					else if constexpr (FS.has(AVX512_VL) && is_ymm_size(MaxSize) && any_i16<From> && any_i8<To>) return _mm256_cvtepi16_epi8(a);
					else if constexpr (FS.has(AVX512_VL) && is_xmm_size(MaxSize) && any_i16<From> && any_i8<To>) return _mm_cvtepi16_epi8(a);
					else static_assert(always_false_v<To>);
				}

				template<typename S, size_t N>
					requires (any_small_int<S> && sizeof(SIMD_Vector<S, N>) > 32)
				static SIMD_Vector<S, N> eval(op_min, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using namespace concepts;
					if constexpr (sizeof(SIMD_Vector<S, N>) > 64) return { min(a.lo(), b.lo()), min(a.hi(),b.hi()) };
					else if constexpr (is_i16<S>) return _mm512_min_epi16(a, b);
					else if constexpr (is_u16<S>) return _mm512_min_epu16(a, b);
					else if constexpr (is_i8<S>) return _mm512_min_epi8(a, b);
					else if constexpr (is_u8<S>) return _mm512_min_epu8(a, b);
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N>
					requires (any_small_int<S> && sizeof(SIMD_Vector<S, N>) > 32)
				static SIMD_Vector<S, N> eval(op_max, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using namespace concepts;
					if constexpr (sizeof(SIMD_Vector<S, N>) > 64) return { max(a.lo(), b.lo()), max(a.hi(),b.hi()) };
					else if constexpr (is_i16<S>) return _mm512_max_epi16(a, b);
					else if constexpr (is_u16<S>) return _mm512_max_epu16(a, b);
					else if constexpr (is_i8<S>) return _mm512_max_epi8(a, b);
					else if constexpr (is_u8<S>) return _mm512_max_epu8(a, b);
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N, typename I>
					requires (concepts::any_small_int<S>&& concepts::any_int<I>&& inRange(sizeof(SIMD_Vector<S, N>), FS.has(AVX512_VL) ? 0 : 33, 64))
				static SIMD_Vector<S, N> eval(op_permx, const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& ind)
				{
					using namespace concepts;
					using canon_t = typename same_size_uint_t<S>::type;
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(I) != sizeof(S)) return permx(a, vcvt<canon_t>(ind));
					else if constexpr (any_i8<S>) return vcvt<S>(permx(vcvt<uint16_t>(a), ind));
					//TODO: add > 64 byte permutex!
					else if constexpr (zmm_sized<T> && any_i16<S>) return _mm512_permutexvar_epi16(ind, a);
					else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && any_i16<S>) return _mm256_permutexvar_epi16(ind, a);
					else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && any_i16<S>) return _mm_permutexvar_epi16(ind, a);
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N, typename I>
					requires (concepts::any_small_int<S>&& concepts::any_int<I>&& inRange(sizeof(SIMD_Vector<S, N>), FS.has(AVX512_VL) ? 0 : 33, 64))
				static SIMD_Vector<S, N> eval(op_permx2, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b, const SIMD_Vector<I, N>& ind)
				{
					using namespace concepts;
					using canon_t = typename same_size_uint_t<S>::type;
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(I) != sizeof(S)) return permx2(a, b, vcvt<canon_t>(ind));
					else if constexpr (any_i8<S>) return vcvt<S>(permx2(vcvt<uint16_t>(a), vcvt<uint16_t>(b), ind));
					//TODO: add > 64 byte permutex2!
					else if constexpr (zmm_sized<T> && any_i16<S>) return _mm512_permutex2var_epi16(a, ind, b);
					else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && any_i16<S>) return _mm256_permutex2var_epi16(a, ind, b);
					else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && any_i16<S>) return _mm_permutex2var_epi16(a, ind, b);
					else static_assert(always_false_v<S>);
				}

				template<typename S, size_t N>
					requires (sizeof(S) < 4 && sizeof(SIMD_Vector<S, N>) >= (FS.has(AVX512_VL) ? 0 : 33))
				static SIMD_Vector<S, N> eval(op_load<S, N>, const void* p, const typename SIMD_Vector<S, N>::MaskT& mask, const SIMD_Vector<S, N>& src)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					SIMD_Vector<S, N> ret;
					const S* sp = (const S*)p;
					if constexpr (sizeof(T) > 64) return { load<S,N / 2>(sp,mask.lo(), src.lo()), load<S,N / 2>(sp + N / 2, mask.hi(), src.hi()) };
					else if constexpr (zmm_sized<T> && any_i16<S>) return _mm512_mask_loadu_epi16(src, mask, p);
					else if constexpr (zmm_sized<T> && any_i8<S>) return _mm512_mask_loadu_epi8(src, mask, p);
					else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && any_i16<S>) return _mm256_mask_loadu_epi16(src, mask, p);
					else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && any_i8<S>) return _mm256_mask_loadu_epi8(src, mask, p);
					else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && any_i16<S>) return _mm_mask_loadu_epi16(src, mask, p);
					else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && any_i8<S>) return _mm_mask_loadu_epi8(src, mask, p);
					else static_assert(always_false_v<S>);
				}

				template<typename S, size_t N>
					requires (sizeof(S) < 4 && sizeof(SIMD_Vector<S, N>) >= (FS.has(AVX512_VL) ? 0 : 33))
				static void eval(op_store, SIMD_Vector<S, N> vec, void* p, const typename SIMD_Vector<S, N>::MaskT& mask)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					SIMD_Vector<S, N> ret;
					const S* sp = (const S*)p;
					if constexpr (sizeof(T) > 64)
					{
						store(vec.lo(), p, mask.lo());
						store(vec.hi(), sp + N / 2, mask.hi());
					}
					else if constexpr (zmm_sized<T> && any_i16<S>) return _mm512_mask_storeu_epi16(p, mask, vec);
					else if constexpr (zmm_sized<T> && any_i8<S>) return _mm512_mask_storeu_epi8(p, mask, vec);
					else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && any_i16<S>) return _mm256_mask_storeu_epi16(p, mask, vec);
					else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && any_i8<S>) return _mm256_mask_storeu_epi8(p, mask, vec);
					else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && any_i16<S>) return _mm_mask_storeu_epi16(p, mask, vec);
					else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && any_i8<S>) return _mm_mask_storeu_epi8(p, mask, vec);
					else static_assert(always_false_v<S>);
				}

				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 32 && sizeof(S) < 4)
				static SIMD_Vector<S, N> eval(op_unpacklo, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					if constexpr (sizeof(SIMD_Vector<S, N>) > 64) return { unpacklo(a.lo(),b.lo()), unpacklo(a.hi(),b.hi()) };
					else if constexpr (any_i16<S>) return _mm512_unpacklo_epi16(a, b);
					else if constexpr (any_i8<S>) return _mm512_unpacklo_epi8(a, b);
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 32 && sizeof(S) < 4)
				static SIMD_Vector<S, N> eval(op_unpackhi, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					if constexpr (sizeof(SIMD_Vector<S, N>) > 64) return { unpackhi(a.lo(),b.lo()), unpackhi(a.hi(),b.hi()) };
					else if constexpr (any_i16<S>) return _mm512_unpackhi_epi16(a, b);
					else if constexpr (any_i8<S>) return _mm512_unpackhi_epi8(a, b);
					else static_assert(always_false_v<S>);
				}

				/*
				template<typename S, size_t N>
				requires (sizeof(SIMD_Vector<S, N>) >= (FS.has(AVX512_VL) ? 0 : 33) && sizeof(S) < 4)
				static typename SIMD_Vector<S,N>::MaskT eval(op_maskvec2uint, const SIMD_Vector<S, N>& v)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 64) return { vec2mask(v.lo()), vec2mask(v.hi()) };
					else if constexpr (zmm_sized<T> && sizeof(S) == 2) return _mm512_movepi16_mask(v);
					else if constexpr (zmm_sized<T> && sizeof(S) == 1) return _mm512_movepi8_mask(v);
					else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && sizeof(S) == 2) return _mm256_movepi16_mask(v);
					else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && sizeof(S) == 1) return _mm256_movepi8_mask(v);
					else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && sizeof(S) == 2) return _mm_movepi16_mask(v);
					else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && sizeof(S) == 1) return _mm_movepi8_mask(v);
					else static_assert(always_false_v<T>);
				}*/
			};
		}
	}
};