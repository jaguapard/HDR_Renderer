#pragma once
#include "shared.h"
namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		using namespace meta;
		struct ISA_AVX512_BW
		{
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_add>)
			static auto eval(const SIMD_Vector<S, N>& a)
			{
				using namespace meta;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return T{ abs(a.lo()), abs(a.hi()) };
				else if constexpr (zmm_sized<T> && is_i16<S>) return _mm512_abs_epi16(a);
				else if constexpr (zmm_sized<T> && is_i8<S>) return _mm512_abs_epi8(a);
				else return fail_ack_t{};
			}

			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_sub>)
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace meta;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return T{ sub(a.lo(), b.lo()), sub(a.hi(), b.hi()) };
				else if constexpr (zmm_sized<T> && any_i16<S>) return _mm512_sub_epi16(a, b);
				else if constexpr (zmm_sized<T> && any_i8<S>) return _mm512_sub_epi8(a, b);
				else return fail_ack_t{};
			}

			template<typename Op, typename S, size_t N>
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				requires (std::same_as<Op, op_mul>)
			{
				using namespace meta;
				using T = SIMD_Vector<S, N>;
				using canon_t = std::conditional_t<(std::is_signed_v<S>), int16_t, uint16_t>;
				if constexpr (sizeof(T) > 64) return T{ mul(a.lo(), b.lo()), mul(a.hi(), b.hi()) };
				else if constexpr (zmm_sized<T> && any_i16<S>) return _mm512_mullo_epi16(a, b);
				else if constexpr (zmm_sized<T> && any_i8<S>) return vcvt<S>(mul(vcvt<canon_t>(a), vcvt<canon_t>(b)));
				else return fail_ack_t{};
			}

			template<typename Op, typename S, size_t N, typename I>
				requires (meta::any_int<I>&& std::same_as<Op, op_shl>&& any_small_int<S>)
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& b)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return T{ shift_left(a.lo(),b.lo()), shift_left(a.hi(),b.hi()) };
				else if constexpr (any_i8<S>) return vcvt<S>(shift_left(vcvt<uint16_t>(a), b));
				else if constexpr (zmm_sized<T> && any_i16<S>) return _mm512_sllv_epi16(a, b);
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && any_i16<S>) return _mm256_sllv_epi16(a, b);
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && any_i16<S>) return _mm_sllv_epi16(a, b);
				else return fail_ack_t{};
			}
			template<typename Op, typename S, size_t N, typename I>
				requires (meta::any_int<I>&& std::same_as<Op, op_shr>&& any_small_int<S>)
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& b)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return T{ shift_right(a.lo(),b.lo()), shift_right(a.hi(),b.hi()) };
				else if constexpr (any_i8<S>) return vcvt<S>(shift_right(vcvt<uint16_t>(a), b));
				else if constexpr (!std::is_same_v<I, uint16_t>) return shift_right(a, vcvt<uint16_t>(b));
				else if constexpr (zmm_sized<T> && any_i16<S>) return _mm512_srlv_epi16(a, b);
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && any_i16<S>) return _mm256_srlv_epi16(a, b);
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && any_i16<S>) return _mm_srlv_epi16(a, b);
				else return fail_ack_t{};
			}
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_cmpeq>)
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace meta;
				using T = SIMD_Vector<S, N>;
				using M = mask_t<S, N>;
				if constexpr (sizeof(T) > 64) return M{ cmp_equal(a.lo(),b.lo()), cmp_equal(a.hi(),b.hi()) };
				else if constexpr (zmm_sized<T>)
				{
					if constexpr (is_i16<S>) return _mm512_cmpeq_epi16_mask(a, b);
					else if constexpr (is_u16<S>) return _mm512_cmpeq_epu16_mask(a, b);
					else if constexpr (is_i8<S>) return _mm512_cmpeq_epi8_mask(a, b);
					else if constexpr (is_u8<S>) return _mm512_cmpeq_epu8_mask(a, b);
					else return fail_ack_t{};
				}
				else if constexpr (FS.has(AVX512_VL))
				{
					if constexpr (ymm_sized<T>)
					{
						if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm256_cmpeq_epi16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm256_cmpeq_epu16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm256_cmpeq_epi8_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm256_cmpeq_epu8_mask(a, b);
						else return fail_ack_t{};
					}
					else if constexpr (xmm_sized<T>)
					{
						if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm_cmpeq_epi16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm_cmpeq_epu16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm_cmpeq_epi8_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm_cmpeq_epu8_mask(a, b);
						else return fail_ack_t{};
					}
					else return fail_ack_t{};
				}
				else return fail_ack_t{};
			}
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_cmpneq>)
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace meta;
				using T = SIMD_Vector<S, N>;
				using M = mask_t<S, N>;
				if constexpr (sizeof(T) > 64) return M{ cmp_not_equal(a.lo(),b.lo()), cmp_not_equal(a.hi(),b.hi()) };
				else if constexpr (zmm_sized<T>)
				{
					if constexpr (is_i16<S>) return _mm512_cmpneq_epi16_mask(a, b);
					else if constexpr (is_u16<S>) return _mm512_cmpneq_epu16_mask(a, b);
					else if constexpr (is_i8<S>) return _mm512_cmpneq_epi8_mask(a, b);
					else if constexpr (is_u8<S>) return _mm512_cmpneq_epu8_mask(a, b);
					else return fail_ack_t{};
				}
				else if constexpr (FS.has(AVX512_VL))
				{
					if constexpr (ymm_sized<T>)
					{
						if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm256_cmpneq_epi16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm256_cmpneq_epu16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm256_cmpneq_epi8_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm256_cmpneq_epu8_mask(a, b);
						else return fail_ack_t{};
					}
					else if constexpr (xmm_sized<T>)
					{
						if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm_cmpneq_epi16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm_cmpneq_epu16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm_cmpneq_epi8_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm_cmpneq_epu8_mask(a, b);
						else return fail_ack_t{};
					}
					else return fail_ack_t{};
				}
				else return fail_ack_t{};
			}

			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_cmplt>)
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace meta;
				using T = SIMD_Vector<S, N>;
				using M = mask_t<S, N>;
				if constexpr (sizeof(T) > 64) return M{ cmp_less(a.lo(),b.lo()), cmp_less(a.hi(),b.hi()) };
				else if constexpr (zmm_sized<T>)
				{
					if constexpr (is_i16<S>) return _mm512_cmplt_epi16_mask(a, b);
					else if constexpr (is_u16<S>) return _mm512_cmplt_epu16_mask(a, b);
					else if constexpr (is_i8<S>) return _mm512_cmplt_epi8_mask(a, b);
					else if constexpr (is_u8<S>) return _mm512_cmplt_epu8_mask(a, b);
					else return fail_ack_t{};
				}
				else if constexpr (FS.has(AVX512_VL))
				{
					if constexpr (ymm_sized<T>)
					{
						if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm256_cmplt_epi16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm256_cmplt_epu16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm256_cmplt_epi8_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm256_cmplt_epu8_mask(a, b);
						else return fail_ack_t{};
					}
					else if constexpr (xmm_sized<T>)
					{
						if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm_cmplt_epi16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm_cmplt_epu16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm_cmplt_epi8_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm_cmplt_epu8_mask(a, b);
						else return fail_ack_t{};
					}
					else return fail_ack_t{};
				}
				else return fail_ack_t{};
			}

			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_cmple>)
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace meta;
				using T = SIMD_Vector<S, N>;
				using M = mask_t<S, N>;
				if constexpr (sizeof(T) > 64) return M{ cmp_less_or_equal(a.lo(),b.lo()), cmp_less_or_equal(a.hi(),b.hi()) };
				else if constexpr (zmm_sized<T>)
				{
					if constexpr (is_i16<S>) return _mm512_cmple_epi16_mask(a, b);
					else if constexpr (is_u16<S>) return _mm512_cmple_epu16_mask(a, b);
					else if constexpr (is_i8<S>) return _mm512_cmple_epi8_mask(a, b);
					else if constexpr (is_u8<S>) return _mm512_cmple_epu8_mask(a, b);
					else return fail_ack_t{};
				}
				else if constexpr (FS.has(AVX512_VL))
				{
					if constexpr (ymm_sized<T>)
					{
						if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm256_cmple_epi16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm256_cmple_epu16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm256_cmple_epi8_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm256_cmple_epu8_mask(a, b);
						else return fail_ack_t{};
					}
					else if constexpr (xmm_sized<T>)
					{
						if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm_cmple_epi16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm_cmple_epu16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm_cmple_epi8_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm_cmple_epu8_mask(a, b);
						else return fail_ack_t{};
					}
					else return fail_ack_t{};
				}
				else return fail_ack_t{};
			}

			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_cmpgt>)
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace meta;
				using T = SIMD_Vector<S, N>;
				using M = mask_t<S, N>;
				if constexpr (sizeof(T) > 64) return M{ cmp_greater(a.lo(),b.lo()), cmp_greater(a.hi(),b.hi()) };
				else if constexpr (zmm_sized<T>)
				{
					if constexpr (is_i16<S>) return _mm512_cmpgt_epi16_mask(a, b);
					else if constexpr (is_u16<S>) return _mm512_cmpgt_epu16_mask(a, b);
					else if constexpr (is_i8<S>) return _mm512_cmpgt_epi8_mask(a, b);
					else if constexpr (is_u8<S>) return _mm512_cmpgt_epu8_mask(a, b);
					else return fail_ack_t{};
				}
				else if constexpr (FS.has(AVX512_VL))
				{
					if constexpr (ymm_sized<T>)
					{
						if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm256_cmpgt_epi16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm256_cmpgt_epu16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm256_cmpgt_epi8_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm256_cmpgt_epu8_mask(a, b);
						else return fail_ack_t{};
					}
					else if constexpr (xmm_sized<T>)
					{
						if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm_cmpgt_epi16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm_cmpgt_epu16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm_cmpgt_epi8_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm_cmpgt_epu8_mask(a, b);
						else return fail_ack_t{};
					}
					else return fail_ack_t{};
				}
				else return fail_ack_t{};
			}

			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_cmpge>)
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace meta;
				using T = SIMD_Vector<S, N>;
				using M = mask_t<S, N>;
				if constexpr (sizeof(T) > 64) return M{ cmp_greater_or_equal(a.lo(),b.lo()), cmp_greater_or_equal(a.hi(),b.hi()) };
				else if constexpr (zmm_sized<T>)
				{
					if constexpr (is_i16<S>) return _mm512_cmpge_epi16_mask(a, b);
					else if constexpr (is_u16<S>) return _mm512_cmpge_epu16_mask(a, b);
					else if constexpr (is_i8<S>) return _mm512_cmpge_epi8_mask(a, b);
					else if constexpr (is_u8<S>) return _mm512_cmpge_epu8_mask(a, b);
					else return fail_ack_t{};
				}
				else if constexpr (FS.has(AVX512_VL))
				{
					if constexpr (ymm_sized<T>)
					{
						if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm256_cmpge_epi16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm256_cmpge_epu16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm256_cmpge_epi8_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm256_cmpge_epu8_mask(a, b);
						else return fail_ack_t{};
					}
					else if constexpr (xmm_sized<T>)
					{
						if constexpr (FS.has(AVX512_BW) && is_i16<S>) return _mm_cmpge_epi16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u16<S>) return _mm_cmpge_epu16_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_i8<S>) return _mm_cmpge_epi8_mask(a, b);
						else if constexpr (FS.has(AVX512_BW) && is_u8<S>) return _mm_cmpge_epu8_mask(a, b);
						else return fail_ack_t{};
					}
					else return fail_ack_t{};
				}
				else return fail_ack_t{};
			}

			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_mask_mov>)
			static auto eval(const SIMD_Vector<S, N>& ifBitClear, const typename SIMD_Vector<S, N>::MaskT& mask, const SIMD_Vector<S, N>& ifBitSet)
			{
				using namespace meta;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return T{ mask_mov(ifBitClear.lo(), mask.lo(), ifBitSet.lo()), mask_mov(ifBitClear.hi(), mask.hi(), ifBitSet.hi()) };
				else if constexpr (zmm_sized<T> && sizeof(S) == 2) return _mm512_mask_mov_epi16(vreinterpret<__m512i>(ifBitClear), mask, vreinterpret<__m512i>(ifBitSet));
				else if constexpr (zmm_sized<T> && sizeof(S) == 1) return _mm512_mask_mov_epi8(vreinterpret<__m512i>(ifBitClear), mask, vreinterpret<__m512i>(ifBitSet));
				else if constexpr (ymm_sized<T> && FS.has(AVX512_VL) && sizeof(S) == 2) return _mm256_mask_mov_epi16(vreinterpret<__m256i>(ifBitClear), mask, vreinterpret<__m256i>(ifBitSet));
				else if constexpr (ymm_sized<T> && FS.has(AVX512_VL) && sizeof(S) == 1) return _mm256_mask_mov_epi8(vreinterpret<__m256i>(ifBitClear), mask, vreinterpret<__m256i>(ifBitSet));
				else if constexpr (xmm_sized<T> && FS.has(AVX512_VL) && sizeof(S) == 2) return _mm_mask_mov_epi16(vreinterpret<__m128i>(ifBitClear), mask, vreinterpret<__m128i>(ifBitSet));
				else if constexpr (xmm_sized<T> && FS.has(AVX512_VL) && sizeof(S) == 1) return _mm_mask_mov_epi8(vreinterpret<__m128i>(ifBitClear), mask, vreinterpret<__m128i>(ifBitSet));
				else return fail_ack_t{};
			}

			template<typename Op, size_t N, typename From>
				requires (IsCvtOp<Op>)
			static auto eval(const SIMD_Vector<From, N>& a)
			{
				using namespace meta;
				using To = typename Op::cvt_to_t;
				using TV = SIMD_Vector<To, N>;
				using FV = SIMD_Vector<From, N>;
				constexpr size_t MaxSize = std::max(sizeof(TV), sizeof(FV));

				if constexpr (MaxSize > 64) return TV{ vcvt<To>(a.lo()), vcvt<To>(a.hi()) };
				else if constexpr (is_zmm_size(MaxSize) && any_i16<From> && any_i8<To>) return _mm512_cvtepi16_epi8(a);
				else if constexpr (is_zmm_size(MaxSize) && is_i8<From> && any_i16<To>) return _mm512_cvtepi8_epi16(a);
				else if constexpr (is_zmm_size(MaxSize) && is_u8<From> && any_i16<To>) return _mm512_cvtepu8_epi16(a);
				else if constexpr (FS.has(AVX512_VL) && is_ymm_size(MaxSize) && any_i16<From> && any_i8<To>) return _mm256_cvtepi16_epi8(a);
				else if constexpr (FS.has(AVX512_VL) && is_xmm_size(MaxSize) && any_i16<From> && any_i8<To>) return _mm_cvtepi16_epi8(a);
				else return fail_ack_t{};
			}

			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_min>)
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace meta;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return T{ min(a.lo(), b.lo()), min(a.hi(),b.hi()) };
				else if constexpr (zmm_sized<T> && is_i16<S>) return _mm512_min_epi16(a, b);
				else if constexpr (zmm_sized<T> && is_u16<S>) return _mm512_min_epu16(a, b);
				else if constexpr (zmm_sized<T> && is_i8<S>) return _mm512_min_epi8(a, b);
				else if constexpr (zmm_sized<T> && is_u8<S>) return _mm512_min_epu8(a, b);
				else return fail_ack_t{};
			}
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_max>)
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace meta;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return T{ max(a.lo(), b.lo()), max(a.hi(),b.hi()) };
				else if constexpr (zmm_sized<T> && is_i16<S>) return _mm512_max_epi16(a, b);
				else if constexpr (zmm_sized<T> && is_u16<S>) return _mm512_max_epu16(a, b);
				else if constexpr (zmm_sized<T> && is_i8<S>) return _mm512_max_epi8(a, b);
				else if constexpr (zmm_sized<T> && is_u8<S>) return _mm512_max_epu8(a, b);
				else return fail_ack_t{};
			}
			template<typename Op, typename S, size_t N, typename I>
				requires (meta::any_int<I>&& std::same_as<Op, op_permx> && sizeof(S) < 4)
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& ind)
			{
				using namespace meta;
				using canon_t = typename ScalarTraits<S>::UintT;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(I) != sizeof(S)) return permx(a, vcvt<canon_t>(ind));
				else if constexpr (sizeof(S) == 1)
				{
					//1 byte permute can be emulated by zero-extending the values to 16 bits
					//permuting as 16 bits, then narrowing back, removing redundant zeros
					//Note that values of the items must not change, since permute is data-movement operation
					auto a16 = vcvt<uint16_t>(vcast<SIMD_Vector<uint8_t, N>>(a)); //reinterpret a as 8-bit ints, zero-extend
					auto p = vcast<SIMD_Vector<uint16_t, N>>(permx(a16, ind)); //permute as 16 bit ints
					auto ret8 = vcvt<uint8_t>(p); //narrow back to 8 bits
					return vcast<SIMD_Vector<S, N>>(ret8); //return reinterpreted back to input type
				}
				//TODO: add > 64 byte permutex!
				else if constexpr (zmm_sized<T> && any_i16<S>) return _mm512_permutexvar_epi16(ind, a);
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && any_i16<S>) return _mm256_permutexvar_epi16(ind, a);
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && any_i16<S>) return _mm_permutexvar_epi16(ind, a);
				else return fail_ack_t{};
			}
			template<typename Op, typename S, size_t N, typename I>
				requires (std::same_as<Op, op_permx2>&& meta::any_int<I>)
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b, const SIMD_Vector<I, N>& ind)
			{
				using namespace meta;
				using canon_t = typename ScalarTraits<S>::UintT;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(I) != sizeof(S)) return permx2(a, b, vcvt<canon_t>(ind));
				else if constexpr (any_i8<S>) return vcvt<S>(permx2(vcvt<uint16_t>(a), vcvt<uint16_t>(b), ind));
				//TODO: add > 64 byte permutex2!
				else if constexpr (zmm_sized<T> && any_i16<S>) return _mm512_permutex2var_epi16(a, ind, b);
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && any_i16<S>) return _mm256_permutex2var_epi16(a, ind, b);
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && any_i16<S>) return _mm_permutex2var_epi16(a, ind, b);
				else return fail_ack_t{};
			}

			template<typename Op>
				requires (meta::IsLoadOp<Op>)
			static auto eval(const void* p, const typename SIMD_Vector<typename Op::S, Op::N>::MaskT& mask, const SIMD_Vector<typename Op::S, Op::N>& src)
			{
				using namespace meta;
				using S = typename Op::S;
				constexpr size_t N = Op::N;
				using T = SIMD_Vector<S, N>;
				const S* sp = (const S*)p;
				if constexpr (sizeof(T) > 64) return T{ load<S,N / 2>(sp,mask.lo(), src.lo()), load<S,N / 2>(sp + N / 2, mask.hi(), src.hi()) };
				else if constexpr (zmm_sized<T> && any_i16<S>) return _mm512_mask_loadu_epi16(src, mask, p);
				else if constexpr (zmm_sized<T> && any_i8<S>) return _mm512_mask_loadu_epi8(src, mask, p);
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && any_i16<S>) return _mm256_mask_loadu_epi16(src, mask, p);
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && any_i8<S>) return _mm256_mask_loadu_epi8(src, mask, p);
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && any_i16<S>) return _mm_mask_loadu_epi16(src, mask, p);
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && any_i8<S>) return _mm_mask_loadu_epi8(src, mask, p);
				else return fail_ack_t{};
			}

			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_store>)
			static auto eval(SIMD_Vector<S, N> vec, void* p, const typename SIMD_Vector<S, N>::MaskT& mask)
			{
				using namespace meta;
				using T = SIMD_Vector<S, N>;
				SIMD_Vector<S, N> ret;
				const S* sp = (const S*)p;
				if constexpr (sizeof(T) > 64)
				{
					store(vec.lo(), p, mask.lo());
					store(vec.hi(), sp + N / 2, mask.hi());
					return success_ack_t{};
				}
				else if constexpr (zmm_sized<T> && any_i16<S>) { _mm512_mask_storeu_epi16(p, mask, vec); return success_ack_t{}; }
				else if constexpr (zmm_sized<T> && any_i8<S>) { _mm512_mask_storeu_epi8(p, mask, vec); return success_ack_t{}; }
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && any_i16<S>) { _mm256_mask_storeu_epi16(p, mask, vec); return success_ack_t{}; }
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && any_i8<S>) { _mm256_mask_storeu_epi8(p, mask, vec); return success_ack_t{}; }
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && any_i16<S>) { _mm_mask_storeu_epi16(p, mask, vec); return success_ack_t{}; }
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && any_i8<S>) { _mm_mask_storeu_epi8(p, mask, vec); return success_ack_t{}; }
				else return fail_ack_t{};
			}

			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_unpacklo>)
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return T{ unpacklo(a.lo(),b.lo()), unpacklo(a.hi(),b.hi()) };
				else if constexpr (zmm_sized<T> && any_i16<S>) return _mm512_unpacklo_epi16(a, b);
				else if constexpr (any_i8<S>) return _mm512_unpacklo_epi8(a, b);
				else return fail_ack_t{};
			}
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_unpackhi>)
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return T{ unpackhi(a.lo(),b.lo()), unpackhi(a.hi(),b.hi()) };
				else if constexpr (zmm_sized<T> && any_i16<S>) return _mm512_unpackhi_epi16(a, b);
				else if constexpr (zmm_sized<T> && any_i8<S>) return _mm512_unpackhi_epi8(a, b);
				else return fail_ack_t{};
			}
		};
	}
}