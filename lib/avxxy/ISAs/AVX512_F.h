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
					if constexpr (zmm_sized<T> && is_f64<S>) return _mm512_add_pd(a, b);
					else if constexpr (zmm_sized<T> && is_f32<S>) return _mm512_add_ps(a, b);
					else if constexpr (zmm_sized<T> && any_i64<S>) return _mm512_add_epi64(a, b);
					else if constexpr (zmm_sized<T> && any_i32<S>) return _mm512_add_epi32(a, b);
					else return fail_ack_t{};
				}
				//TODO: check these!
				//else if constexpr (!FS.has(Feature::AVX2) && std::is_signed_v<S>) return vcvt<S>(add(vcvt<int32_t>(a), vcvt<int32_t>(b)));
				//else if constexpr (!FS.has(Feature::AVX2) && std::is_unsigned_v<S>) return vcvt<S>(add(vcvt<uint32_t>(a), vcvt<uint32_t>(b)));
				else return fail_ack_t{};
			}
			template<typename Op, typename S, size_t N>
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				requires (std::same_as<Op, op_sub>)
			{
				using namespace meta;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return T{ sub(a.lo(), b.lo()), sub(a.hi(), b.hi()) };
				else if constexpr (zmm_sized<T> && is_f64<S>) return _mm512_sub_pd(a, b);
				else if constexpr (zmm_sized<T> && is_f32<S>) return _mm512_sub_ps(a, b);
				else if constexpr (zmm_sized<T> && any_i64<S>) return _mm512_sub_epi64(a, b);
				else if constexpr (zmm_sized<T> && any_i32<S>) return _mm512_sub_epi32(a, b);
				//TODO: check these!
				//else if constexpr (zmm_sized<T> && !FS.has(Feature::AVX2) && std::is_signed_v<S>) return vcvt<S>(sub(vcvt<int32_t>(a), vcvt<int32_t>(b)));
				//else if constexpr (zmm_sized<T> && !FS.has(Feature::AVX2) && std::is_unsigned_v<S>) return vcvt<S>(sub(vcvt<uint32_t>(a), vcvt<uint32_t>(b)));
				else return fail_ack_t{};
			}

			template<typename Op, typename S, size_t N>
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				requires (std::same_as<Op, op_mul>)
			{
				using namespace meta;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return T{ mul(a.lo(), b.lo()), mul(a.hi(), b.hi()) };
				else if constexpr (zmm_sized<T> && is_f64<S>) return _mm512_mul_pd(a, b);
				else if constexpr (zmm_sized<T> && is_f32<S>) return _mm512_mul_ps(a, b);
				else if constexpr (zmm_sized<T> && any_i64<S>) return _mm512_mullox_epi64(a, b);
				else if constexpr (zmm_sized<T> && any_i32<S>) return _mm512_mullo_epi32(a, b);
				//TODO: check these!
				//else if constexpr (!FS.has(Feature::AVX2) && std::is_signed_v<S>) return vcvt<S>(mul(vcvt<int32_t>(a), vcvt<int32_t>(b)));
				//else if constexpr (!FS.has(Feature::AVX2) && std::is_unsigned_v<S>) return vcvt<S>(mul(vcvt<uint32_t>(a), vcvt<uint32_t>(b)));
				else return fail_ack_t{};
			}

			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_div>)
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace meta;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return T{ div(a.lo(), b.lo()), div(a.hi(),b.hi()) };
				else if constexpr (zmm_sized<T> && is_f64<S>) return _mm512_div_pd(a, b);
				else if constexpr (zmm_sized<T> && is_f32<S>) return _mm512_div_ps(a, b);
				else if constexpr (zmm_sized<T> && any_i32<S>) return vcvt<S>(div(vcvt<double>(a), vcvt<double>(b))); //emulate 32 bit integer division via double precision division
				else if constexpr (zmm_sized<T> && (any_i16<S> || any_i8<S>)) return vcvt<S>(div(vcvt<float>(a), vcvt<float>(b))); //emulate small integer division via single precision division
				else return fail_ack_t{};
			}

			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_abs>)
			static auto eval(const SIMD_Vector<S, N>& a)
			{
				using namespace meta;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return T{ abs(a.lo()), abs(a.hi()) };
				else if constexpr (zmm_sized<T> && is_f64<S>) return _mm512_abs_pd(a);
				else if constexpr (zmm_sized<T> && is_f32<S>) return _mm512_abs_ps(a);
				else if constexpr (zmm_sized<T> && is_i64<S>) return _mm512_abs_epi64(a);
				else if constexpr (zmm_sized<T> && is_i32<S>) return _mm512_abs_epi32(a);
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && is_i64<S>) return _mm256_abs_epi64(a);
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && is_i64<S>) return _mm_abs_epi64(a);
				else return fail_ack_t{};
			}

			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_or>)
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return T{ logic_or(a.lo(),b.lo()), logic_or(a.hi(),b.hi()) };
				else if constexpr (zmm_sized<T>) return _mm512_or_si512(vreinterpret<__m512i>(a), vreinterpret<__m512i>(b));
				else return fail_ack_t{};
			}
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_and>)
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return T{ logic_and(a.lo(),b.lo()), logic_and(a.hi(),b.hi()) };
				else if constexpr (zmm_sized<T>) return _mm512_and_si512(vreinterpret<__m512i>(a), vreinterpret<__m512i>(b));
				else return fail_ack_t{};
			}
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_xor>)
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return T{ logic_xor(a.lo(),b.lo()), logic_xor(a.hi(),b.hi()) };
				else if constexpr (zmm_sized<T>) return _mm512_xor_si512(vreinterpret<__m512i>(a), vreinterpret<__m512i>(b));
				else return fail_ack_t{};
			}
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_not>)
			static auto eval(const SIMD_Vector<S, N>& a)
			{
				if constexpr (sizeof(SIMD_Vector<S, N>) > 32)
				{
					using U = typename ScalarTraits<S>::UintT;
					S val = std::bit_cast<S>(~U(0));
					return logic_xor(a, val);
				}
				else return fail_ack_t{};
			}

			template<typename Op, typename S, size_t N, typename I>
				requires (meta::any_int<S>&& meta::any_int<I>&& std::same_as<Op, op_shl>)
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& b)
			{
				using canon_t = typename ScalarTraits<S>::UintT;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return T{ shift_left(a.lo(),b.lo()), shift_left(a.hi(),b.hi()) };
				else if constexpr (zmm_sized<T>)
				{
					if constexpr (!std::is_same_v<I, canon_t>) return shift_left(a, vcvt<canon_t>(b));
					else if constexpr (any_i64<S>) return _mm512_sllv_epi64(a, b);
					else if constexpr (any_i32<S>) return _mm512_sllv_epi32(a, b);
					else return fail_ack_t{};
				}
				else return fail_ack_t{};
				//else return vcvt<S>(shift_left(vcvt<uint32_t>(a), vcvt<uint32_t>(b))); //emulate shift by 32 bit shift for small types
			}
			template<typename Op, typename S, size_t N, typename I>
				requires (meta::any_int<S>&& meta::any_int<I>&& std::same_as<Op, op_shr>)
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& b)
			{
				using canon_t = typename ScalarTraits<S>::UintT;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return T{ shift_right(a.lo(),b.lo()), shift_right(a.hi(),b.hi()) };
				else if constexpr (zmm_sized<T>)
				{
					if constexpr (!std::is_same_v<I, canon_t>) return shift_right(a, vcvt<canon_t>(b));
					else if constexpr (any_i64<S>) return _mm512_srlv_epi64(a, b);
					else if constexpr (any_i32<S>) return _mm512_srlv_epi32(a, b);
					else return fail_ack_t{};
				}
				else return fail_ack_t{};
				//else return vcvt<S>(shift_right(vcvt<uint32_t>(a), vcvt<uint32_t>(b))); //emulate shift by 32 bit shift for small types
			}

			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_min>)
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace meta;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return T{ min(a.lo(), b.lo()), min(a.hi(),b.hi()) };
				else if constexpr (zmm_sized<T> && is_f64<S>) return _mm512_min_pd(a, b);
				else if constexpr (zmm_sized<T> && is_f32<S>) return _mm512_min_ps(a, b);
				else if constexpr (zmm_sized<T> && is_i64<S>) return _mm512_min_epi64(a, b);
				else if constexpr (zmm_sized<T> && is_u64<S>) return _mm512_min_epu64(a, b);
				else if constexpr (zmm_sized<T> && is_i32<S>) return _mm512_min_epi32(a, b);
				else if constexpr (zmm_sized<T> && is_u32<S>) return _mm512_min_epu32(a, b);
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && is_i64<S>) return _mm256_min_epi64(a, b);
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && is_u64<S>) return _mm256_min_epu64(a, b);
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && is_i64<S>) return _mm_min_epi64(a, b);
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && is_u64<S>) return _mm_min_epu64(a, b);
				else return fail_ack_t{};
			}
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_max>)
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using namespace meta;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return T{ max(a.lo(), b.lo()), max(a.hi(),b.hi()) };
				else if constexpr (zmm_sized<T> && is_f64<S>) return _mm512_max_pd(a, b);
				else if constexpr (zmm_sized<T> && is_f32<S>) return _mm512_max_ps(a, b);
				else if constexpr (zmm_sized<T> && is_i64<S>) return _mm512_max_epi64(a, b);
				else if constexpr (zmm_sized<T> && is_u64<S>) return _mm512_max_epu64(a, b);
				else if constexpr (zmm_sized<T> && is_i32<S>) return _mm512_max_epi32(a, b);
				else if constexpr (zmm_sized<T> && is_u32<S>) return _mm512_max_epu32(a, b);
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && is_i64<S>) return _mm256_max_epi64(a, b);
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && is_u64<S>) return _mm256_max_epu64(a, b);
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && is_i64<S>) return _mm_max_epi64(a, b);
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && is_u64<S>) return _mm_max_epu64(a, b);
				else return fail_ack_t{};
			}

			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_mask_mov>)
			static auto eval(const SIMD_Vector<S, N>& ifBitClear, const typename SIMD_Vector<S, N>::MaskT& mask, const SIMD_Vector<S, N>& ifBitSet)
			{
				using namespace meta;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return T{ mask_mov(ifBitClear.lo(), mask.lo(), ifBitSet.lo()), mask_mov(ifBitClear.hi(), mask.hi(), ifBitSet.hi()) };
				else if constexpr (zmm_sized<T>)
				{
					if constexpr (is_f64<S>) return _mm512_mask_mov_pd(ifBitClear, mask, ifBitSet);
					else if constexpr (is_f32<S>) return _mm512_mask_mov_ps(ifBitClear, mask, ifBitSet);
					else if constexpr (any_i64<S>) return _mm512_mask_mov_epi64(ifBitClear, mask, ifBitSet);
					else if constexpr (any_i32<S>) return _mm512_mask_mov_epi32(ifBitClear, mask, ifBitSet);
					else return fail_ack_t{};
				}
				else if constexpr (FS.has(AVX512_VL))
				{
					if constexpr (ymm_sized<T> && is_f64<S>) return _mm256_mask_mov_pd(ifBitClear, mask, ifBitSet);
					else if constexpr (ymm_sized<T> && is_f32<S>) return _mm256_mask_mov_ps(ifBitClear, mask, ifBitSet);
					else if constexpr (ymm_sized<T> && any_i64<S>) return _mm256_mask_mov_epi64(ifBitClear, mask, ifBitSet);
					else if constexpr (ymm_sized<T> && any_i32<S>) return _mm256_mask_mov_epi32(ifBitClear, mask, ifBitSet);

					else if constexpr (xmm_sized<T> && is_f64<S>) return _mm_mask_mov_pd(ifBitClear, mask, ifBitSet);
					else if constexpr (xmm_sized<T> && is_f32<S>) return _mm_mask_mov_ps(ifBitClear, mask, ifBitSet);
					else if constexpr (xmm_sized<T> && any_i64<S>) return _mm_mask_mov_epi64(ifBitClear, mask, ifBitSet);
					else if constexpr (xmm_sized<T> && any_i32<S>) return _mm_mask_mov_epi32(ifBitClear, mask, ifBitSet);
					else return fail_ack_t{};
				}
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
					if constexpr (is_f64<S>) return _mm512_cmp_pd_mask(a, b, _CMP_EQ_OQ);
					else if constexpr (is_f32<S>) return _mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ);
					else if constexpr (is_i64<S>) return _mm512_cmpeq_epi64_mask(a, b);
					else if constexpr (is_u64<S>) return _mm512_cmpeq_epu64_mask(a, b);
					else if constexpr (is_i32<S>) return _mm512_cmpeq_epi32_mask(a, b);
					else if constexpr (is_u32<S>) return _mm512_cmpeq_epu32_mask(a, b);
					else return fail_ack_t{};
				}
				else if constexpr (FS.has(AVX512_VL))
				{
					if constexpr (ymm_sized<T>)
					{
						if constexpr (is_f64<S>) return _mm256_cmp_pd_mask(a, b, _CMP_EQ_OQ);
						else if constexpr (is_f32<S>) return _mm256_cmp_ps_mask(a, b, _CMP_EQ_OQ);
						else if constexpr (is_i64<S>) return _mm256_cmpeq_epi64_mask(a, b);
						else if constexpr (is_u64<S>) return _mm256_cmpeq_epu64_mask(a, b);
						else if constexpr (is_i32<S>) return _mm256_cmpeq_epi32_mask(a, b);
						else if constexpr (is_u32<S>) return _mm256_cmpeq_epu32_mask(a, b);
						else return fail_ack_t{};
					}
					else if constexpr (xmm_sized<T>)
					{
						if constexpr (is_f64<S>) return _mm_cmp_pd_mask(a, b, _CMP_EQ_OQ);
						else if constexpr (is_f32<S>) return _mm_cmp_ps_mask(a, b, _CMP_EQ_OQ);
						else if constexpr (is_i64<S>) return _mm_cmpeq_epi64_mask(a, b);
						else if constexpr (is_u64<S>) return _mm_cmpeq_epu64_mask(a, b);
						else if constexpr (is_i32<S>) return _mm_cmpeq_epi32_mask(a, b);
						else if constexpr (is_u32<S>) return _mm_cmpeq_epu32_mask(a, b);
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
					if constexpr (is_f64<S>) return _mm512_cmp_pd_mask(a, b, _CMP_NEQ_OQ);
					else if constexpr (is_f32<S>) return _mm512_cmp_ps_mask(a, b, _CMP_NEQ_OQ);
					else if constexpr (is_i64<S>) return _mm512_cmpneq_epi64_mask(a, b);
					else if constexpr (is_u64<S>) return _mm512_cmpneq_epu64_mask(a, b);
					else if constexpr (is_i32<S>) return _mm512_cmpneq_epi32_mask(a, b);
					else if constexpr (is_u32<S>) return _mm512_cmpneq_epu32_mask(a, b);
					else return fail_ack_t{};
				}
				else if constexpr (FS.has(AVX512_VL))
				{
					if constexpr (ymm_sized<T>)
					{
						if constexpr (is_f64<S>) return _mm256_cmp_pd_mask(a, b, _CMP_NEQ_OQ);
						else if constexpr (is_f32<S>) return _mm256_cmp_ps_mask(a, b, _CMP_NEQ_OQ);
						else if constexpr (is_i64<S>) return _mm256_cmpneq_epi64_mask(a, b);
						else if constexpr (is_u64<S>) return _mm256_cmpneq_epu64_mask(a, b);
						else if constexpr (is_i32<S>) return _mm256_cmpneq_epi32_mask(a, b);
						else if constexpr (is_u32<S>) return _mm256_cmpneq_epu32_mask(a, b);
						else return fail_ack_t{};
					}
					else if constexpr (xmm_sized<T>)
					{
						if constexpr (is_f64<S>) return _mm_cmp_pd_mask(a, b, _CMP_NEQ_OQ);
						else if constexpr (is_f32<S>) return _mm_cmp_ps_mask(a, b, _CMP_NEQ_OQ);
						else if constexpr (is_i64<S>) return _mm_cmpneq_epi64_mask(a, b);
						else if constexpr (is_u64<S>) return _mm_cmpneq_epu64_mask(a, b);
						else if constexpr (is_i32<S>) return _mm_cmpneq_epi32_mask(a, b);
						else if constexpr (is_u32<S>) return _mm_cmpneq_epu32_mask(a, b);
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
					if constexpr (is_f64<S>) return _mm512_cmp_pd_mask(a, b, _CMP_GT_OQ);
					else if constexpr (is_f32<S>) return _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ);
					else if constexpr (is_i64<S>) return _mm512_cmpgt_epi64_mask(a, b);
					else if constexpr (is_u64<S>) return _mm512_cmpgt_epu64_mask(a, b);
					else if constexpr (is_i32<S>) return _mm512_cmpgt_epi32_mask(a, b);
					else if constexpr (is_u32<S>) return _mm512_cmpgt_epu32_mask(a, b);
					else return fail_ack_t{};
				}
				else if constexpr (FS.has(AVX512_VL))
				{
					if constexpr (ymm_sized<T>)
					{
						if constexpr (is_f64<S>) return _mm256_cmp_pd_mask(a, b, _CMP_GT_OQ);
						else if constexpr (is_f32<S>) return _mm256_cmp_ps_mask(a, b, _CMP_GT_OQ);
						else if constexpr (is_i64<S>) return _mm256_cmpgt_epi64_mask(a, b);
						else if constexpr (is_u64<S>) return _mm256_cmpgt_epu64_mask(a, b);
						else if constexpr (is_i32<S>) return _mm256_cmpgt_epi32_mask(a, b);
						else if constexpr (is_u32<S>) return _mm256_cmpgt_epu32_mask(a, b);
						else return fail_ack_t{};
					}
					else if constexpr (xmm_sized<T>)
					{
						if constexpr (is_f64<S>) return _mm_cmp_pd_mask(a, b, _CMP_GT_OQ);
						else if constexpr (is_f32<S>) return _mm_cmp_ps_mask(a, b, _CMP_GT_OQ);
						else if constexpr (is_i64<S>) return _mm_cmpgt_epi64_mask(a, b);
						else if constexpr (is_u64<S>) return _mm_cmpgt_epu64_mask(a, b);
						else if constexpr (is_i32<S>) return _mm_cmpgt_epi32_mask(a, b);
						else if constexpr (is_u32<S>) return _mm_cmpgt_epu32_mask(a, b);
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
					if constexpr (is_f64<S>) return _mm512_cmp_pd_mask(a, b, _CMP_GE_OQ);
					else if constexpr (is_f32<S>) return _mm512_cmp_ps_mask(a, b, _CMP_GE_OQ);
					else if constexpr (is_i64<S>) return _mm512_cmpge_epi64_mask(a, b);
					else if constexpr (is_u64<S>) return _mm512_cmpge_epu64_mask(a, b);
					else if constexpr (is_i32<S>) return _mm512_cmpge_epi32_mask(a, b);
					else if constexpr (is_u32<S>) return _mm512_cmpge_epu32_mask(a, b);
					else return fail_ack_t{};
				}
				else if constexpr (FS.has(AVX512_VL))
				{
					if constexpr (ymm_sized<T>)
					{
						if constexpr (is_f64<S>) return _mm256_cmp_pd_mask(a, b, _CMP_GE_OQ);
						else if constexpr (is_f32<S>) return _mm256_cmp_ps_mask(a, b, _CMP_GE_OQ);
						else if constexpr (is_i64<S>) return _mm256_cmpge_epi64_mask(a, b);
						else if constexpr (is_u64<S>) return _mm256_cmpge_epu64_mask(a, b);
						else if constexpr (is_i32<S>) return _mm256_cmpge_epi32_mask(a, b);
						else if constexpr (is_u32<S>) return _mm256_cmpge_epu32_mask(a, b);
						else return fail_ack_t{};
					}
					else if constexpr (xmm_sized<T>)
					{
						if constexpr (is_f64<S>) return _mm_cmp_pd_mask(a, b, _CMP_GE_OQ);
						else if constexpr (is_f32<S>) return _mm_cmp_ps_mask(a, b, _CMP_GE_OQ);
						else if constexpr (is_i64<S>) return _mm_cmpge_epi64_mask(a, b);
						else if constexpr (is_u64<S>) return _mm_cmpge_epu64_mask(a, b);
						else if constexpr (is_i32<S>) return _mm_cmpge_epi32_mask(a, b);
						else if constexpr (is_u32<S>) return _mm_cmpge_epu32_mask(a, b);
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
					if constexpr (is_f64<S>) return _mm512_cmp_pd_mask(a, b, _CMP_LT_OQ);
					else if constexpr (is_f32<S>) return _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ);
					else if constexpr (is_i64<S>) return _mm512_cmplt_epi64_mask(a, b);
					else if constexpr (is_u64<S>) return _mm512_cmplt_epu64_mask(a, b);
					else if constexpr (is_i32<S>) return _mm512_cmplt_epi32_mask(a, b);
					else if constexpr (is_u32<S>) return _mm512_cmplt_epu32_mask(a, b);
					else return fail_ack_t{};
				}
				else if constexpr (FS.has(AVX512_VL))
				{
					if constexpr (ymm_sized<T>)
					{
						if constexpr (is_f64<S>) return _mm256_cmp_pd_mask(a, b, _CMP_LT_OQ);
						else if constexpr (is_f32<S>) return _mm256_cmp_ps_mask(a, b, _CMP_LT_OQ);
						else if constexpr (is_i64<S>) return _mm256_cmplt_epi64_mask(a, b);
						else if constexpr (is_u64<S>) return _mm256_cmplt_epu64_mask(a, b);
						else if constexpr (is_i32<S>) return _mm256_cmplt_epi32_mask(a, b);
						else if constexpr (is_u32<S>) return _mm256_cmplt_epu32_mask(a, b);
						else return fail_ack_t{};
					}
					else if constexpr (xmm_sized<T>)
					{
						if constexpr (is_f64<S>) return _mm_cmp_pd_mask(a, b, _CMP_LT_OQ);
						else if constexpr (is_f32<S>) return _mm_cmp_ps_mask(a, b, _CMP_LT_OQ);
						else if constexpr (is_i64<S>) return _mm_cmplt_epi64_mask(a, b);
						else if constexpr (is_u64<S>) return _mm_cmplt_epu64_mask(a, b);
						else if constexpr (is_i32<S>) return _mm_cmplt_epi32_mask(a, b);
						else if constexpr (is_u32<S>) return _mm_cmplt_epu32_mask(a, b);
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
				if constexpr (sizeof(T) > 64) return M{ cmp_equal(a.lo(),b.lo()), cmp_equal(a.hi(),b.hi()) };
				else if constexpr (zmm_sized<T>)
				{
					if constexpr (is_f64<S>) return _mm512_cmp_pd_mask(a, b, _CMP_LE_OQ);
					else if constexpr (is_f32<S>) return _mm512_cmp_ps_mask(a, b, _CMP_LE_OQ);
					else if constexpr (is_i64<S>) return _mm512_cmple_epi64_mask(a, b);
					else if constexpr (is_u64<S>) return _mm512_cmple_epu64_mask(a, b);
					else if constexpr (is_i32<S>) return _mm512_cmple_epi32_mask(a, b);
					else if constexpr (is_u32<S>) return _mm512_cmple_epu32_mask(a, b);
					else return fail_ack_t{};
				}
				else if constexpr (FS.has(AVX512_VL))
				{
					if constexpr (ymm_sized<T>)
					{
						if constexpr (is_f64<S>) return _mm256_cmp_pd_mask(a, b, _CMP_LE_OQ);
						else if constexpr (is_f32<S>) return _mm256_cmp_ps_mask(a, b, _CMP_LE_OQ);
						else if constexpr (is_i64<S>) return _mm256_cmple_epi64_mask(a, b);
						else if constexpr (is_u64<S>) return _mm256_cmple_epu64_mask(a, b);
						else if constexpr (is_i32<S>) return _mm256_cmple_epi32_mask(a, b);
						else if constexpr (is_u32<S>) return _mm256_cmple_epu32_mask(a, b);
						else return fail_ack_t{};
					}
					else if constexpr (xmm_sized<T>)
					{
						if constexpr (is_f64<S>) return _mm_cmp_pd_mask(a, b, _CMP_LE_OQ);
						else if constexpr (is_f32<S>) return _mm_cmp_ps_mask(a, b, _CMP_LE_OQ);
						else if constexpr (is_i64<S>) return _mm_cmple_epi64_mask(a, b);
						else if constexpr (is_u64<S>) return _mm_cmple_epu64_mask(a, b);
						else if constexpr (is_i32<S>) return _mm_cmple_epi32_mask(a, b);
						else if constexpr (is_u32<S>) return _mm_cmple_epu32_mask(a, b);
						else return fail_ack_t{};
					}
					else return fail_ack_t{};
				}
				else return fail_ack_t{};
			}

			template<typename Op, typename S, size_t N, typename I>
				requires (meta::any_int<I>&& std::same_as<Op, op_permx>)
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
				else if constexpr (zmm_sized<T> && is_f64<S>) return _mm512_permutexvar_pd(ind, a);
				else if constexpr (zmm_sized<T> && is_f32<S>) return _mm512_permutexvar_ps(ind, a);
				else if constexpr (zmm_sized<T> && any_i64<S>) return _mm512_permutexvar_epi64(ind, a);
				else if constexpr (zmm_sized<T> && any_i32<S>) return _mm512_permutexvar_epi32(ind, a);
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && is_f64<S>) return _mm256_permutexvar_pd(ind, a);
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && is_f32<S>) return _mm256_permutexvar_ps(ind, a);
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && any_i64<S>) return _mm256_permutexvar_epi64(ind, a);
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && any_i32<S>) return _mm256_permutexvar_epi32(ind, a);
				else return fail_ack_t{};
			}
			template<typename Op, typename S, size_t N, typename I>
				requires (meta::any_int<I>&& std::same_as<Op, op_permx2>)
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b, const SIMD_Vector<I, N>& ind)
			{
				using namespace meta;
				using canon_t = typename ScalarTraits<S>::UintT;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(I) != sizeof(S)) return permx2(a, vcvt<canon_t>(ind));
				else if constexpr (sizeof(T) > 64)
				{
					T pa = permx(a, ind);
					T pb = permx(b, ind);
					return mask_mov(pb, (ind & (2 * N - 1)) < N, pa);
				}
				else if constexpr (zmm_sized<T> && is_f64<S>) return _mm512_permutex2var_pd(a, ind, b);
				else if constexpr (zmm_sized<T> && is_f32<S>) return _mm512_permutex2var_ps(a, ind, b);
				else if constexpr (zmm_sized<T> && any_i64<S>) return _mm512_permutex2var_epi64(a, ind, b);
				else if constexpr (zmm_sized<T> && any_i32<S>) return _mm512_permutex2var_epi32(a, ind, b);
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && is_f64<S>) return _mm256_permutex2var_pd(a, ind, b);
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && is_f32<S>) return _mm256_permutex2var_ps(a, ind, b);
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && any_i64<S>) return _mm256_permutex2var_epi64(a, ind, b);
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && any_i32<S>) return _mm256_permutex2var_epi32(a, ind, b);
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && is_f64<S>) return _mm_permutex2var_pd(a, ind, b);
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && is_f32<S>) return _mm_permutex2var_ps(a, ind, b);
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && any_i64<S>) return _mm_permutex2var_epi64(a, ind, b);
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && any_i32<S>) return _mm_permutex2var_epi32(a, ind, b);
				else return fail_ack_t{};
			}

			template<typename Op, typename S, size_t N>
				requires (std::is_floating_point_v<S>&& std::same_as<Op, op_floor>)
			static auto eval(const SIMD_Vector<S, N>& a)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return T{ floor(a.lo()), floor(a.hi()) };
				else if constexpr (zmm_sized<T> && is_f64<S>) return _mm512_floor_pd(a);
				else if constexpr (zmm_sized<T> && is_f32<S>) return _mm512_floor_ps(a);
				else return fail_ack_t{};
			}
			template<typename Op, typename S, size_t N>
				requires (std::is_floating_point_v<S>&& std::same_as<Op, op_ceil>)
			static auto eval(const SIMD_Vector<S, N>& a)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return T{ ceil(a.lo()), ceil(a.hi()) };
				else if constexpr (zmm_sized<T> && is_f64<S>) return _mm512_ceil_pd(a);
				else if constexpr (zmm_sized<T> && is_f32<S>) return _mm512_ceil_ps(a);
				else return fail_ack_t{};
			}
			template<typename Op, size_t N>
				requires (std::same_as<Op, op_sqrtf>)
			static SIMD_Vector<float, N> eval(const SIMD_Vector<float, N>& a)
			{
				using T = SIMD_Vector<float, N>;
				if constexpr (sizeof(T) > 64) return { sqrtf(a.lo()), sqrtf(a.hi()) };
				else if constexpr (zmm_sized<T>) return _mm512_sqrt_ps(a);
				else return fail_ack_t{};
			}
			template<typename Op, size_t N>
				requires (std::same_as<Op, op_sqrtd>)
			static SIMD_Vector<double, N> eval(const SIMD_Vector<double, N>& a)
			{
				using T = SIMD_Vector<double, N>;
				if constexpr (sizeof(T) > 64) return { sqrtd(a.lo()), sqrtd(a.hi()) };
				else if constexpr (zmm_sized<T>) return _mm512_sqrt_pd(a);
				else return fail_ack_t{};
			}

			template<typename Op, size_t N, typename From>
				requires (meta::IsCvtOp<Op>)
			static auto eval(const SIMD_Vector<From, N>& a)
			{
				using namespace meta;
				using To = typename Op::cvt_to_t;
				using TV = SIMD_Vector<To, N>;
				using FV = SIMD_Vector<From, N>;
				constexpr size_t MaxSize = std::max(sizeof(TV), sizeof(FV));

				//TODO: add FP16 and BF16 conversions!
				//TODO: these allow to rewrite div to not do conversion by itself
				if constexpr (any_small_int<From> && !any_int<To>)
				{
					using interm_t = std::conditional_t<(std::is_signed_v<From>), int32_t, uint32_t>;
					return vcvt<To>(vcvt<interm_t>(a));
				}
				else if constexpr (!any_int<From> && any_small_int<To>)
				{
					using interm_t = std::conditional_t<(std::is_signed_v<To>), int32_t, uint32_t>;
					return vcvt<To>(vcvt<interm_t>(a));
				}
				else if constexpr (MaxSize > 64) return TV{ vcvt<To>(a.lo()), vcvt<To>(a.hi()) };
				else if constexpr (is_zmm_size(MaxSize))
				{
					if constexpr (is_fp16<From> && is_f64<To>) return vcvt<To>(vcvt<float>(a));
					//from double
					else if constexpr (is_f64<From> && is_i32<To>) return _mm512_cvttpd_epi32(a);
					else if constexpr (is_f64<From> && is_u32<To>) return _mm512_cvttpd_epu32(a);
					else if constexpr (is_f64<From> && is_f32<To>) return _mm512_cvtpd_ps(a);

					//from float
					else if constexpr (is_f32<From> && is_i32<To>) return _mm512_cvttps_epi32(a);
					else if constexpr (is_f32<From> && is_u32<To>) return _mm512_cvttps_epu32(a);
					else if constexpr (is_f32<From> && is_f64<To>) return _mm512_cvtps_pd(a);

					//from i64
					else if constexpr (any_i64<From> && any_i32<To>) return _mm512_cvtepi64_epi32(a);
					else if constexpr (any_i64<From> && any_i16<To>) return _mm512_cvtepi64_epi16(a);
					else if constexpr (any_i64<From> && any_i8<To>) return _mm512_cvtepi64_epi8(a);

					//from i32
					else if constexpr (is_i32<From> && is_f64<To>) return _mm512_cvtepi32_pd(a);
					else if constexpr (is_i32<From> && is_f32<To>) return _mm512_cvtepi32_ps(a);
					else if constexpr (is_i32<From> && any_i64<To>) return _mm512_cvtepi32_epi64(a);
					else if constexpr (is_i32<From> && any_i16<To>) return _mm512_cvtepi32_epi16(a);
					else if constexpr (is_i32<From> && any_i8<To>) return _mm512_cvtepi32_epi8(a);

					//from u32
					else if constexpr (is_u32<From> && is_f64<To>) return _mm512_cvtepu32_pd(a);
					else if constexpr (is_u32<From> && is_f32<To>) return _mm512_cvtepu32_ps(a);
					else if constexpr (is_u32<From> && any_i64<To>) return _mm512_cvtepu32_epi64(a);

					//from 16 bit ints
					else if constexpr (is_i16<From> && any_i64<To>) return _mm512_cvtepi16_epi64(a);
					else if constexpr (is_i16<From> && any_i32<To>) return _mm512_cvtepi16_epi32(a);
					else if constexpr (is_u16<From> && any_i64<To>) return _mm512_cvtepu16_epi64(a);
					else if constexpr (is_u16<From> && any_i32<To>) return _mm512_cvtepu16_epi32(a);

					//from 8 bit ints
					else if constexpr (is_i8<From> && any_i64<To>) return _mm512_cvtepi8_epi64(a);
					else if constexpr (is_i8<From> && any_i32<To>) return _mm512_cvtepi8_epi32(a);
					else if constexpr (is_u8<From> && any_i64<To>) return _mm512_cvtepu8_epi64(a);
					else if constexpr (is_u8<From> && any_i32<To>) return _mm512_cvtepu8_epi32(a);

					else if constexpr (is_fp16<From> && is_f32<To>) return _mm512_cvtph_ps(a);
					else if constexpr (is_f32<From> && is_fp16<To>) return _mm512_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT);
					else return fail_ack_t{};
				}
				else if constexpr (FS.has(AVX512_VL) && is_ymm_size(MaxSize))
				{
					if constexpr (is_f64<From> && is_u32<To>) return _mm256_cvttpd_epu32(a);
					else if constexpr (is_f32<From> && is_u32<To>) return _mm256_cvttps_epu32(a);
					else if constexpr (any_i64<From> && any_i32<To>) return _mm256_cvtepi64_epi32(a);
					else if constexpr (any_i64<From> && any_i16<To>) return _mm256_cvtepi64_epi16(a);
					else if constexpr (any_i64<From> && any_i8<To>) return _mm256_cvtepi64_epi8(a);
					else if constexpr (any_i32<From> && any_i16<To>) return _mm256_cvtepi32_epi16(a);
					else if constexpr (any_i32<From> && any_i8<To>) return _mm256_cvtepi32_epi8(a);
					else return fail_ack_t{};
				}
				else if constexpr (FS.has(AVX512_VL) && is_xmm_size(MaxSize))
				{
					if constexpr (is_f64<From> && is_u32<To>) return _mm_cvttpd_epu32(a);
					else if constexpr (is_f32<From> && is_u32<To>) return _mm_cvttps_epu32(a);
					else if constexpr (any_i64<From> && any_i32<To>) return _mm_cvtepi64_epi32(a);
					else if constexpr (any_i64<From> && any_i16<To>) return _mm_cvtepi64_epi16(a);
					else if constexpr (any_i64<From> && any_i8<To>) return _mm_cvtepi64_epi8(a);
					else if constexpr (any_i32<From> && any_i16<To>) return _mm_cvtepi32_epi16(a);
					else if constexpr (any_i32<From> && any_i8<To>) return _mm_cvtepi32_epi8(a);
					else return fail_ack_t{};
				}
				else return fail_ack_t{};
			}

			template<typename Op, typename I>
				requires (meta::any_int<I>&& IsGatherOp<Op>)
			static auto eval(const void* base, const SIMD_Vector<I, Op::N>& ind, const typename SIMD_Vector<Op::S, Op::N>::MaskT& mask, const SIMD_Vector<Op::S, Op::N>& src)
			{
				//put everything up here to prevent else if chain breaks (since compilation gives useless errors by thinking unsanitized inputs surviving to native gathers
				using namespace meta;
				using CanonicalIndex_t = std::conditional_t<(sizeof(I) <= 4), int32_t, int64_t>;
				using S = typename Op::S;
				static constexpr size_t N = Op::N;
				using RetVec_t = SIMD_Vector<S, N>;
				using IndVec_t = SIMD_Vector<I, N>;
				constexpr size_t Scale = Op::Scale;
				constexpr size_t MaxSize = std::max(sizeof(RetVec_t), sizeof(IndVec_t));

				//if scale is not native, emulate it by gathering with scale 1 and manually calculated byte offsets. 
				//TODO: Can optimize a little by checking if Scale*maxint(I) fits into smaller sizes
				if constexpr (Scale != 1 && Scale != 2 && Scale != 4 && Scale != 8) return gather<S, N, 1>(base, vcvt<int64_t>(ind) * Scale, mask, src);

				//TODO: emulation of small int gathers (where elements gathered are small ints)
				else if constexpr (!std::is_same_v<I, CanonicalIndex_t>) return gather<S, N, Scale>(base, vcvt<CanonicalIndex_t>(ind), mask, src);

				//if we get here, means that indices are already in good format (4-byte or 8-byte)
				//break up large gather into halves
				else if constexpr (MaxSize > 64) return RetVec_t{
					gather<S, N / 2, Scale, I>(base, ind.lo(), mask.lo(), src.lo()),
					gather<S, N / 2, Scale, I>(base, ind.hi(), mask.hi(), src.hi()) };
				else if constexpr (is_zmm_size(MaxSize))
				{
					//clang is a cry-baby with ind here for some reason, so force convert it. Pay attention to size!
					std::conditional_t<(meta::zmm_sized<IndVec_t>), __m512i, __m256i> ni = ind;
					if constexpr (is_i64<I> && is_f64<S>) return _mm512_mask_i64gather_pd(src, mask, ni, base, Scale);
					else if constexpr (is_i64<I> && is_f32<S>) return _mm512_mask_i64gather_ps(src, mask, ni, base, Scale);
					else if constexpr (is_i64<I> && any_i64<S>) return _mm512_mask_i64gather_epi64(src, mask, ni, base, Scale);
					else if constexpr (is_i64<I> && any_i32<S>) return _mm512_mask_i64gather_epi32(src, mask, ni, base, Scale);

					else if constexpr (is_i32<I> && is_f64<S>) return _mm512_mask_i32gather_pd(src, mask, ni, base, Scale);
					else if constexpr (is_i32<I> && is_f32<S>) return _mm512_mask_i32gather_ps(src, mask, ni, base, Scale);
					else if constexpr (is_i32<I> && any_i64<S>) return _mm512_mask_i32gather_epi64(src, mask, ni, base, Scale);
					else if constexpr (is_i32<I> && any_i32<S>) return _mm512_mask_i32gather_epi32(src, mask, ni, base, Scale);
					else return fail_ack_t{};
				}
				else if constexpr (FS.has(AVX512_VL))
				{
					if constexpr (is_ymm_size(MaxSize))
					{
						std::conditional_t<(meta::ymm_sized<IndVec_t>), __m256i, __m128i> ni = ind;
						if constexpr (is_i64<I> && is_f64<S>) return _mm256_mmask_i64gather_pd(src, mask, ni, base, Scale);
						else if constexpr (is_i64<I> && is_f32<S>) return _mm256_mmask_i64gather_ps(src, mask, ni, base, Scale);
						else if constexpr (is_i64<I> && any_i64<S>) return _mm256_mmask_i64gather_epi64(src, mask, ni, base, Scale);
						else if constexpr (is_i64<I> && any_i32<S>) return _mm256_mmask_i64gather_epi32(src, mask, ni, base, Scale);

						else if constexpr (is_i32<I> && is_f64<S>) return _mm256_mmask_i32gather_pd(src, mask, ni, base, Scale);
						else if constexpr (is_i32<I> && is_f32<S>) return _mm256_mmask_i32gather_ps(src, mask, ni, base, Scale);
						else if constexpr (is_i32<I> && any_i64<S>) return _mm256_mmask_i32gather_epi64(src, mask, ni, base, Scale);
						else if constexpr (is_i32<I> && any_i32<S>) return _mm256_mmask_i32gather_epi32(src, mask, ni, base, Scale);
						else return fail_ack_t{};
					}
					else if constexpr (is_xmm_size(MaxSize))
					{
						__m128i ni = ind;
						if constexpr (is_i64<I> && is_f64<S>) return _mm_mmask_i64gather_pd(src, mask, ni, base, Scale);
						else if constexpr (is_i64<I> && is_f32<S>) return _mm_mmask_i64gather_ps(src, mask, ni, base, Scale);
						else if constexpr (is_i64<I> && any_i64<S>) return _mm_mmask_i64gather_epi64(src, mask, ni, base, Scale);
						else if constexpr (is_i64<I> && any_i32<S>) return _mm_mmask_i64gather_epi32(src, mask, ni, base, Scale);

						else if constexpr (is_i32<I> && is_f64<S>) return _mm_mmask_i32gather_pd(src, mask, ni, base, Scale);
						else if constexpr (is_i32<I> && is_f32<S>) return _mm_mmask_i32gather_ps(src, mask, ni, base, Scale);
						else if constexpr (is_i32<I> && any_i64<S>) return _mm_mmask_i32gather_epi64(src, mask, ni, base, Scale);
						else if constexpr (is_i32<I> && any_i32<S>) return _mm_mmask_i32gather_epi32(src, mask, ni, base, Scale);
						else return fail_ack_t{};
					}
					else return fail_ack_t{};
				}
				else return fail_ack_t{};
			}

			template<typename Op, typename S, size_t N, typename I>
				requires (meta::any_int<I>&& meta::IsScatterOp<Op>)
			static auto eval(const SIMD_Vector<S, N>& v, void* base, const SIMD_Vector<I, N>& ind, const typename SIMD_Vector<S, N>::MaskT& mask)
			{
				//put everything up here to prevent else if chain breaks (since compilation gives useless errors by thinking unsanitized inputs surviving to native gathers
				using CanonicalIndex_t = std::conditional_t<(sizeof(I) <= 4), int32_t, int64_t>;
				using RetVec_t = SIMD_Vector<S, N>;
				using IndVec_t = SIMD_Vector<I, N>;
				constexpr size_t MaxSize = std::max(sizeof(RetVec_t), sizeof(IndVec_t));
				constexpr size_t Scale = Op::Scale;
				using namespace meta;

				//if scale is not native, emulate it by gathering with scale 1 and manually calculated byte offsets. 
				//TODO: Can optimize a little by checking if Scale*maxint(I) fits into smaller sizes
				if constexpr (Scale != 1 && Scale != 2 && Scale != 4 && Scale != 8) return scatter<S, N, 1>(v, base, vcvt<int64_t>(ind) * Scale, mask);

				//TODO: emulation of small int scatter (where elements gathered are small ints)
				else if constexpr (!std::is_same_v<I, CanonicalIndex_t>) return scatter<S, N, Scale>(v, base, vcvt<CanonicalIndex_t>(ind), mask);

				//if we get here, means that indices are already in good format (4-byte or 8-byte)
				//break up large scatter into halves
				else if constexpr (MaxSize > 64)
				{
					scatter<S, N / 2, Scale, I>(v.lo(), base, ind.lo(), mask.lo());
					scatter<S, N / 2, Scale, I>(v.hi(), base, ind.hi(), mask.hi());
					return success_ack_t{};
				}
				//clang is a cry-baby with inds for some reason, so force convert it. Pay attention to size!
				else if constexpr (is_zmm_size(MaxSize))
				{
					std::conditional_t<(zmm_sized<IndVec_t>), __m512i, __m256i> ni = ind;
					if constexpr (is_i64<I> && is_f64<S>) { _mm512_mask_i64scatter_pd(base, mask, ni, v, Scale); return success_ack_t{}; }
					else if constexpr (is_i64<I> && is_f32<S>) { _mm512_mask_i64scatter_ps(base, mask, ni, v, Scale); return success_ack_t{}; }
					else if constexpr (is_i64<I> && any_i64<S>) { _mm512_mask_i64scatter_epi64(base, mask, ni, v, Scale); return success_ack_t{}; }
					else if constexpr (is_i64<I> && any_i32<S>) { _mm512_mask_i64scatter_epi32(base, mask, ni, v, Scale); return success_ack_t{}; }

					else if constexpr (is_i32<I> && is_f64<S>) { _mm512_mask_i32scatter_pd(base, mask, ni, v, Scale); return success_ack_t{}; }
					else if constexpr (is_i32<I> && is_f32<S>) { _mm512_mask_i32scatter_ps(base, mask, ni, v, Scale); return success_ack_t{}; }
					else if constexpr (is_i32<I> && any_i64<S>) { _mm512_mask_i32scatter_epi64(base, mask, ni, v, Scale); return success_ack_t{}; }
					else if constexpr (is_i32<I> && any_i32<S>) { _mm512_mask_i32scatter_epi32(base, mask, ni, v, Scale); return success_ack_t{}; }
					else return fail_ack_t{};
				}
				else if constexpr (FS.has(AVX512_VL))
				{
					if constexpr (is_ymm_size(MaxSize))
					{
						std::conditional_t<(meta::ymm_sized<IndVec_t>), __m256i, __m128i> ni = ind;
						if constexpr (is_i64<I> && is_f64<S>) { _mm256_mask_i64scatter_pd(base, mask, ni, v, Scale); return success_ack_t{}; }
						else if constexpr (is_i64<I> && is_f32<S>) { _mm256_mask_i64scatter_ps(base, mask, ni, v, Scale); return success_ack_t{}; }
						else if constexpr (is_i64<I> && any_i64<S>) { _mm256_mask_i64scatter_epi64(base, mask, ni, v, Scale); return success_ack_t{}; }
						else if constexpr (is_i64<I> && any_i32<S>) { _mm256_mask_i64scatter_epi32(base, mask, ni, v, Scale); return success_ack_t{}; }

						else if constexpr (is_i32<I> && is_f64<S>) { _mm256_mask_i32scatter_pd(base, mask, ni, v, Scale); return success_ack_t{}; }
						else if constexpr (is_i32<I> && is_f32<S>) { _mm256_mask_i32scatter_ps(base, mask, ni, v, Scale); return success_ack_t{}; }
						else if constexpr (is_i32<I> && any_i64<S>) { _mm256_mask_i32scatter_epi64(base, mask, ni, v, Scale); return success_ack_t{}; }
						else if constexpr (is_i32<I> && any_i32<S>) { _mm256_mask_i32scatter_epi32(base, mask, ni, v, Scale); return success_ack_t{}; }
						else return fail_ack_t{};
					}
					else if constexpr (is_xmm_size(MaxSize))
					{
						__m128i ni = ind;
						if constexpr (is_i64<I> && is_f64<S>) { _mm_mask_i64scatter_pd(base, mask, ni, v, Scale); return success_ack_t{}; }
						else if constexpr (is_i64<I> && is_f32<S>) { _mm_mask_i64scatter_ps(base, mask, ni, v, Scale); return success_ack_t{}; }
						else if constexpr (is_i64<I> && any_i64<S>) { _mm_mask_i64scatter_epi64(base, mask, ni, v, Scale); return success_ack_t{}; }
						else if constexpr (is_i64<I> && any_i32<S>) { _mm_mask_i64scatter_epi32(base, mask, ni, v, Scale); return success_ack_t{}; }

						else if constexpr (is_i32<I> && is_f64<S>) { _mm_mask_i32scatter_pd(base, mask, ni, v, Scale); return success_ack_t{}; }
						else if constexpr (is_i32<I> && is_f32<S>) { _mm_mask_i32scatter_ps(base, mask, ni, v, Scale); return success_ack_t{}; }
						else if constexpr (is_i32<I> && any_i64<S>) { _mm_mask_i32scatter_epi64(base, mask, ni, v, Scale); return success_ack_t{}; }
						else if constexpr (is_i32<I> && any_i32<S>) { _mm_mask_i32scatter_epi32(base, mask, ni, v, Scale); return success_ack_t{}; }
						else return fail_ack_t{};
					}
					else return fail_ack_t{};
				}
				else return fail_ack_t{};
			}
			template<typename Op>
				requires (meta::IsLoadOp<Op>)
			static auto eval(const void* p, const typename SIMD_Vector<typename Op::S, Op::N>::MaskT& mask, const SIMD_Vector<typename Op::S, Op::N>& src)
			{
				using namespace meta;
				using S = typename Op::S;
				static constexpr size_t N = Op::N;
				using T = SIMD_Vector<S, N>;
				SIMD_Vector<S, N> ret;
				const S* sp = (const S*)p;
				if constexpr (sizeof(T) > 64) return T{ load<S,N / 2>(sp,mask.lo(), src.lo()), load<S,N / 2>(sp + N / 2, mask.hi(), src.hi()) };
				else if constexpr (zmm_sized<T> && is_f64<S>) return _mm512_mask_loadu_pd(src, mask, p);
				else if constexpr (zmm_sized<T> && is_f32<S>) return _mm512_mask_loadu_ps(src, mask, p);
				else if constexpr (zmm_sized<T> && any_i64<S>) return _mm512_mask_loadu_epi64(src, mask, p);
				else if constexpr (zmm_sized<T> && any_i32<S>) return _mm512_mask_loadu_epi32(src, mask, p);
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && is_f64<S>) return _mm256_mask_loadu_pd(src, mask, p);
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && is_f32<S>) return _mm256_mask_loadu_ps(src, mask, p);
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && any_i64<S>) return _mm256_mask_loadu_epi64(src, mask, p);
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && any_i32<S>) return _mm256_mask_loadu_epi32(src, mask, p);
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && is_f64<S>) return _mm_mask_loadu_pd(src, mask, p);
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && is_f32<S>) return _mm_mask_loadu_ps(src, mask, p);
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && any_i64<S>) return _mm_mask_loadu_epi64(src, mask, p);
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && any_i32<S>) return _mm_mask_loadu_epi32(src, mask, p);
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
				if constexpr (sizeof(T) > 64) {
					store(vec.lo(), p, mask.lo());
					store(vec.hi(), sp + N / 2, mask.hi());
					return success_ack_t{};
				}
				else if constexpr (zmm_sized<T> && is_f64<S>) { _mm512_mask_storeu_pd(p, mask, vec); return success_ack_t{}; }
				else if constexpr (zmm_sized<T> && is_f32<S>) { _mm512_mask_storeu_ps(p, mask, vec); return success_ack_t{}; }
				else if constexpr (zmm_sized<T> && any_i64<S>) { _mm512_mask_storeu_epi64(p, mask, vec); return success_ack_t{}; }
				else if constexpr (zmm_sized<T> && any_i32<S>) { _mm512_mask_storeu_epi32(p, mask, vec); return success_ack_t{}; }
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && is_f64<S>) { _mm256_mask_storeu_pd(p, mask, vec); return success_ack_t{}; }
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && is_f32<S>) { _mm256_mask_storeu_ps(p, mask, vec); return success_ack_t{}; }
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && any_i64<S>) { _mm256_mask_storeu_epi64(p, mask, vec); return success_ack_t{}; }
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && any_i32<S>) { _mm256_mask_storeu_epi32(p, mask, vec); return success_ack_t{}; }
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && is_f64<S>) { _mm_mask_storeu_pd(p, mask, vec); return success_ack_t{}; }
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && is_f32<S>) { _mm_mask_storeu_ps(p, mask, vec); return success_ack_t{}; }
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && any_i64<S>) { _mm_mask_storeu_epi64(p, mask, vec); return success_ack_t{}; }
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && any_i32<S>) { _mm_mask_storeu_epi32(p, mask, vec); return success_ack_t{}; }
				else return fail_ack_t{};
			}

			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_compress>)
			static auto eval(const typename SIMD_Vector<S, N>::MaskT& mask, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& src = 0)
			{
				//TODO: more than 64 bytes!
				//if constexpr (sizeof(SIMD_Vector<S, N>) > 64) {};
				//else
				using T = SIMD_Vector<S, N>;
				if constexpr (zmm_sized<T> && is_f64<S>) return _mm512_mask_compress_pd(src, mask, a);
				else if constexpr (zmm_sized<T> && is_f32<S>) return _mm512_mask_compress_ps(src, mask, a);
				else if constexpr (zmm_sized<T> && any_i64<S>) return _mm512_mask_compress_epi64(src, mask, a);
				else if constexpr (zmm_sized<T> && any_i32<S>) return _mm512_mask_compress_epi32(src, mask, a);
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && is_f64<S>) return _mm256_mask_compress_pd(src, mask, a);
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && is_f32<S>) return _mm256_mask_compress_ps(src, mask, a);
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && any_i64<S>) return _mm256_mask_compress_epi64(src, mask, a);
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && any_i32<S>) return _mm256_mask_compress_epi32(src, mask, a);
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && is_f64<S>) return _mm_mask_compress_pd(src, mask, a);
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && is_f32<S>) return _mm_mask_compress_ps(src, mask, a);
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && any_i64<S>) return _mm_mask_compress_epi64(src, mask, a);
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && any_i32<S>) return _mm_mask_compress_epi32(src, mask, a);
				else return fail_ack_t{};
			}

			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_unpacklo>)
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return T{ unpacklo(a.lo(),b.lo()), unpacklo(a.hi(),b.hi()) };
				else if constexpr (zmm_sized<T> && is_f64<S>) return _mm512_unpacklo_pd(a, b);
				else if constexpr (zmm_sized<T> && is_f32<S>) return _mm512_unpacklo_ps(a, b);
				else if constexpr (zmm_sized<T> && any_i64<S>) return _mm512_unpacklo_epi64(a, b);
				else if constexpr (zmm_sized<T> && any_i32<S>) return _mm512_unpacklo_epi32(a, b);
				else return fail_ack_t{};
			}
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_unpackhi>)
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return T{ unpackhi(a.lo(),b.lo()), unpackhi(a.hi(),b.hi()) };
				else if constexpr (zmm_sized<T> && is_f64<S>) return _mm512_unpackhi_pd(a, b);
				else if constexpr (zmm_sized<T> && is_f32<S>) return _mm512_unpackhi_ps(a, b);
				else if constexpr (zmm_sized<T> && any_i64<S>) return _mm512_unpackhi_epi64(a, b);
				else if constexpr (zmm_sized<T> && any_i32<S>) return _mm512_unpackhi_epi32(a, b);
				else return fail_ack_t{};
			}
		private:

		};
	}
}