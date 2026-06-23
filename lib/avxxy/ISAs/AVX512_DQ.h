#pragma once
#include "shared.h"
namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		using namespace meta;
		struct ISA_AVX512_DQ
		{
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_and>)
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return T{ logic_and(a.lo(),b.lo()), logic_and(a.hi(),b.hi()) };
				else if constexpr (zmm_sized<T> && is_f64<S>) return _mm512_and_pd(a, b);
				else if constexpr (zmm_sized<T> && is_f32<S>) return _mm512_and_ps(a, b);
				else return fail_ack_t{};
			}
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_or>)
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return T{ logic_or(a.lo(),b.lo()), logic_or(a.hi(),b.hi()) };
				else if constexpr (zmm_sized<T> && is_f64<S>) return _mm512_or_pd(a, b);
				else if constexpr (zmm_sized<T> && is_f32<S>) return _mm512_or_ps(a, b);
				else return fail_ack_t{};
			}
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_xor>)
			static auto eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(SIMD_Vector<S, N>) > 64) return T{ logic_xor(a.lo(),b.lo()), logic_xor(a.hi(),b.hi()) };
				else if constexpr (zmm_sized<SIMD_Vector<S, N>> && is_f64<S>) return _mm512_xor_pd(a, b);
				else if constexpr (zmm_sized<SIMD_Vector<S, N>> && is_f32<S>) return _mm512_xor_ps(a, b);
				else return fail_ack_t{};
			}
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_not>)
			static auto eval(const SIMD_Vector<S, N>& a)
			{
				if constexpr (sizeof(SIMD_Vector<S, N>) > 32)
				{
					using U = ScalarTraits<S>::UintT;
					S val = std::bit_cast<S>(~U(0));
					return logic_xor(a, val);
				}
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
				if constexpr (MaxSize > 64) return TV{ vcvt<To>(a.lo()), vcvt<To>(a.hi()) };
				else if constexpr (is_zmm_size(MaxSize) && is_i64<From> && is_f64<To>) return _mm512_cvtepi64_pd(a);
				else if constexpr (is_zmm_size(MaxSize) && is_u64<From> && is_f64<To>) return _mm512_cvtepu64_pd(a);
				else if constexpr (is_zmm_size(MaxSize) && is_i64<From> && is_f32<To>) return _mm512_cvtepi64_ps(a);
				else if constexpr (is_zmm_size(MaxSize) && is_u64<From> && is_f32<To>) return _mm512_cvtepu64_ps(a);
				else if constexpr (is_zmm_size(MaxSize) && is_f64<From> && is_i64<To>) return _mm512_cvttpd_epi64(a);
				else if constexpr (is_zmm_size(MaxSize) && is_f64<From> && is_u64<To>) return _mm512_cvttpd_epu64(a);
				else if constexpr (is_zmm_size(MaxSize) && is_f32<From> && is_i64<To>) return _mm512_cvttps_epi64(a);
				else if constexpr (is_zmm_size(MaxSize) && is_f32<From> && is_u64<To>) return _mm512_cvttps_epu64(a);
				else if constexpr (FS.has(AVX512_VL) && is_ymm_size(MaxSize) && is_i64<From> && is_f64<To>) return _mm256_cvtepi64_pd(a);
				else if constexpr (FS.has(AVX512_VL) && is_ymm_size(MaxSize) && is_u64<From> && is_f64<To>) return _mm256_cvtepu64_pd(a);
				else if constexpr (FS.has(AVX512_VL) && is_ymm_size(MaxSize) && is_i64<From> && is_f32<To>) return _mm256_cvtepi64_ps(a);
				else if constexpr (FS.has(AVX512_VL) && is_ymm_size(MaxSize) && is_u64<From> && is_f32<To>) return _mm256_cvtepu64_ps(a);
				else if constexpr (FS.has(AVX512_VL) && is_ymm_size(MaxSize) && is_f64<From> && is_i64<To>) return _mm256_cvttpd_epi64(a);
				else if constexpr (FS.has(AVX512_VL) && is_ymm_size(MaxSize) && is_f64<From> && is_u64<To>) return _mm256_cvttpd_epu64(a);
				else if constexpr (FS.has(AVX512_VL) && is_ymm_size(MaxSize) && is_f32<From> && is_i64<To>) return _mm256_cvttps_epi64(a);
				else if constexpr (FS.has(AVX512_VL) && is_ymm_size(MaxSize) && is_f32<From> && is_u64<To>) return _mm256_cvttps_epu64(a);
				else if constexpr (FS.has(AVX512_VL) && is_xmm_size(MaxSize) && is_i64<From> && is_f64<To>) return _mm_cvtepi64_pd(a);
				else if constexpr (FS.has(AVX512_VL) && is_xmm_size(MaxSize) && is_u64<From> && is_f64<To>) return _mm_cvtepu64_pd(a);
				else if constexpr (FS.has(AVX512_VL) && is_xmm_size(MaxSize) && is_i64<From> && is_f32<To>) return _mm_cvtepi64_ps(a);
				else if constexpr (FS.has(AVX512_VL) && is_xmm_size(MaxSize) && is_u64<From> && is_f32<To>) return _mm_cvtepu64_ps(a);
				else if constexpr (FS.has(AVX512_VL) && is_xmm_size(MaxSize) && is_f64<From> && is_i64<To>) return _mm_cvttpd_epi64(a);
				else if constexpr (FS.has(AVX512_VL) && is_xmm_size(MaxSize) && is_f64<From> && is_u64<To>) return _mm_cvttpd_epu64(a);
				else if constexpr (FS.has(AVX512_VL) && is_xmm_size(MaxSize) && is_f32<From> && is_i64<To>) return _mm_cvttps_epi64(a);
				else if constexpr (FS.has(AVX512_VL) && is_xmm_size(MaxSize) && is_f32<From> && is_u64<To>) return _mm_cvttps_epu64(a);
				else return fail_ack_t{};
			}

			//TODO: add movm, movmask

			template<typename Op, typename S, size_t N>
			static auto eval(op_mul, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				requires (std::same_as<Op, op_mul>&& any_i64<S>)
			{
				using namespace meta;
				using T = SIMD_Vector<S, N>;
				if constexpr (sizeof(T) > 64) return T{ mul(a.lo(), b.lo()), mul(a.hi(), b.hi()) };
				else if constexpr (zmm_sized<T> && any_i64<S>) return _mm512_mullo_epi64(a, b);
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && any_i64<S>) return _mm256_mullo_epi64(a, b);
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && any_i64<S>) return _mm_mullo_epi64(a, b);
				else return fail_ack_t{};
			}
		};
	}
}