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
			using namespace concepts;
			using namespace utils;
			template<FeatureSet FS>
			struct AVX512DQ
			{
				template<typename S, size_t N>
				requires (sizeof(SIMD_Vector<S,N>) > 32 && (is_f32<S> || is_f64<S>))
				static SIMD_Vector<S, N> eval(op_and, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					if constexpr (sizeof(SIMD_Vector<S, N>) > 64) return { logic_and(a.lo(),b.lo()), logic_and(a.hi(),b.hi()) };
					else if constexpr (zmm_sized<SIMD_Vector<S, N>> && is_f64<S>) return _mm512_and_pd(a, b);
					else if constexpr (zmm_sized<SIMD_Vector<S, N>> && is_f32<S>) return _mm512_and_ps(a, b);
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 32 && (is_f32<S> || is_f64<S>))
				static SIMD_Vector<S, N> eval(op_or, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					if constexpr (sizeof(SIMD_Vector<S, N>) > 64) return { logic_or(a.lo(),b.lo()), logic_or(a.hi(),b.hi()) };
					else if constexpr (zmm_sized<SIMD_Vector<S, N>> && is_f64<S>) return _mm512_or_pd(a, b);
					else if constexpr (zmm_sized<SIMD_Vector<S, N>> && is_f32<S>) return _mm512_or_ps(a, b);
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 32 && (is_f32<S> || is_f64<S>))
				static SIMD_Vector<S, N> eval(op_xor, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					if constexpr (sizeof(SIMD_Vector<S, N>) > 64) return { logic_xor(a.lo(),b.lo()), logic_xor(a.hi(),b.hi()) };
					else if constexpr (zmm_sized<SIMD_Vector<S, N>> && is_f64<S>) return _mm512_xor_pd(a, b);
					else if constexpr (zmm_sized<SIMD_Vector<S, N>> && is_f32<S>) return _mm512_xor_ps(a, b);
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 32)
				static SIMD_Vector<S, N> eval(op_not, const SIMD_Vector<S, N>& a)
				{
					using U = same_size_uint_t<S>::type;
					S val = std::bit_cast<S>(~U(0));
					return logic_xor(a, val);
				}

				template<typename To, size_t N, typename From>
					requires (
				(std::max(sizeof(SIMD_Vector<To, N>), sizeof(SIMD_Vector<From, N>)) >= (FS.has(AVX512_VL) ? 0 : 33)) &&
				(
					(any_i64<From> && (is_f32<To> || is_f64<To>)) || 
					(any_i64<To> && (is_f32<From> || is_f64<From>))
				))
				static SIMD_Vector<To, N> eval(op_cvt<To>, const SIMD_Vector<From, N>& a)
				{
					using namespace concepts;
					using TV = SIMD_Vector<To, N>;
					using FV = SIMD_Vector<From, N>;
					constexpr size_t MaxSize = std::max(sizeof(TV), sizeof(FV));
					if constexpr (is_zmm_size(MaxSize) && is_i64<From> && is_f64<To>) return _mm512_cvtepi64_pd(a);
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
					else static_assert(always_false_v<From, To>);
				}

				//TODO: add movm, movmask

				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_mul, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
					requires (sizeof(SIMD_Vector<S, N>) >= (FS.has(AVX512_VL) ? 0 : 33) && any_i64<S>)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 64) return { mul(a.lo(), b.lo()), mul(a.hi(), b.hi()) };
					else if constexpr (zmm_sized<T> && any_i64<S>) return _mm512_mullo_epi64(a, b);
					else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && any_i64<S>) return _mm256_mullo_epi64(a, b);
					else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && any_i64<S>) return _mm_mullo_epi64(a, b);
					else static_assert(always_false_v<T>);
				}

				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) >= (FS.has(AVX512_VL) ? 0 : 33) && sizeof(S) >= 4)
				static SIMD_BitMask<N> eval(op_vec2mask, const SIMD_Vector<S, N>& v)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 64) return { vec2mask(v.lo()), vec2mask(v.hi()) };
					else if constexpr (zmm_sized<T> && sizeof(S) == 8) return _mm512_movepi64_mask(vreinterpret<__m512i>(v));
					else if constexpr (zmm_sized<T> && sizeof(S) == 4) return _mm512_movepi32_mask(vreinterpret<__m512i>(v));
					else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && sizeof(S) == 8) return _mm256_movepi64_mask(vreinterpret<__m256i>(v));
					else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && sizeof(S) == 4) return _mm256_movepi32_mask(vreinterpret<__m256i>(v));
					else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && sizeof(S) == 8) return _mm_movepi64_mask(vreinterpret<__m128i>(v));
					else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && sizeof(S) == 4) return _mm_movepi32_mask(vreinterpret<__m128i>(v));
					else static_assert(always_false_v<T>);
				}
			};
		}
	}
}