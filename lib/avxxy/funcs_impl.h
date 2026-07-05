#pragma once
#include "funcs.h"
#include "SIMD_Vector.h"
#include "FeatureSet.h"
#include <source_location>
#include "tables.h"
#include "typedefs.h"

namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		template<typename S, size_t N, bool Lo>
		static SIMD_Vector<S, N> unpack_base(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
		{
			SIMD_Vector<S, N> ret;
			constexpr size_t pairs_per_xmm = 8 / sizeof(S); //8, since unpack only processes lower/upper half of each input
			constexpr size_t elements_per_xmm = 16 / sizeof(S); //how much elements of type S fit into one 128 bit lane
			constexpr size_t xmm_count = sizeof(ret) / 16;
			for (size_t xmm_i = 0; xmm_i < xmm_count; ++xmm_i) //for each 128-bit lane
			{
				for (size_t i = 0; i < elements_per_xmm; i += 2)
				{
					size_t srcI = xmm_i * elements_per_xmm + i / 2 + (Lo ? 0 : elements_per_xmm / 2);
					ret[xmm_i * elements_per_xmm + i] = a[srcI];
					ret[xmm_i * elements_per_xmm + i + 1] = b[srcI];
				}
			}
			return ret;
		}

		//scream your lungs out if scalar fallback is reached and this function is enabled via AVXXY_NOISY_SCALAR define
		static void scream(std::source_location loc = std::source_location::current())
		{
#ifdef AVXXY_NOISY_SCALAR
			std::cout << "\nScalar fallback reached:" << loc.function_name() << "\n";
#endif
		}
	}


	namespace internals
	{
		//Attempts to split vector at HeadByte boundary.
		//Returns a pair of values, first value is not greater than HeadBytes large.
		//Second value not greater than sizeof(a) - HeadBytes bytes large
		//If input size is smaller or equal to HeadBytes, returns pair of a and std::nullopt
		template<size_t HeadBytes, typename S, size_t N>
		requires (HeadBytes % sizeof(S) == 0)
		auto vsplit(const SIMD_Vector<S, N>& a)
		{
			using T = SIMD_Vector<S, N>;
			if constexpr (sizeof(T) <= HeadBytes) return std::make_pair(a, std::nullopt);
			else
			{
				constexpr size_t HeadN = HeadBytes / sizeof(S);
				SIMD_Vector<S, HeadN> head;
				SIMD_Vector<S, N - HeadN> tail;
				memcpy(&head, &a[0], sizeof(head));
				memcpy(&tail, &a[HeadN], sizeof(tail));
				return std::make_pair(head, tail);
			}
		}
	}



	//#define AVXXY_RUN(op) internals::Dispatcher::run<internals::op>
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> add(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace internals;
		using namespace meta;
		using T = SIMD_Vector<S, N>;

		if constexpr (FS.has(AVX512_FP16) && zmm_sized<T> && is_fp16<S>) return _mm512_add_ph(a, b);
		else if constexpr (FS.has(AVX512_FP16) && FS.has(AVX512_VL) && ymm_sized<T> && is_fp16<S>) return _mm256_add_ph(a, b);
		else if constexpr (FS.has(AVX512_FP16) && FS.has(AVX512_VL) && xmm_sized<T> && is_fp16<S>) return _mm_add_ph(a, b);
		else if constexpr (!FS.has(AVX512_FP16) && is_fp16<S>) return vcvt<fp16_t>(add(vcvt<float>(a), vcvt<float>(b)));

		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f64<S>) return _mm512_add_pd(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f32<S>) return _mm512_add_ps(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && any_i64<S>) return _mm512_add_epi64(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && any_i32<S>) return _mm512_add_epi32(a, b);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_i16<S>) return _mm512_add_epi16(a, b);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_i8<S>) return _mm512_add_epi8(a, b);

		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i64<S>) return _mm256_add_epi64(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i32<S>) return _mm256_add_epi32(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i16<S>) return _mm256_add_epi16(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i8<S>) return _mm256_add_epi8(a, b);
		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f64<S>) return _mm256_add_pd(a, b);
		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f32<S>) return _mm256_add_ps(a, b);

		else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_f64<S>) return _mm_add_pd(a, b);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && any_i64<S>) return _mm_add_epi64(a, b);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && any_i32<S>) return _mm_add_epi32(a, b);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && any_i16<S>) return _mm_add_epi16(a, b);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && any_i8<S>) return _mm_add_epi8(a, b);
		else if constexpr (FS.has(SSE) && xmm_sized<T> && is_f32<S>) return _mm_add_ps(a, b);
		else if constexpr (sizeof(T) > 16) return { add(a.lo(), b.lo()), add(a.hi(), b.hi()) };
		else
		{
			internals::scream();
			SIMD_Vector<S, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = a[i] + b[i];
			return ret;
		}
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> sub(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace internals;
		using namespace meta;
		using T = SIMD_Vector<S, N>;

		if constexpr (FS.has(AVX512_FP16) && zmm_sized<T> && is_fp16<S>) return _mm512_sub_ph(a, b);
		else if constexpr (FS.has(AVX512_FP16) && FS.has(AVX512_VL) && ymm_sized<T> && is_fp16<S>) return _mm256_sub_ph(a, b);
		else if constexpr (FS.has(AVX512_FP16) && FS.has(AVX512_VL) && xmm_sized<T> && is_fp16<S>) return _mm_sub_ph(a, b);
		else if constexpr (!FS.has(AVX512_FP16) && is_fp16<S>) return vcvt<fp16_t>(sub(vcvt<float>(a), vcvt<float>(b)));

		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f64<S>) return _mm512_sub_pd(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f32<S>) return _mm512_sub_ps(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && any_i64<S>) return _mm512_sub_epi64(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && any_i32<S>) return _mm512_sub_epi32(a, b);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_i16<S>) return _mm512_sub_epi16(a, b);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_i8<S>) return _mm512_sub_epi8(a, b);

		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i64<S>) return _mm256_sub_epi64(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i32<S>) return _mm256_sub_epi32(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i16<S>) return _mm256_sub_epi16(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i8<S>) return _mm256_sub_epi8(a, b);
		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f64<S>) return _mm256_sub_pd(a, b);
		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f32<S>) return _mm256_sub_ps(a, b);

		else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_f64<S>) return _mm_sub_pd(a, b);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && any_i64<S>) return _mm_sub_epi64(a, b);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && any_i32<S>) return _mm_sub_epi32(a, b);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && any_i16<S>) return _mm_sub_epi16(a, b);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && any_i8<S>) return _mm_sub_epi8(a, b);
		else if constexpr (FS.has(SSE) && xmm_sized<T> && is_f32<S>) return _mm_sub_ps(a, b);
		else if constexpr (sizeof(T) > 16) return { sub(a.lo(), b.lo()), sub(a.hi(), b.hi()) };
		else
		{
			internals::scream();
			SIMD_Vector<S, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = a[i] - b[i];
			return ret;
		}
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> mul(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using namespace internals;
		using T = SIMD_Vector<S, N>;
		using canon_t = std::conditional_t<(std::is_signed_v<S>), int16_t, uint16_t>;

		if constexpr (FS.has(AVX512_FP16) && zmm_sized<T> && is_fp16<S>) return _mm512_mul_ph(a, b);
		else if constexpr (FS.has(AVX512_FP16) && FS.has(AVX512_VL) && ymm_sized<T> && is_fp16<S>) return _mm256_mul_ph(a, b);
		else if constexpr (FS.has(AVX512_FP16) && FS.has(AVX512_VL) && xmm_sized<T> && is_fp16<S>) return _mm_mul_ph(a, b);
		else if constexpr (!FS.has(AVX512_FP16) && is_fp16<S>) return vcvt<fp16_t>(mul(vcvt<float>(a), vcvt<float>(b)));

		else if constexpr (any_i8<S>) return vcvt<S>(mul(vcvt<canon_t>(a), vcvt<canon_t>(b))); //no 8 bit mul as of last AVX512, so emulate

		else if constexpr (FS.has(AVX512_DQ) && zmm_sized<T> && any_i64<S>) return _mm512_mullo_epi64(a, b);
		else if constexpr (FS.has(AVX512_DQ) && FS.has(AVX512_VL) && ymm_sized<T> && any_i64<S>) return _mm256_mullo_epi64(a, b);
		else if constexpr (FS.has(AVX512_DQ) && FS.has(AVX512_VL) && xmm_sized<T> && any_i64<S>) return _mm_mullo_epi64(a, b);

		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && any_i16<S>) return _mm512_mullo_epi16(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f64<S>) return _mm512_mul_pd(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f32<S>) return _mm512_mul_ps(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && any_i64<S>) return _mm512_mullox_epi64(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && any_i32<S>) return _mm512_mullo_epi32(a, b);

		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i64<S>)
		{
			__m256i p1 = _mm256_mul_epu32(a, b); //alo*blo
			__m256i ahi = _mm256_srli_epi64(a, 32);
			__m256i bhi = _mm256_srli_epi64(b, 32);
			__m256i p2 = _mm256_slli_epi64(_mm256_mul_epu32(a, bhi), 32);
			__m256i p3 = _mm256_slli_epi64(_mm256_mul_epu32(b, ahi), 32);
			return _mm256_add_epi64(p3, _mm256_add_epi64(p1, p2));
		}
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i32<S>) return _mm256_mullo_epi32(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i16<S>) return _mm256_mullo_epi16(a, b);
		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f64<S>) return _mm256_mul_pd(a, b);
		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f32<S>) return _mm256_mul_ps(a, b);

		else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_f64<S>) return _mm_mul_pd(a, b);
		else if constexpr (FS.has(SSE) && xmm_sized<T> && is_f32<S>) return _mm_mul_ps(a, b);
		else if constexpr (FS.has(SSE41) && xmm_sized<T> && any_i32<S>) return _mm_mullo_epi32(a, b);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && any_i16<S>) return _mm_mullo_epi16(a, b);

		else if constexpr (FS.has(SSE2) && xmm_sized<T> && any_i64<S>)
		{
			__m128i p1 = _mm_mul_epu32(a, b); //alo*blo
			__m128i ahi = _mm_srli_epi64(a, 32);
			__m128i bhi = _mm_srli_epi64(b, 32);
			__m128i p2 = _mm_slli_epi64(_mm_mul_epu32(a, bhi), 32);
			__m128i p3 = _mm_slli_epi64(_mm_mul_epu32(b, ahi), 32);
			return _mm_add_epi64(p3, _mm_add_epi64(p1, p2));
		}

		else if constexpr (sizeof(T) > 16) return T{ mul(a.lo(), b.lo()), mul(a.hi(), b.hi()) };

		else
		{
			internals::scream();
			SIMD_Vector<S, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = a[i] * b[i];
			return ret;
		}

	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> div(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		using T = SIMD_Vector<S, N>;

		//TODO: maybe BF16 is good enough to emulate 8 bit integer divison? Check whether it can represent numbers exactly. Only signed?
		if constexpr (any_i32<S>) return vcvt<S>(div(vcvt<double>(a), vcvt<double>(b))); //emulate 32 bit integer division via double precision division
		else if constexpr (any_small_int<S>) return vcvt<S>(div(vcvt<float>(a), vcvt<float>(b))); //emulate small integer division via single precision division

		else if constexpr (FS.has(AVX512_FP16) && zmm_sized<T> && is_fp16<S>) return _mm512_div_ph(a, b);
		else if constexpr (FS.has(AVX512_FP16) && FS.has(AVX512_VL) && ymm_sized<T> && is_fp16<S>) return _mm256_div_ph(a, b);
		else if constexpr (FS.has(AVX512_FP16) && FS.has(AVX512_VL) && xmm_sized<T> && is_fp16<S>) return _mm_div_ph(a, b);
		else if constexpr (!FS.has(AVX512_FP16) && is_fp16<S>) return vcvt<fp16_t>(div(vcvt<float>(a), vcvt<float>(b)));

		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f64<S>) return _mm512_div_pd(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f32<S>) return _mm512_div_ps(a, b);

		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f64<S>) return _mm256_div_pd(a, b);
		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f32<S>) return _mm256_div_ps(a, b);

		else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_f64<S>) return _mm_div_pd(a, b);
		else if constexpr (FS.has(SSE) && xmm_sized<T> && is_f32<S>) return _mm_div_ps(a, b);

		else if constexpr (sizeof(T) > 16) return T{ div(a.lo(), b.lo()), div(a.hi(),b.hi()) };
		//manual 64-bit integer divisions are dog slow, while scalar div is pretty fast (comparatively) on newer CPUs, so it's not insane to fall back to scalar
		//even on older ones, bad division algorithms lose out to scalar.
		//Either that, or more work needs to be done to find a good solution. For now, I don't bother and fall back to scalar
		else
		{
			internals::scream();
			SIMD_Vector<S, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = a[i] / b[i];
			return ret;
		}
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> logic_and(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		using T = SIMD_Vector<S, N>;

		//IDK why this is needed, but if intrinsics exist, I guess. Logical operations don't care about type.
		//Maybe they run on different ports? Anyway, stick to native ones
		if constexpr (!is_f64<S> && !is_f32<S> && !any_int<S>) return vcast<S>(logic_and(vcast<U>(a), vcast<U>(b)));
		else if constexpr (FS.has(AVX512_DQ) && zmm_sized<T> && is_f64<S>) return _mm512_and_pd(a, b);
		else if constexpr (FS.has(AVX512_DQ) && zmm_sized<T> && is_f32<S>) return _mm512_and_ps(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T>) return _mm512_and_si512(vcast<__m512i>(a), vcast<__m512i>(b));

		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_int<S>) return _mm256_and_si256(vcast<__m256i>(a), vcast<__m256i>(b));
		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f64<S>) return _mm256_and_pd(vcast<__m256d>(a), vcast<__m256d>(b));
		else if constexpr (FS.has(AVX) && ymm_sized<T>) return _mm256_and_ps(vcast<__m256>(a), vcast<__m256>(b));

		else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_f64<S>) return _mm_and_pd(vcast<__m128d>(a), vcast<__m128d>(b));
		else if constexpr (FS.has(SSE2) && xmm_sized<T>) return _mm_and_si128(vcast<__m128i>(a), vcast<__m128i>(b));
		else if constexpr (FS.has(SSE) && xmm_sized<T>) return _mm_and_ps(vcast<__m128>(a), vcast<__m128>(b));

		else if constexpr (sizeof(T) > 16) return T{ logic_and(a.lo(),b.lo()), logic_and(a.hi(),b.hi()) };
		else
		{
			internals::scream();
			SIMD_Vector<S, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = std::bit_cast<S>(U(std::bit_cast<U>(a[i]) & std::bit_cast<U>(b[i])));
			return ret;
		}
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> logic_or(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		using T = SIMD_Vector<S, N>;

		//IDK why this is needed, but if intrinsics exist, I guess. Logical operations don't care about type.
		//Maybe they run on different ports? Anyway, stick to native ones
		if constexpr (!is_f64<S> && !is_f32<S> && !any_int<S>) return vcast<S>(logic_or(vcast<U>(a), vcast<U>(b)));
		else if constexpr (FS.has(AVX512_DQ) && zmm_sized<T> && is_f64<S>) return _mm512_or_pd(a, b);
		else if constexpr (FS.has(AVX512_DQ) && zmm_sized<T> && is_f32<S>) return _mm512_or_ps(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T>) return _mm512_or_si512(vcast<__m512i>(a), vcast<__m512i>(b));

		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_int<S>) return _mm256_or_si256(vcast<__m256i>(a), vcast<__m256i>(b));
		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f64<S>) return _mm256_or_pd(vcast<__m256d>(a), vcast<__m256d>(b));
		else if constexpr (FS.has(AVX) && ymm_sized<T>) return _mm256_or_ps(vcast<__m256>(a), vcast<__m256>(b));

		else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_f64<S>) return _mm_or_pd(vcast<__m128d>(a), vcast<__m128d>(b));
		else if constexpr (FS.has(SSE2) && xmm_sized<T>) return _mm_or_si128(vcast<__m128i>(a), vcast<__m128i>(b));
		else if constexpr (FS.has(SSE) && xmm_sized<T>) return _mm_or_ps(vcast<__m128>(a), vcast<__m128>(b));

		else if constexpr (sizeof(T) > 16) return T{ logic_or(a.lo(),b.lo()), logic_or(a.hi(),b.hi()) };

		else
		{
			internals::scream();
			SIMD_Vector<S, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = std::bit_cast<S>(U(std::bit_cast<U>(a[i]) | std::bit_cast<U>(b[i])));
			return ret;
		}
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> logic_xor(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		using T = SIMD_Vector<S, N>;

		//IDK why this is needed, but if intrinsics exist, I guess. Logical operations don't care about type.
		//Maybe they run on different ports? Anyway, stick to native ones
		if constexpr (!is_f64<S> && !is_f32<S> && !any_int<S>) return vcast<S>(logic_xor(vcast<U>(a), vcast<U>(b)));
		else if constexpr (FS.has(AVX512_DQ) && zmm_sized<T> && is_f64<S>) return _mm512_xor_pd(a, b);
		else if constexpr (FS.has(AVX512_DQ) && zmm_sized<T> && is_f32<S>) return _mm512_xor_ps(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T>) return _mm512_xor_si512(vcast<__m512i>(a), vcast<__m512i>(b));

		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_int<S>) return _mm256_xor_si256(vcast<__m256i>(a), vcast<__m256i>(b));
		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f64<S>) return _mm256_xor_pd(vcast<__m256d>(a), vcast<__m256d>(b));
		else if constexpr (FS.has(AVX) && ymm_sized<T>) return _mm256_xor_ps(vcast<__m256>(a), vcast<__m256>(b));

		else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_f64<S>) return _mm_xor_pd(vcast<__m128d>(a), vcast<__m128d>(b));
		else if constexpr (FS.has(SSE2) && xmm_sized<T>) return _mm_xor_si128(vcast<__m128i>(a), vcast<__m128i>(b));
		else if constexpr (FS.has(SSE) && xmm_sized<T>) return _mm_xor_ps(vcast<__m128>(a), vcast<__m128>(b));

		else if constexpr (sizeof(T) > 16) return T{ logic_xor(a.lo(),b.lo()), logic_xor(a.hi(),b.hi()) };
		else
		{
			internals::scream();
			SIMD_Vector<S, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = std::bit_cast<S>(U(std::bit_cast<U>(a[i]) ^ std::bit_cast<U>(b[i])));
			return ret;
		}
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> logic_not(const SIMD_Vector<S, N>& a)
	{
		using namespace meta;
		return logic_xor(a, meta::BitsAllOneF(a));
		/*
		using U = typename ScalarTraits<S>::UintT;

		internals::scream();
		SIMD_Vector<S, N> ret;
		using T = typename meta::ScalarTraits<S>::UintT;
		for (size_t i = 0; i < N; ++i) ret[i] = std::bit_cast<S>(T(~std::bit_cast<T>(a[i])));
		return ret;*/
	}
	template<meta::any_int S, size_t N, meta::any_int I>
	__forceinline SIMD_Vector<S, N> shift_left(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& b)
	{
		using namespace internals;
		using namespace meta;
		using T = SIMD_Vector<S, N>;
		using canon_shift_amount_t = typename ScalarTraits<S>::UintT;

		constexpr bool has_native_16bit_shift = FS.has(AVX512_BW) && (zmm_sized<T> || FS.has(AVX512_VL));
		using routing_t = std::conditional_t<has_native_16bit_shift, uint16_t, uint32_t>;

		//zero-extend small integers, shift and convert back. TODO: There could be a better way?
		if constexpr ((any_i16<S> && !has_native_16bit_shift) || (any_i8<S>)) return vrtrunc<S>(shift_left(vrzext<routing_t>(a), b));
		else if constexpr (!std::is_same_v<I, canon_shift_amount_t>) return shift_left(a, vcvt<canon_shift_amount_t>(b));

		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && any_i16<S>) return _mm512_sllv_epi16(a, b);
		else if constexpr (FS.has(AVX512_BW) && FS.has(AVX512_VL) && ymm_sized<T> && any_i16<S>) return _mm256_sllv_epi16(a, b);
		else if constexpr (FS.has(AVX512_BW) && FS.has(AVX512_VL) && xmm_sized<T> && any_i16<S>) return _mm_sllv_epi16(a, b);
		else if constexpr (FS.has(AVX512_BW) && FS.has(AVX512_VL) && any_i8<S>) return vrtrunc<S>(shift_left(vrzext<uint16_t>(a), b));
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && any_i64<S>) return _mm512_sllv_epi64(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && any_i32<S>) return _mm512_sllv_epi32(a, b);

		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i64<S>) return _mm256_sllv_epi64(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i32<S>) return _mm256_sllv_epi32(a, b);
		else if constexpr (FS.has(AVX2) && xmm_sized<T> && any_i64<S>) return _mm_sllv_epi64(a, b); //no shifts in SSE!
		else if constexpr (FS.has(AVX2) && xmm_sized<T> && any_i32<S>) return _mm_sllv_epi32(a, b);

		else if constexpr (sizeof(T) > 16) return T{ shift_left(a.lo(),b.lo()), shift_left(a.hi(),b.hi()) };
		else
		{
			internals::scream();
			SIMD_Vector<S, N> ret;
			//using T = typename concepts::same_size_uint_t<S>::type;
			for (size_t i = 0; i < N; ++i) ret[i] = a[i] << b[i];
			return ret;
		}
	}

	template<meta::any_int S, size_t N, meta::any_int I>
	__forceinline SIMD_Vector<S, N> shift_right(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& b)
	{
		using namespace internals;
		using namespace meta;
		using T = SIMD_Vector<S, N>;
		using canon_shift_amount_t = typename ScalarTraits<S>::UintT;

		constexpr bool has_native_16bit_shift = FS.has(AVX512_BW) && (zmm_sized<T> || FS.has(AVX512_VL));
		using same_signedness_int16_t = std::conditional_t<std::is_signed_v<S>, int16_t, uint16_t>;
		using same_signedness_int32_t = std::conditional_t<std::is_signed_v<S>, int32_t, uint32_t>;
		using routing_t = std::conditional_t<has_native_16bit_shift, same_signedness_int16_t, same_signedness_int32_t>;

		//zero-extend small integers, shift and convert back. TODO: There could be a better way?
		if constexpr ((any_i16<S> && !has_native_16bit_shift) || (any_i8<S>)) return vcvt<S>(shift_right(vcvt<routing_t>(a), b));
		else if constexpr (!std::is_same_v<I, canon_shift_amount_t>) return shift_right(a, vcvt<canon_shift_amount_t>(b));

		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_u16<S>) return _mm512_srlv_epi16(a, b);
		else if constexpr (FS.has(AVX512_BW) && FS.has(AVX512_VL) && ymm_sized<T> && is_u16<S>) return _mm256_srlv_epi16(a, b);
		else if constexpr (FS.has(AVX512_BW) && FS.has(AVX512_VL) && xmm_sized<T> && is_u16<S>) return _mm_srlv_epi16(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_u64<S>) return _mm512_srlv_epi64(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_u32<S>) return _mm512_srlv_epi32(a, b);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_i16<S>) return _mm512_srav_epi16(a, b);
		else if constexpr (FS.has(AVX512_BW) && FS.has(AVX512_VL) && ymm_sized<T> && is_i16<S>) return _mm256_srav_epi16(a, b);
		else if constexpr (FS.has(AVX512_BW) && FS.has(AVX512_VL) && xmm_sized<T> && is_i16<S>) return _mm_srav_epi16(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_i64<S>) return _mm512_srav_epi64(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_i32<S>) return _mm512_srav_epi32(a, b);

		else if constexpr (FS.has(AVX2) && ymm_sized<T> && is_u64<S>) return _mm256_srlv_epi64(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && is_u32<S>) return _mm256_srlv_epi32(a, b);
		else if constexpr (FS.has(AVX2) && xmm_sized<T> && is_u64<S>) return _mm_srlv_epi64(a, b); //no shifts in SSE!
		else if constexpr (FS.has(AVX2) && xmm_sized<T> && is_u32<S>) return _mm_srlv_epi32(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && is_i64<S>) return _mm256_srav_epi64(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && is_i32<S>) return _mm256_srav_epi32(a, b);
		else if constexpr (FS.has(AVX2) && xmm_sized<T> && is_i64<S>) return _mm_srav_epi64(a, b); //no shifts in SSE!
		else if constexpr (FS.has(AVX2) && xmm_sized<T> && is_i32<S>) return _mm_srav_epi32(a, b);

		else if constexpr (sizeof(T) > 16) return { shift_right(a.lo(), b.lo()), shift_right(a.hi(), b.hi()) };
		else
		{
			internals::scream();
			SIMD_Vector<S, N> ret;
			//using T = typename concepts::same_size_uint_t<S>::type;
			for (size_t i = 0; i < N; ++i) ret[i] = a[i] >> b[i];
			return ret;
		}
	}

	template<size_t A, meta::any_int S, size_t N>
	SIMD_Vector<S, N> shift_left(const SIMD_Vector<S, N>& a)
	{
		using namespace internals;
		using namespace meta;
		using T = SIMD_Vector<S, N>;

		if constexpr (A == 0) return a;
		else if constexpr (A >= sizeof(S) * 8) return T(0);
		else if constexpr (any_i8<S>)
		{
			//TODO: this will fail on vectors < 4 sized. Same with shift_right
			auto interm = shift_left<A>(vcast<uint32_t>(a));
			constexpr uint32_t andc = ((1 << A) - 1) & 0xFF;
			constexpr uint32_t andc2 = (andc << 8) | (andc << 16) | (andc << 24);
			return vcast<T>(interm & ~andc2); //remove bits bleeding over neighboring bytes
		}
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && any_i16<S>) return _mm512_slli_epi16(a, A);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && any_i32<S>) return _mm512_slli_epi32(a, A);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && any_i64<S>) return _mm512_slli_epi64(a, A);

		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i16<S>) return _mm256_slli_epi16(a, A);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i32<S>) return _mm256_slli_epi32(a, A);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i64<S>) return _mm256_slli_epi64(a, A);

		else if constexpr (FS.has(SSE2) && xmm_sized<T> && any_i16<S>) return _mm_slli_epi16(a, A);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && any_i32<S>) return _mm_slli_epi32(a, A);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && any_i64<S>) return _mm_slli_epi64(a, A);
		else if constexpr (sizeof(T) > 16) return { shift_left<A>(a.lo()), shift_left<A>(a.hi()) };
		else
		{
			T ret;
			for (size_t i = 0; i < N; ++i) ret[i] = a[i] << A;
			return ret;
		}
	}

	template<size_t A, meta::any_int S, size_t N>
	SIMD_Vector<S, N> shift_right(const SIMD_Vector<S, N>& a)
	{
		using namespace internals;
		using namespace meta;
		using T = SIMD_Vector<S, N>;

		if constexpr (A == 0) return a;
		else if constexpr (A >= sizeof(S) * 8) return T(0);
		else if constexpr (any_i8<S>)
		{
			auto interm = shift_right<A>(vcast<uint32_t>(a)); //everything has 32-bit shifts!
			constexpr uint32_t fin = ((1 << (8 - A)) - 1) & 0xFF;
			constexpr uint32_t andc2 = (fin << 0) | (fin << 8) | (fin << 16) | (fin << 24); //zero-out A most significant bits in each byte, removing bits shifted in from neighbors
			
			auto shiftedInZeros = interm & andc2;
			if constexpr (is_u8<S>) return vcast<T>(shiftedInZeros);
			else
			{
				mask_t<S, N> zcmp = vcast<int8_t>(a) < 0;
				auto shiftedInOnes = vcast<T>(shiftedInZeros | ~andc2); //force shifted in bits to ones for initially negative inputs
				return mask_mov(vcast<T>(shiftedInZeros), zcmp, vcast<T>(shiftedInOnes));
			}
		}
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_u16<S>) return _mm512_srli_epi16(a, A);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_u32<S>) return _mm512_srli_epi32(a, A);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_u64<S>) return _mm512_srli_epi64(a, A);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_i16<S>) return _mm512_srai_epi16(a, A);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_i32<S>) return _mm512_srai_epi32(a, A);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_i64<S>) return _mm512_srai_epi64(a, A);

		else if constexpr (FS.has(AVX2) && ymm_sized<T> && is_u16<S>) return _mm256_srli_epi16(a, A);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && is_u32<S>) return _mm256_srli_epi32(a, A);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && is_u64<S>) return _mm256_srli_epi64(a, A);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && is_i16<S>) return _mm256_srai_epi16(a, A);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && is_i32<S>) return _mm256_srai_epi32(a, A);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && is_i64<S>) return _mm256_srai_epi64(a, A);

		else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_u16<S>) return _mm_srli_epi16(a, A);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_u32<S>) return _mm_srli_epi32(a, A);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_u64<S>) return _mm_srli_epi64(a, A);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_i16<S>) return _mm_srai_epi16(a, A);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_i32<S>) return _mm_srai_epi32(a, A);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_i64<S>) return _mm_srai_epi64(a, A);
		else if constexpr (sizeof(T) > 16) return { shift_right<A>(a.lo()), shift_right<A>(a.hi()) };
		else
		{
			T ret;
			for (size_t i = 0; i < N; ++i) ret[i] = a[i] << A;
			return ret;
		}
	}

	template<typename S, size_t N, meta::any_int I>
	__forceinline SIMD_Vector<S, N> permx(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& indBase)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		using canon_t = typename ScalarTraits<S>::UintT;
		using T = SIMD_Vector<S, N>;

		//Native permutexvar implementations wrap around themselves, but on truncated vectors, they still follow XMM wrapping, which pulls in garbage. 
		//For example, if permx is called on SIMD_Vector<uint8_t, 4>, API documents wrapping around 4, but native permutex/shuffle will still wrap around 16 bytes.
		//In this case, if ind is 5, garbage will be pulled in, while API documents that it will wrap to 1 and stay within input's bounds.
		SIMD_Vector<I, N> ind = indBase;
		if constexpr (sizeof(T) < 16) ind &= N - 1;

		if constexpr (!is_f64<S> && !is_f32<S> && !any_int<S>) return vcast<S>(permx(vcast<U>(a), ind));
		//TODO: some workaround for 127+ 8-bit perms?
		else if constexpr (sizeof(I) != sizeof(S)) return permx(a, vcvt<canon_t>(ind));
		else if constexpr (FS.has(AVX512_VBMI) && zmm_sized<T> && any_i8<S>) return _mm512_permutexvar_epi8(ind, a);
		else if constexpr (FS.has(AVX512_VBMI) && FS.has(AVX512_VL) & ymm_sized<T> && any_i8<S>) return _mm256_permutexvar_epi8(ind, a);
		else if constexpr (FS.has(AVX512_VBMI) && FS.has(AVX512_VL) & xmm_sized<T> && any_i8<S>) return _mm_permutexvar_epi8(ind, a);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && any_i16<S>) return _mm512_permutexvar_epi16(ind, a);
		else if constexpr (FS.has(AVX512_BW) && FS.has(AVX512_VL) && ymm_sized<T> && any_i16<S>) return _mm256_permutexvar_epi16(ind, a);
		else if constexpr (FS.has(AVX512_BW) && FS.has(AVX512_VL) && xmm_sized<T> && any_i16<S>) return _mm_permutexvar_epi16(ind, a);
		else if constexpr (FS.has(AVX512_BW) && sizeof(S) == 1) return vrtrunc<S>(permx(vrzext<uint16_t>(a), ind));

		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f64<S>) return _mm512_permutexvar_pd(ind, a);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f32<S>) return _mm512_permutexvar_ps(ind, a);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && any_i64<S>) return _mm512_permutexvar_epi64(ind, a);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && any_i32<S>) return _mm512_permutexvar_epi32(ind, a);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && any_i16<S>) return vrtrunc<S>(permx(vrzext<uint32_t>(a), ind)); //probably worth it to extend 16->32, but 8->32 may be better with pshufb breakup? TODO: test if it's good

		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && is_f64<S>) return _mm256_permutexvar_pd(ind, a);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && is_f32<S>) return _mm256_permutexvar_ps(ind, a);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && any_i64<S>) return _mm256_permutexvar_epi64(ind, a);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && any_i32<S>) return _mm256_permutexvar_epi32(ind, a); //although this (and ps version) exist in AVX2, it could make compiler's job of optimizing permutex+mask_mov into mask_permutex easier

		//TODO: maybe add _mm version via permx2 for AVX512F?
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && is_f32<S>) return _mm256_permutevar8x32_ps(a, ind);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i32<S>) return _mm256_permutevar8x32_epi32(a, ind);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && sizeof(S) == 8) //emulate with 4-byte perm. //TODO: maybe allowing it to SSSE3 or AVX is better?
		{
			//TODO: domain crossing vs 1 extra instruction byte with _mm256_shuffle_epi32?
			using X = std::conditional_t<is_f64<S>, float, uint32_t>;
			auto a32 = vcast<X>(a);
			__m256i ind32 = _mm256_castps_si256(_mm256_moveldup_ps(_mm256_castsi256_ps(ind)));//duplicate each low 32-bits of 64-bit index element into 32-bit lanes. Wrap around and power of 2 vector size limitation allow this to work.
			auto perm = permx(a32, _mm256_or_si256(_mm256_slli_epi32(ind32, 1), _mm256_setr_epi32(0, 1, 0, 1, 0, 1, 0, 1))); //index is multiplied by 2 and added to alterating 0, 1, emulating 64 bit behavior
			return vcast<S>(perm);
		}
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i16<S>)
		{
			__m256i ind1 = _mm256_srli_epi32(ind, 1);
			__m256i ind2 = _mm256_srli_epi32(ind, 17);
			__m256i p1 = _mm256_permutevar8x32_epi32(a, ind1); //perm even indexed words
			__m256i p2 = _mm256_permutevar8x32_epi32(a, ind2); //perm odd indexed words
			//now p1 and p2 contain our wanted elements + garbage neighbor, since permute granularity is 2x the wanted one
			//also, we discarded LSB of the original index, and it's needed to reconstruct the result
			//if LSB was 0, it means than upper 16 bits are garbage (wanted element resides in lower half)
			//if LSB was 1, it means than lower 16 bits are garbage (wanted element resides in upper half)
			//thus: 
			//if (lsb == 0 && p1) -> element resides in lower and supposed to be there, pass through
			//if (lsb == 1 && p1) -> element resides in upper, but wants to be in lower, shift right by 16
			//if (lsb == 0 && p2) -> element resides in lower, but wants upper -> shift left by 16
			//if (lsb == 1 && p2) -> element resides in upper, wants upper -> pass through

			//__m256i p1a = _mm256_and_si256(p1, _mm256_set1_epi32(0xFFFF));
			__m256i p1s = _mm256_srli_epi32(p1, 16);
			__m256i p2s = _mm256_slli_epi32(p2, 16);
			//__m256i p2a = _mm256_and_si256(p2, _mm256_set1_epi32(0xFFFF0000));
			__m256 bmask1 = _mm256_castsi256_ps(_mm256_slli_epi32(ind, 31));
			__m256 bmask2 = _mm256_castsi256_ps(_mm256_slli_epi32(ind, 15));

			__m256 b1 = _mm256_blendv_ps(_mm256_castsi256_ps(p1), _mm256_castsi256_ps(p1s), bmask1);
			__m256 b2 = _mm256_blendv_ps(_mm256_castsi256_ps(p2s), _mm256_castsi256_ps(p2), bmask2);
			return _mm256_blend_epi16(_mm256_castps_si256(b1), _mm256_castps_si256(b2), 0b10101010);
		}
		else if constexpr (FS.has(AVX) && xmm_sized<T> && sizeof(S) == 4) return T::fromBits(_mm_permutevar_ps(vcast<__m128>(a), ind));
		else if constexpr (FS.has(AVX) && xmm_sized<T> && sizeof(S) == 8) return T::fromBits(_mm_permutevar_pd(vcast<__m128d>(a), shift_left<1>(ind))); //TY intel for laying this trap for me. for some reason, it takes bit 1 and 65, NOT 0 or 64!!! While ps version is actually sane. lol.

		//TODO: these may break with >127 bytes. Also check if they work at all
		else if constexpr (FS.has(SSSE3) && xmm_sized<T> && sizeof(S) == 1) return _mm_shuffle_epi8(a, ind & 0x7F); //discard sign bit to avoid unwanted zero-masking
		else if constexpr (FS.has(SSSE3) && xmm_sized<T> && sizeof(S) == 2)
		{
			__m128i ind2 = _mm_slli_epi16(ind, 1);
			__m128i db = _mm_shuffle_epi8(ind2, _mm_setr_epi8(0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14)); //duplicate low byte of each word
			__m128i ind3 = _mm_or_si128(db, _mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1));
			__m128i ind4 = _mm_and_si128(ind3, _mm_set1_epi8(0x7F));
			return T::fromBits(_mm_shuffle_epi8(vcast<__m128i>(a), ind4));
		}
		else if constexpr (FS.has(SSSE3) && xmm_sized<T> && sizeof(S) == 4)
		{
			__m128i ind2 = _mm_slli_epi32(ind, 2);
			__m128i db = _mm_shuffle_epi8(ind2, _mm_setr_epi8(0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12)); //duplicate low byte of each dword
			__m128i ind3 = _mm_or_si128(db, _mm_setr_epi8(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3));
			__m128i ind4 = _mm_and_si128(ind3, _mm_set1_epi8(0x7F));
			return T::fromBits(_mm_shuffle_epi8(vcast<__m128i>(a), ind4));
		}
		else if constexpr (FS.has(SSSE3) && xmm_sized<T> && sizeof(S) == 8)
		{
			__m128i ind2 = _mm_slli_epi64(ind, 3);
			__m128i db = _mm_shuffle_epi8(ind2, _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8)); //duplicate low byte of each qword
			__m128i ind3 = _mm_or_si128(db, _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7));
			__m128i ind4 = _mm_and_si128(ind3, _mm_set1_epi8(0x7F));
			return T::fromBits(_mm_shuffle_epi8(vcast<__m128i>(a), ind4));
		}
		else if constexpr (sizeof(T) > 16)
		{
			auto alo = a.lo();
			auto ahi = a.hi();
			return T{ permx2(alo, ahi, ind.lo()), permx2(alo, ahi, ind.hi()) };
		}
		else
		{
			internals::scream();
			SIMD_Vector<S, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = a[ind[i] & (N - 1)];
			return ret;
		}

	}

	template<typename S, size_t N, meta::any_int I>
	__forceinline SIMD_Vector<S, N> permx2(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b, const SIMD_Vector<I, N>& indBase)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		using canon_t = typename ScalarTraits<S>::UintT;
		using T = SIMD_Vector<S, N>;

		SIMD_Vector<I, N> ind = indBase;
		auto emulation = [&]() {
			T pa = permx(a, ind);
			T pb = permx(b, ind);
			return mask_mov(pb, (ind & (2 * N - 1)) < N, pa);
			};

		//Native permutex2var implementations wrap around themselves, but on truncated vectors, it's a whole other can of worms:
		//1) Native still wraps only around 2*N - 1 for XMM size, not truncated size. Since vectors are automatically expanded, it will pull garbage for indices outside -N+1..N-1 range
		//2) The criterion for picking table b is also XMM-sized, meaning it compares for 16 bytes/sizeof(S).
		//Thus, in a scenario: permx2<uint8_t, 4>({0,1,2,3}, {4,5,6,7}, {3,6,1,0}) API documents result: {3,6,1,0},
		//But non-scalar implementation will act as: N == 16, so return result is: {3, garbage from expanded A, 1, 0} (b never even considered).
		//This if block fixes it		
		if constexpr (sizeof(T) < 16)
		{
			ind &= 2 * N - 1;
			return emulation();
		}
		else if constexpr (!is_f64<S> && !is_f32<S> && !any_int<S>) return vcast<S>(permx2(vcast<U>(a), vcast<U>(b), ind));
		else if constexpr (sizeof(I) != sizeof(S)) return permx2(a, b, vcvt<canon_t>(ind));

		else if constexpr (FS.has(AVX512_VBMI) && zmm_sized<T> && any_i8<S>) return _mm512_permutex2var_epi8(a, ind, b);
		else if constexpr (FS.has(AVX512_VBMI) && FS.has(AVX512_VL) && ymm_sized<T> && any_i8<S>) return _mm256_permutex2var_epi8(a, ind, b);
		else if constexpr (FS.has(AVX512_VBMI) && FS.has(AVX512_VL) && xmm_sized<T> && any_i8<S>) return _mm_permutex2var_epi8(a, ind, b);

		else if constexpr (any_i8<S>) return vcvt<S>(permx2(vcvt<uint16_t>(a), vcvt<uint16_t>(b), ind));
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && any_i16<S>) return _mm512_permutex2var_epi16(a, ind, b);
		else if constexpr (FS.has(AVX512_BW) && FS.has(AVX512_VL) && ymm_sized<T> && any_i16<S>) return _mm256_permutex2var_epi16(a, ind, b);
		else if constexpr (FS.has(AVX512_BW) && FS.has(AVX512_VL) && xmm_sized<T> && any_i16<S>) return _mm_permutex2var_epi16(a, ind, b);

		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f64<S>) return _mm512_permutex2var_pd(a, ind, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f32<S>) return _mm512_permutex2var_ps(a, ind, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && any_i64<S>) return _mm512_permutex2var_epi64(a, ind, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && any_i32<S>) return _mm512_permutex2var_epi32(a, ind, b);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && is_f64<S>) return _mm256_permutex2var_pd(a, ind, b);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && is_f32<S>) return _mm256_permutex2var_ps(a, ind, b);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && any_i64<S>) return _mm256_permutex2var_epi64(a, ind, b);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && any_i32<S>) return _mm256_permutex2var_epi32(a, ind, b);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && xmm_sized<T> && is_f64<S>) return _mm_permutex2var_pd(a, ind, b);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && xmm_sized<T> && is_f32<S>) return _mm_permutex2var_ps(a, ind, b);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && xmm_sized<T> && any_i64<S>) return _mm_permutex2var_epi64(a, ind, b);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && xmm_sized<T> && any_i32<S>) return _mm_permutex2var_epi32(a, ind, b);
		else if constexpr (true || sizeof(T) > 16) return emulation(); //TODO: seems to work, but very sus. Although, what else to do?
		else
		{
			internals::scream();
			SIMD_Vector<S, N> ret;
			for (size_t i = 0; i < N; ++i)
			{
				auto j = ind[i] & (2 * N - 1);
				ret[i] = j < N ? a[j] : b[j - N];
			}
			return ret;
		}
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<float, N> sqrtf(const SIMD_Vector<S, N>& a)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		using T = SIMD_Vector<S, N>;

		if constexpr (!is_f32<S>) return sqrtf(vcvt<float>(a));
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T>) return _mm512_sqrt_ps(a);
		else if constexpr (FS.has(AVX) && ymm_sized<T>) return _mm256_sqrt_ps(a);
		else if constexpr (FS.has(SSE) && xmm_sized<T>) return _mm_sqrt_ps(a);
		else if constexpr (sizeof(T) > 16) return T{ sqrtf(a.lo()), sqrtf(a.hi()) };
		else
		{
			internals::scream();
			SIMD_Vector<float, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = std::sqrt(float(a[i]));
			return ret;
		}
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<double, N> sqrtd(const SIMD_Vector<S, N>& a)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		using T = SIMD_Vector<S, N>;

		if constexpr (!is_f64<S>) return sqrtd(vcvt<double>(a));
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T>) return _mm512_sqrt_pd(a);
		else if constexpr (FS.has(AVX) && ymm_sized<T>) return _mm256_sqrt_pd(a);
		else if constexpr (FS.has(SSE2) && xmm_sized<T>) return _mm_sqrt_pd(a);
		else if constexpr (sizeof(T) > 16) return T{ sqrtd(a.lo()), sqrtd(a.hi()) };
		else
		{
			internals::scream();
			SIMD_Vector<double, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = std::sqrt(double(a[i]));
			return ret;
		}
	}


	template<meta::IsScalarType To, size_t N, meta::IsScalarType From>
	__forceinline SIMD_Vector<To, N> vcvt(const SIMD_Vector<From, N>& a)
	{
		using namespace meta;
		using namespace internals;
		using TV = SIMD_Vector<To, N>;
		using FV = SIMD_Vector<From, N>;
		constexpr size_t MaxSize = std::max(sizeof(TV), sizeof(FV));

		//Route all FP16 conversions to it's only friend - float
		if constexpr ((is_fp16<From> && !is_f32<To>) || (!is_f32<From> && is_fp16<To>)) return vcvt<To>(vcvt<float>(a));
		else if constexpr (sizeof(From) == sizeof(To) && any_int<From> && any_int<To>) return vcast<To>(a);
		//Route small int to FP through their 32 bit types of same signedness
		else if constexpr (any_small_int<From> && !any_int<To>)
		{
			using interm_t = std::conditional_t<(std::is_signed_v<From>), int32_t, uint32_t>;
			return vcvt<To>(vcvt<interm_t>(a));
		}
		else if constexpr (!any_int<From> && any_small_int<To>)
		{
			using interm_t = std::conditional_t<(std::is_signed_v<To>), int32_t, uint32_t>;
			return vcvt<To>(vcvt<interm_t>(a));
		}
		else if constexpr (FS.has(AVX512_DQ) && is_zmm_size(MaxSize) && is_i64<From> && is_f64<To>) return _mm512_cvtepi64_pd(a);
		else if constexpr (FS.has(AVX512_DQ) && is_zmm_size(MaxSize) && is_u64<From> && is_f64<To>) return _mm512_cvtepu64_pd(a);
		else if constexpr (FS.has(AVX512_DQ) && is_zmm_size(MaxSize) && is_i64<From> && is_f32<To>) return _mm512_cvtepi64_ps(a);
		else if constexpr (FS.has(AVX512_DQ) && is_zmm_size(MaxSize) && is_u64<From> && is_f32<To>) return _mm512_cvtepu64_ps(a);
		else if constexpr (FS.has(AVX512_DQ) && is_zmm_size(MaxSize) && is_f64<From> && is_i64<To>) return _mm512_cvttpd_epi64(a);
		else if constexpr (FS.has(AVX512_DQ) && is_zmm_size(MaxSize) && is_f64<From> && is_u64<To>) return _mm512_cvttpd_epu64(a);
		else if constexpr (FS.has(AVX512_DQ) && is_zmm_size(MaxSize) && is_f32<From> && is_i64<To>) return _mm512_cvttps_epi64(a);
		else if constexpr (FS.has(AVX512_DQ) && is_zmm_size(MaxSize) && is_f32<From> && is_u64<To>) return _mm512_cvttps_epu64(a);
		else if constexpr (FS.has(AVX512_DQ) && FS.has(AVX512_VL) && is_ymm_size(MaxSize) && is_i64<From> && is_f64<To>) return _mm256_cvtepi64_pd(a);
		else if constexpr (FS.has(AVX512_DQ) && FS.has(AVX512_VL) && is_ymm_size(MaxSize) && is_u64<From> && is_f64<To>) return _mm256_cvtepu64_pd(a);
		else if constexpr (FS.has(AVX512_DQ) && FS.has(AVX512_VL) && is_ymm_size(MaxSize) && is_i64<From> && is_f32<To>) return _mm256_cvtepi64_ps(a);
		else if constexpr (FS.has(AVX512_DQ) && FS.has(AVX512_VL) && is_ymm_size(MaxSize) && is_u64<From> && is_f32<To>) return _mm256_cvtepu64_ps(a);
		else if constexpr (FS.has(AVX512_DQ) && FS.has(AVX512_VL) && is_ymm_size(MaxSize) && is_f64<From> && is_i64<To>) return _mm256_cvttpd_epi64(a);
		else if constexpr (FS.has(AVX512_DQ) && FS.has(AVX512_VL) && is_ymm_size(MaxSize) && is_f64<From> && is_u64<To>) return _mm256_cvttpd_epu64(a);
		else if constexpr (FS.has(AVX512_DQ) && FS.has(AVX512_VL) && is_ymm_size(MaxSize) && is_f32<From> && is_i64<To>) return _mm256_cvttps_epi64(a);
		else if constexpr (FS.has(AVX512_DQ) && FS.has(AVX512_VL) && is_ymm_size(MaxSize) && is_f32<From> && is_u64<To>) return _mm256_cvttps_epu64(a);
		else if constexpr (FS.has(AVX512_DQ) && FS.has(AVX512_VL) && is_xmm_size(MaxSize) && is_i64<From> && is_f64<To>) return _mm_cvtepi64_pd(a);
		else if constexpr (FS.has(AVX512_DQ) && FS.has(AVX512_VL) && is_xmm_size(MaxSize) && is_u64<From> && is_f64<To>) return _mm_cvtepu64_pd(a);
		else if constexpr (FS.has(AVX512_DQ) && FS.has(AVX512_VL) && is_xmm_size(MaxSize) && is_i64<From> && is_f32<To>) return _mm_cvtepi64_ps(a);
		else if constexpr (FS.has(AVX512_DQ) && FS.has(AVX512_VL) && is_xmm_size(MaxSize) && is_u64<From> && is_f32<To>) return _mm_cvtepu64_ps(a);
		else if constexpr (FS.has(AVX512_DQ) && FS.has(AVX512_VL) && is_xmm_size(MaxSize) && is_f64<From> && is_i64<To>) return _mm_cvttpd_epi64(a);
		else if constexpr (FS.has(AVX512_DQ) && FS.has(AVX512_VL) && is_xmm_size(MaxSize) && is_f64<From> && is_u64<To>) return _mm_cvttpd_epu64(a);
		else if constexpr (FS.has(AVX512_DQ) && FS.has(AVX512_VL) && is_xmm_size(MaxSize) && is_f32<From> && is_i64<To>) return _mm_cvttps_epi64(a);
		else if constexpr (FS.has(AVX512_DQ) && FS.has(AVX512_VL) && is_xmm_size(MaxSize) && is_f32<From> && is_u64<To>) return _mm_cvttps_epu64(a);

		else if constexpr (FS.has(AVX512_BW) && is_zmm_size(MaxSize) && any_i16<From> && any_i8<To>) return _mm512_cvtepi16_epi8(a);
		else if constexpr (FS.has(AVX512_BW) && is_zmm_size(MaxSize) && is_i8<From> && any_i16<To>) return _mm512_cvtepi8_epi16(a);
		else if constexpr (FS.has(AVX512_BW) && is_zmm_size(MaxSize) && is_u8<From> && any_i16<To>) return _mm512_cvtepu8_epi16(a);
		else if constexpr (FS.has(AVX512_BW) && FS.has(AVX512_VL) && is_ymm_size(MaxSize) && any_i16<From> && any_i8<To>) return _mm256_cvtepi16_epi8(a);
		else if constexpr (FS.has(AVX512_BW) && FS.has(AVX512_VL) && is_xmm_size(MaxSize) && any_i16<From> && any_i8<To>) return _mm_cvtepi16_epi8(a);

		//from double
		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && is_f64<From> && is_i32<To>) return _mm512_cvttpd_epi32(a);
		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && is_f64<From> && is_u32<To>) return _mm512_cvttpd_epu32(a);
		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && is_f64<From> && is_f32<To>) return _mm512_cvtpd_ps(a);

		//from float
		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && is_f32<From> && is_i32<To>) return _mm512_cvttps_epi32(a);
		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && is_f32<From> && is_u32<To>) return _mm512_cvttps_epu32(a);
		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && is_f32<From> && is_f64<To>) return _mm512_cvtps_pd(a);

		//from i64
		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && any_i64<From> && any_i32<To>) return _mm512_cvtepi64_epi32(a);
		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && any_i64<From> && any_i16<To>) return _mm512_cvtepi64_epi16(a);
		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && any_i64<From> && any_i8<To>) return _mm512_cvtepi64_epi8(a);

		//from i32
		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && is_i32<From> && is_f64<To>) return _mm512_cvtepi32_pd(a);
		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && is_i32<From> && is_f32<To>) return _mm512_cvtepi32_ps(a);
		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && is_i32<From> && any_i64<To>) return _mm512_cvtepi32_epi64(a);
		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && is_i32<From> && any_i16<To>) return _mm512_cvtepi32_epi16(a);
		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && is_i32<From> && any_i8<To>) return _mm512_cvtepi32_epi8(a);

		//from u32
		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && is_u32<From> && is_f64<To>) return _mm512_cvtepu32_pd(a);
		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && is_u32<From> && is_f32<To>) return _mm512_cvtepu32_ps(a);
		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && is_u32<From> && any_i64<To>) return _mm512_cvtepu32_epi64(a);

		//from 16 bit ints
		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && is_i16<From> && any_i64<To>) return _mm512_cvtepi16_epi64(a);
		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && is_i16<From> && any_i32<To>) return _mm512_cvtepi16_epi32(a);
		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && is_u16<From> && any_i64<To>) return _mm512_cvtepu16_epi64(a);
		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && is_u16<From> && any_i32<To>) return _mm512_cvtepu16_epi32(a);

		//from 8 bit ints
		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && is_i8<From> && any_i64<To>) return _mm512_cvtepi8_epi64(a);
		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && is_i8<From> && any_i32<To>) return _mm512_cvtepi8_epi32(a);
		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && is_u8<From> && any_i64<To>) return _mm512_cvtepu8_epi64(a);
		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && is_u8<From> && any_i32<To>) return _mm512_cvtepu8_epi32(a);

		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && is_fp16<From> && is_f32<To>) return _mm512_cvtph_ps(a);
		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && is_f32<From> && is_fp16<To>) return _mm512_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT);

		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && is_ymm_size(MaxSize) && is_f64<From> && is_u32<To>) return _mm256_cvttpd_epu32(a);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && is_ymm_size(MaxSize) && is_f32<From> && is_u32<To>) return _mm256_cvttps_epu32(a);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && is_ymm_size(MaxSize) && any_i64<From> && any_i32<To>) return _mm256_cvtepi64_epi32(a);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && is_ymm_size(MaxSize) && any_i64<From> && any_i16<To>) return _mm256_cvtepi64_epi16(a);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && is_ymm_size(MaxSize) && any_i64<From> && any_i8<To>) return _mm256_cvtepi64_epi8(a);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && is_ymm_size(MaxSize) && any_i32<From> && any_i16<To>) return _mm256_cvtepi32_epi16(a);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && is_ymm_size(MaxSize) && any_i32<From> && any_i8<To>) return _mm256_cvtepi32_epi8(a);

		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && is_xmm_size(MaxSize) && is_f64<From> && is_u32<To>) return _mm_cvttpd_epu32(a);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && is_xmm_size(MaxSize) && is_f32<From> && is_u32<To>) return _mm_cvttps_epu32(a);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && is_xmm_size(MaxSize) && any_i64<From> && any_i32<To>) return _mm_cvtepi64_epi32(a);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && is_xmm_size(MaxSize) && any_i64<From> && any_i16<To>) return _mm_cvtepi64_epi16(a);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && is_xmm_size(MaxSize) && any_i64<From> && any_i8<To>) return _mm_cvtepi64_epi8(a);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && is_xmm_size(MaxSize) && any_i32<From> && any_i16<To>) return _mm_cvtepi32_epi16(a);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && is_xmm_size(MaxSize) && any_i32<From> && any_i8<To>) return _mm_cvtepi32_epi8(a);

		else if constexpr (FS.has(F16C) && is_ymm_size(MaxSize) && is_f32<From> && is_fp16<To>) return _mm256_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT);
		else if constexpr (FS.has(F16C) && is_xmm_size(MaxSize) && is_f32<From> && is_fp16<To>) return _mm_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT);
		else if constexpr (FS.has(F16C) && is_ymm_size(MaxSize) && is_fp16<From> && is_f32<To>) return _mm256_cvtph_ps(a);
		else if constexpr (FS.has(F16C) && is_xmm_size(MaxSize) && is_fp16<From> && is_f32<To>) return _mm_cvtph_ps(a);

		else if constexpr (FS.has(AVX2) && is_ymm_size(MaxSize) && is_i16<From> && any_i32<To>) return _mm256_cvtepi16_epi32(a);
		else if constexpr (FS.has(AVX2) && is_ymm_size(MaxSize) && is_i16<From> && any_i64<To>) return _mm256_cvtepi16_epi64(a);
		else if constexpr (FS.has(AVX2) && is_ymm_size(MaxSize) && is_i32<From> && any_i64<To>) return _mm256_cvtepi32_epi64(a);
		else if constexpr (FS.has(AVX) && is_ymm_size(MaxSize) && is_i32<From> && is_f64<To>) return _mm256_cvtepi32_pd(a);
		else if constexpr (FS.has(AVX) && is_ymm_size(MaxSize) && is_i32<From> && is_f32<To>) return _mm256_cvtepi32_ps(a);
		else if constexpr (FS.has(AVX2) && is_ymm_size(MaxSize) && is_i8<From> && any_i16<To>) return _mm256_cvtepi8_epi16(a);
		else if constexpr (FS.has(AVX2) && is_ymm_size(MaxSize) && is_i8<From> && any_i32<To>) return _mm256_cvtepi8_epi32(a);
		else if constexpr (FS.has(AVX2) && is_ymm_size(MaxSize) && is_i8<From> && any_i64<To>) return _mm256_cvtepi8_epi64(a);
		else if constexpr (FS.has(AVX2) && is_ymm_size(MaxSize) && is_u16<From> && any_i32<To>) return _mm256_cvtepu16_epi32(a);
		else if constexpr (FS.has(AVX2) && is_ymm_size(MaxSize) && is_u16<From> && any_i64<To>) return _mm256_cvtepu16_epi64(a);
		else if constexpr (FS.has(AVX2) && is_ymm_size(MaxSize) && is_u32<From> && any_i64<To>) return _mm256_cvtepu32_epi64(a);
		else if constexpr (FS.has(AVX2) && is_ymm_size(MaxSize) && is_u8<From> && any_i16<To>) return _mm256_cvtepu8_epi16(a);
		else if constexpr (FS.has(AVX2) && is_ymm_size(MaxSize) && is_u8<From> && any_i32<To>) return _mm256_cvtepu8_epi32(a);
		else if constexpr (FS.has(AVX2) && is_ymm_size(MaxSize) && is_u8<From> && any_i64<To>) return _mm256_cvtepu8_epi64(a);
		else if constexpr (FS.has(AVX) && is_ymm_size(MaxSize) && is_f64<From> && is_i32<To>) return _mm256_cvttpd_epi32(a);
		else if constexpr (FS.has(AVX) && is_ymm_size(MaxSize) && is_f32<From> && is_i32<To>) return _mm256_cvttps_epi32(a);
		else if constexpr (FS.has(AVX) && is_ymm_size(MaxSize) && is_f32<From> && is_f64<To>) return _mm256_cvtps_pd(a);
		else if constexpr (FS.has(AVX) && is_ymm_size(MaxSize) && is_f64<From> && is_f32<To>) return _mm256_cvtpd_ps(a);

		//TODO: put it in proper place!
		else if constexpr (FS.has(AVX2) && is_ymm_size(MaxSize) && any_i16<From> && any_i8<To>)
		{
			__m256i sh = _mm256_shuffle_epi8(a, _mm256_set1_epi64x(0x0E'0C'0A'08'06'04'02'00)); //don't care about odd 64-bit members, so can just broadcast
			//__m256i trunc1 = _mm256_and_si256(a, _mm256_set1_epi16(0xFF)); //force upper bytes of each word to zero
			//__m256i packus = _mm256_packus_epi16(trunc1, trunc1); //upper 64-bit halves of each 128-bit lane are duplicated result
			return TV::fromBits(_mm256_permute4x64_epi64(sh, 2 << 2)); //0, 2, 0, 0, upper discarded
		}
		else if constexpr (FS.has(AVX2) && is_ymm_size(MaxSize) && any_i32<From> && any_i8<To>)
		{
			__m256i sh = _mm256_shuffle_epi8(a, _mm256_set1_epi32(0x0C'08'04'00));
			return TV::fromBits(_mm256_permutevar8x32_epi32(sh, _mm256_set1_epi64x(4ull << 32)));
			//return _mm_unpacklo_epi32(_mm256_castsi256_si128(sh), _mm256_extracti128_si256(sh, 1));
			//return TV::fromBits(_mm256_permute
		}
		else if constexpr (FS.has(SSE41) && is_xmm_size(MaxSize) && is_i16<From> && any_i32<To>) return _mm_cvtepi16_epi32(a);
		else if constexpr (FS.has(SSE41) && is_xmm_size(MaxSize) && is_i16<From> && any_i64<To>) return _mm_cvtepi16_epi64(a);
		else if constexpr (FS.has(SSE41) && is_xmm_size(MaxSize) && is_i32<From> && any_i64<To>) return _mm_cvtepi32_epi64(a);
		else if constexpr (FS.has(SSE41) && is_xmm_size(MaxSize) && is_i8<From> && any_i16<To>) return _mm_cvtepi8_epi16(a);
		else if constexpr (FS.has(SSE41) && is_xmm_size(MaxSize) && is_i8<From> && any_i32<To>) return _mm_cvtepi8_epi32(a);
		else if constexpr (FS.has(SSE41) && is_xmm_size(MaxSize) && is_i8<From> && any_i64<To>) return _mm_cvtepi8_epi64(a);
		else if constexpr (FS.has(SSE41) && is_xmm_size(MaxSize) && is_u16<From> && any_i32<To>) return _mm_cvtepu16_epi32(a);
		else if constexpr (FS.has(SSE41) && is_xmm_size(MaxSize) && is_u16<From> && any_i64<To>) return _mm_cvtepu16_epi64(a);
		else if constexpr (FS.has(SSE41) && is_xmm_size(MaxSize) && is_u32<From> && any_i64<To>) return _mm_cvtepu32_epi64(a);
		else if constexpr (FS.has(SSE41) && is_xmm_size(MaxSize) && is_u8<From> && any_i16<To>) return _mm_cvtepu8_epi16(a);
		else if constexpr (FS.has(SSE41) && is_xmm_size(MaxSize) && is_u8<From> && any_i32<To>) return _mm_cvtepu8_epi32(a);
		else if constexpr (FS.has(SSE41) && is_xmm_size(MaxSize) && is_u8<From> && any_i64<To>) return _mm_cvtepu8_epi64(a);
		else if constexpr (FS.has(SSE2) && is_xmm_size(MaxSize) && is_f32<From> && is_i32<To>) return _mm_cvttps_epi32(a);
		else if constexpr (FS.has(SSE2) && is_xmm_size(MaxSize) && is_f32<From> && is_f64<To>) return _mm_cvtps_pd(a);
		else if constexpr (FS.has(SSE2) && is_xmm_size(MaxSize) && is_f64<From> && is_i32<To>) return _mm_cvttpd_epi32(a);
		else if constexpr (FS.has(SSE2) && is_xmm_size(MaxSize) && is_f64<From> && is_f32<To>) return _mm_cvtpd_ps(a);
		else if constexpr (FS.has(SSE2) && is_xmm_size(MaxSize) && is_i32<From> && is_f32<To>) return _mm_cvtepi32_ps(a);
		else if constexpr (FS.has(SSE2) && is_xmm_size(MaxSize) && is_i32<From> && is_f64<To>) return _mm_cvtepi32_pd(a);

		//TODO: test these. Also, add packus fallback
		else if constexpr (FS.has(SSSE3) && is_xmm_size(MaxSize) && any_i8<To> && any_i64<From>) return _mm_shuffle_epi8(a, _mm_setr_epi8(0, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
		else if constexpr (FS.has(SSSE3) && is_xmm_size(MaxSize) && any_i8<To> && any_i32<From>) return _mm_shuffle_epi8(a, _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
		else if constexpr (FS.has(SSSE3) && is_xmm_size(MaxSize) && any_i8<To> && any_i16<From>) return _mm_shuffle_epi8(a, _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1));
		else if constexpr (FS.has(SSSE3) && is_xmm_size(MaxSize) && any_i16<To> && any_i32<From>) return _mm_shuffle_epi8(a, _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1));
		else if constexpr (FS.has(SSSE3) && is_xmm_size(MaxSize) && any_i16<To> && any_i64<From>) return _mm_shuffle_epi8(a, _mm_setr_epi8(0, 1, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
		else if constexpr (FS.has(SSE2) && is_xmm_size(MaxSize) && any_i32<To> && any_i64<From>) return _mm_shuffle_epi32(a, 0 | (2 << 2));
		else if constexpr (FS.has(SSE) && is_xmm_size(MaxSize) && any_i32<To> && any_i64<From>) return _mm_shuffle_ps(vcast<__m128>(a), vcast<__m128>(a), 0 | (2 << 2));

		else if constexpr (MaxSize > 16) return TV{ vcvt<To>(a.lo()), vcvt<To>(a.hi()) };
		else
		{
			internals::scream();
			SIMD_Vector<To, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = a[i];
			return ret;
		}
	}

	template<typename S, size_t... Ns>
	auto concat(const SIMD_Vector<S, Ns>&... vectors)
	{
		constexpr size_t total_N = (Ns + ...);
		SIMD_Vector<S, total_N> ret;

		std::byte* p = reinterpret_cast<std::byte*>(&ret);
		auto append = [&](const auto& v)
			{
				memcpy(p, &v, sizeof(v));
				p += sizeof(v);
			};

		(append(vectors), ...);
		return ret;
	}

	template<typename S2, typename S, size_t N> requires (sizeof(S2) >= sizeof(S))
		SIMD_Vector<S2, N> vrzext(const SIMD_Vector<S, N>& a)
	{
		if constexpr (sizeof(S) == sizeof(S2)) return vcast<S2>(a);
		else
		{
			using U = meta::ScalarTraits<S>::UintT;
			using U2 = meta::ScalarTraits<S2>::UintT;
			auto ex = vcvt<U2>(vcast<U>(a));
			return vcast<S2>(ex);
		}
	}

	template<typename S2, typename S, size_t N> requires (sizeof(S2) <= sizeof(S))
		SIMD_Vector<S2, N> vrtrunc(const SIMD_Vector<S, N>& a)
	{
		if constexpr (sizeof(S) == sizeof(S2)) return vcast<S2>(a);
		else
		{
			//TODO: pre-AVX512 there are almost none narrowing conversions (or they use saturation)
			//Thus, some other way is needed to truncate them.
			//Saturation can be used by AND-ing with zeroes in upper bytes, that way saturation == truncation
			using U = meta::ScalarTraits<S>::UintT;
			using U2 = meta::ScalarTraits<S2>::UintT;
			auto ex = vcvt<U2>(vcast<U>(a));
			return vcast<S2>(ex);
		}
	}

	template<typename To, typename S, size_t N> requires (std::is_trivially_copyable_v<To>)
		auto vcast(const SIMD_Vector<S, N>& a)
	{
		using T = SIMD_Vector<S, N>;
		if constexpr (meta::IsScalarType<To>)
		{
			constexpr size_t RetN = sizeof(T) / sizeof(To) + bool(sizeof(T) % sizeof(To));
			SIMD_Vector<To, RetN> ret;
			memcpy(&ret, &a, std::min(sizeof(ret), sizeof(a)));
			return ret;
		}
		else
		{
			To ret;
			memcpy(&ret, &a, std::min(sizeof(ret), sizeof(a)));
			return ret;
		}
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> mask_mov(const SIMD_Vector<S, N>& ifBitClear, const mask_t<S, N>& mask, const SIMD_Vector<S, N>& ifBitSet)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		using T = SIMD_Vector<S, N>;

		if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) return vcast<S>(mask_mov(vcast<U>(ifBitClear), mask, vcast<U>(ifBitSet)));
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && any_i16<S>) return _mm512_mask_mov_epi16(ifBitClear, mask, ifBitSet);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && any_i8<S>) return _mm512_mask_mov_epi8(ifBitClear, mask, ifBitSet);
		else if constexpr (FS.has(AVX512_BW) && ymm_sized<T> && FS.has(AVX512_VL) && any_i16<S>) return _mm256_mask_mov_epi16(ifBitClear, mask, ifBitSet);
		else if constexpr (FS.has(AVX512_BW) && ymm_sized<T> && FS.has(AVX512_VL) && any_i8<S>) return _mm256_mask_mov_epi8(ifBitClear, mask, ifBitSet);
		else if constexpr (FS.has(AVX512_BW) && xmm_sized<T> && FS.has(AVX512_VL) && any_i16<S>) return _mm_mask_mov_epi16(ifBitClear, mask, ifBitSet);
		else if constexpr (FS.has(AVX512_BW) && xmm_sized<T> && FS.has(AVX512_VL) && any_i8<S>) return _mm_mask_mov_epi8(ifBitClear, mask, ifBitSet);

		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f64<S>) return _mm512_mask_mov_pd(ifBitClear, mask, ifBitSet);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f32<S>) return _mm512_mask_mov_ps(ifBitClear, mask, ifBitSet);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && any_i64<S>) return _mm512_mask_mov_epi64(ifBitClear, mask, ifBitSet);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && any_i32<S>) return _mm512_mask_mov_epi32(ifBitClear, mask, ifBitSet);

		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && is_f64<S>) return _mm256_mask_mov_pd(ifBitClear, mask, ifBitSet);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && is_f32<S>) return _mm256_mask_mov_ps(ifBitClear, mask, ifBitSet);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && any_i64<S>) return _mm256_mask_mov_epi64(ifBitClear, mask, ifBitSet);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && any_i32<S>) return _mm256_mask_mov_epi32(ifBitClear, mask, ifBitSet);

		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && xmm_sized<T> && is_f64<S>) return _mm_mask_mov_pd(ifBitClear, mask, ifBitSet);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && xmm_sized<T> && is_f32<S>) return _mm_mask_mov_ps(ifBitClear, mask, ifBitSet);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && xmm_sized<T> && any_i64<S>) return _mm_mask_mov_epi64(ifBitClear, mask, ifBitSet);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && xmm_sized<T> && any_i32<S>) return _mm_mask_mov_epi32(ifBitClear, mask, ifBitSet);

		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_int<S>) return _mm256_blendv_epi8(ifBitClear, ifBitSet, mask);
		else if constexpr (FS.has(AVX) && ymm_sized<T> && sizeof(S) == 8) return _mm256_blendv_pd(vcast<__m256d>(ifBitClear), vcast<__m256d>(ifBitSet), mask);
		else if constexpr (FS.has(AVX) && ymm_sized<T> && sizeof(S) == 4) return _mm256_blendv_ps(vcast<__m256>(ifBitClear), vcast<__m256>(ifBitSet), mask);

		else if constexpr (FS.has(SSE41) && xmm_sized<T> && is_f32<S>) return _mm_blendv_ps(ifBitClear, ifBitSet, mask);
		else if constexpr (FS.has(SSE41) && xmm_sized<T> && is_f64<S>) return _mm_blendv_pd(ifBitClear, ifBitSet, mask);
		else if constexpr (FS.has(SSE41) && xmm_sized<T> && any_int<S>) return _mm_blendv_epi8(ifBitClear, ifBitSet, mask);

		else if constexpr (FS.has(SSE) && xmm_sized<T>) return (ifBitSet & mask) | (ifBitClear & ~mask);

		else if constexpr (sizeof(T) > 16) return T{ mask_mov(ifBitClear.lo(), mask.lo(), ifBitSet.lo()), mask_mov(ifBitClear.hi(), mask.hi(), ifBitSet.hi()) };
		else
		{
			internals::scream();
			SIMD_Vector<S, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = mask[i] ? ifBitSet[i] : ifBitClear[i];
			return ret;
		}
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> maskz_mov(const mask_t<S, N>& mask, const SIMD_Vector<S, N>& ifBitSet)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		return mask_mov(SIMD_Vector<S, N>(0), mask, ifBitSet);
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> blend(const mask_t<S, N>& mask, const SIMD_Vector<S, N>& ifBitClear, const SIMD_Vector<S, N>& ifBitSet)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		return mask_mov(ifBitClear, mask, ifBitSet);
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> load(const void* p)
	{
		using namespace meta;
		using namespace internals;
		using T = SIMD_Vector<S, N>;
#ifdef __clang__
		using unaligned256i = __m256i_u;
		using unaligned128i = __m128i_u;
#else
		using unaligned256i = __m256i;
		using unaligned128i = __m128i;
#endif

		auto ld = [&]() {
			if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f64<S>) return _mm512_loadu_pd(p);
			else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f32<S>) return _mm512_loadu_ps(p);
			else if constexpr (FS.has(AVX512_F) && zmm_sized<T>) return _mm512_loadu_si512(p);
			else if constexpr (FS.has(AVX) && ymm_sized<T> && any_int<S>) return _mm256_loadu_si256(reinterpret_cast<const unaligned256i*>(p));
			else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f64<S>) return _mm256_loadu_pd(reinterpret_cast<const double*>(p));
			else if constexpr (FS.has(AVX) && ymm_sized<T>) return _mm256_loadu_ps(reinterpret_cast<const float*>(p));
			else if constexpr (FS.has(SSE2) && xmm_sized<T> && any_int<S>) return _mm_loadu_si128(reinterpret_cast<const unaligned128i*>(p));
			else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_f64<S>) return _mm_loadu_pd(reinterpret_cast<const double*>(p));
			else if constexpr (FS.has(SSE) && xmm_sized<T> && is_f32<S>) return _mm_loadu_ps(reinterpret_cast<const float*>(p));
			else if constexpr (sizeof(T) > 16) return T{ load<S,N / 2>(p), load<S,N / 2>(reinterpret_cast<const S*>(p) + N / 2) };
			else
			{
				T ret;
				memcpy(&ret, p, sizeof(ret));
				return ret;
			}
			};
		//TODO: investigate differences between loadu and lddqu: https://stackoverflow.com/questions/47425851/whats-the-difference-between-mm256-lddqu-si256-and-mm256-loadu-si256
		return T::fromBits(ld());
	}

	template<typename S, size_t N>
	SIMD_Vector<S, N> load_a(const void* p)
	{
		using namespace meta;
		using namespace internals;
		using T = SIMD_Vector<S, N>;

		auto ld = [&]() {
			if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f64<S>) return _mm512_load_pd(p);
			else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f32<S>) return _mm512_load_ps(p);
			else if constexpr (FS.has(AVX512_F) && zmm_sized<T>) return _mm512_load_si512(p);
			else if constexpr (FS.has(AVX) && ymm_sized<T> && any_int<S>) return _mm256_load_si256(reinterpret_cast<const __m256i*>(p));
			else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f64<S>) return _mm256_load_pd(reinterpret_cast<const double*>(p));
			else if constexpr (FS.has(AVX) && ymm_sized<T>) return _mm256_load_ps(reinterpret_cast<const float*>(p));
			else if constexpr (FS.has(SSE2) && xmm_sized<T> && any_int<S>) return _mm_load_si128(reinterpret_cast<const __m128i*>(p));
			else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_f64<S>) return _mm_load_pd(reinterpret_cast<const double*>(p));
			else if constexpr (FS.has(SSE) && xmm_sized<T>) return _mm_load_ps(reinterpret_cast<const float*>(p));
			else if constexpr (sizeof(T) > 16) return T{ load_a<S,N / 2>(p), load_a<S,N / 2>(reinterpret_cast<const S*>(p) + N / 2) };
			else
			{
				T ret;
				memcpy(&ret, p, sizeof(ret));
				return ret;
			}
			};
		return T::fromBits(ld());
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> load(const void* p, const mask_t<S, N>& mask)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		using T = SIMD_Vector<S, N>;
		const S* sp = reinterpret_cast<const S*>(p);

		auto zload = [&]() {
			if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && any_i16<S>) return _mm512_maskz_loadu_epi16(mask, p);
			else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && any_i8<S>) return _mm512_maskz_loadu_epi8(mask, p);
			else if constexpr (FS.has(AVX512_BW) && FS.has(AVX512_VL) && ymm_sized<T> && any_i16<S>) return _mm256_maskz_loadu_epi16(mask, p);
			else if constexpr (FS.has(AVX512_BW) && FS.has(AVX512_VL) && ymm_sized<T> && any_i8<S>) return _mm256_maskz_loadu_epi8(mask, p);
			else if constexpr (FS.has(AVX512_BW) && FS.has(AVX512_VL) && xmm_sized<T> && any_i16<S>) return _mm_maskz_loadu_epi16(mask, p);
			else if constexpr (FS.has(AVX512_BW) && FS.has(AVX512_VL) && xmm_sized<T> && any_i8<S>) return _mm_maskz_loadu_epi8(mask, p);

			else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f64<S>) return _mm512_maskz_loadu_pd(mask, p);
			else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f32<S>) return _mm512_maskz_loadu_ps(mask, p);
			else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && any_i64<S>) return _mm512_maskz_loadu_epi64(mask, p);
			else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && any_i32<S>) return _mm512_maskz_loadu_epi32(mask, p);
			else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && is_f64<S>) return _mm256_maskz_loadu_pd(mask, p);
			else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && is_f32<S>) return _mm256_maskz_loadu_ps(mask, p);
			else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && any_i64<S>) return _mm256_maskz_loadu_epi64(mask, p);
			else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && any_i32<S>) return _mm256_maskz_loadu_epi32(mask, p);
			else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && xmm_sized<T> && is_f64<S>) return _mm_maskz_loadu_pd(mask, p);
			else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && xmm_sized<T> && is_f32<S>) return _mm_maskz_loadu_ps(mask, p);
			else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && xmm_sized<T> && any_i64<S>) return _mm_maskz_loadu_epi64(mask, p);
			else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && xmm_sized<T> && any_i32<S>) return _mm_maskz_loadu_epi32(mask, p);

			else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i64<S>) return _mm256_maskload_epi64(reinterpret_cast<const int64_t*>(p), mask);
			else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i32<S>) return _mm256_maskload_epi32(reinterpret_cast<const int32_t*>(p), mask);
			else if constexpr (FS.has(AVX) && ymm_sized<T> && sizeof(S) == 8) return _mm256_maskload_pd(reinterpret_cast<const double*>(p), mask);
			else if constexpr (FS.has(AVX) && ymm_sized<T> && sizeof(S) == 4) return _mm256_maskload_ps(reinterpret_cast<const float*>(p), mask);

			else if constexpr (FS.has(AVX2) && xmm_sized<T> && any_i64<S>) return _mm_maskload_epi64(reinterpret_cast<const int64_t*>(p), mask);
			else if constexpr (FS.has(AVX2) && xmm_sized<T> && any_i32<S>) return _mm_maskload_epi64(reinterpret_cast<const int32_t*>(p), mask);
			else if constexpr (FS.has(AVX) && xmm_sized<T> && sizeof(S) == 8) return _mm_maskload_pd(reinterpret_cast<const double*>(p), mask);
			else if constexpr (FS.has(AVX) && xmm_sized<T> && sizeof(S) == 4) return _mm_maskload_ps(reinterpret_cast<const float*>(p), mask);

			else if constexpr (sizeof(T) > 16) return T{ load<S,N / 2>(p,mask.lo()), load<S,N / 2>(sp + N / 2,mask.hi()) };
			else
			{
				T ret;
				for (size_t i = 0; i < N; ++i) ret[i] = mask[i] ? sp[i] : std::bit_cast<S>(U(0));
				return ret;
			}
			};

		if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) return vcast<S>(load<S, N>(p, mask));
		else return T::fromBits(zload());
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> load(const void* p, const mask_t<S, N>& mask, const SIMD_Vector<S, N>& src)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		using T = SIMD_Vector<S, N>;
		const S* sp = (const S*)p;

		if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) return vcast<S>(load<S, N>(p, mask, vcast<U>(src)));
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && any_i16<S>) return _mm512_mask_loadu_epi16(src, mask, p);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && any_i8<S>) return _mm512_mask_loadu_epi8(src, mask, p);
		else if constexpr (FS.has(AVX512_BW) && FS.has(AVX512_VL) && ymm_sized<T> && any_i16<S>) return _mm256_mask_loadu_epi16(src, mask, p);
		else if constexpr (FS.has(AVX512_BW) && FS.has(AVX512_VL) && ymm_sized<T> && any_i8<S>) return _mm256_mask_loadu_epi8(src, mask, p);
		else if constexpr (FS.has(AVX512_BW) && FS.has(AVX512_VL) && xmm_sized<T> && any_i16<S>) return _mm_mask_loadu_epi16(src, mask, p);
		else if constexpr (FS.has(AVX512_BW) && FS.has(AVX512_VL) && xmm_sized<T> && any_i8<S>) return _mm_mask_loadu_epi8(src, mask, p);

		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f64<S>) return _mm512_mask_loadu_pd(src, mask, p);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f32<S>) return _mm512_mask_loadu_ps(src, mask, p);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && any_i64<S>) return _mm512_mask_loadu_epi64(src, mask, p);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && any_i32<S>) return _mm512_mask_loadu_epi32(src, mask, p);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && is_f64<S>) return _mm256_mask_loadu_pd(src, mask, p);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && is_f32<S>) return _mm256_mask_loadu_ps(src, mask, p);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && any_i64<S>) return _mm256_mask_loadu_epi64(src, mask, p);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && any_i32<S>) return _mm256_mask_loadu_epi32(src, mask, p);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && xmm_sized<T> && is_f64<S>) return _mm_mask_loadu_pd(src, mask, p);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && xmm_sized<T> && is_f32<S>) return _mm_mask_loadu_ps(src, mask, p);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && xmm_sized<T> && any_i64<S>) return _mm_mask_loadu_epi64(src, mask, p);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && xmm_sized<T> && any_i32<S>) return _mm_mask_loadu_epi32(src, mask, p);
		else return mask_mov(src, mask, load<S, N>(p, mask)); //TODO: this is pessimization for large vectors (AVX512 may have caught it)
	}
	template<typename S, size_t N>
	__forceinline void store(const SIMD_Vector<S, N>& v, void* p, const mask_t<S, N>& mask)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		using T = SIMD_Vector<S, N>;
		S* sp = (S*)p;

		if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) store(vcast<U>(v), p, mask);

		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && any_i16<S>) return _mm512_mask_storeu_epi16(p, mask, v);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && any_i8<S>) return _mm512_mask_storeu_epi8(p, mask, v);
		else if constexpr (FS.has(AVX512_BW) && FS.has(AVX512_VL) && ymm_sized<T> && any_i16<S>) return _mm256_mask_storeu_epi16(p, mask, v);
		else if constexpr (FS.has(AVX512_BW) && FS.has(AVX512_VL) && ymm_sized<T> && any_i8<S>) return _mm256_mask_storeu_epi8(p, mask, v);
		else if constexpr (FS.has(AVX512_BW) && FS.has(AVX512_VL) && xmm_sized<T> && any_i16<S>) return _mm_mask_storeu_epi16(p, mask, v);
		else if constexpr (FS.has(AVX512_BW) && FS.has(AVX512_VL) && xmm_sized<T> && any_i8<S>) return _mm_mask_storeu_epi8(p, mask, v);

		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f64<S>) return _mm512_mask_storeu_pd(p, mask, v);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f32<S>) return _mm512_mask_storeu_ps(p, mask, v);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && any_i64<S>) return _mm512_mask_storeu_epi64(p, mask, v);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && any_i32<S>) return _mm512_mask_storeu_epi32(p, mask, v);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && is_f64<S>) return _mm256_mask_storeu_pd(p, mask, v);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && is_f32<S>) return _mm256_mask_storeu_ps(p, mask, v);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && any_i64<S>) return _mm256_mask_storeu_epi64(p, mask, v);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && any_i32<S>) return _mm256_mask_storeu_epi32(p, mask, v);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && xmm_sized<T> && is_f64<S>) return _mm_mask_storeu_pd(p, mask, v);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && xmm_sized<T> && is_f32<S>) return _mm_mask_storeu_ps(p, mask, v);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && xmm_sized<T> && any_i64<S>) return _mm_mask_storeu_epi64(p, mask, v);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && xmm_sized<T> && any_i32<S>) return _mm_mask_storeu_epi32(p, mask, v);

		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i64<S>) return _mm256_maskstore_epi64(reinterpret_cast<int64_t*>(p), mask, v);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i32<S>) return _mm256_maskstore_epi32(reinterpret_cast<int32_t*>(p), mask, v);
		else if constexpr (FS.has(AVX) && ymm_sized<T> && sizeof(S) == 8) return _mm256_maskstore_pd(reinterpret_cast<double*>(p), mask, vcast<__m256d>(v));
		else if constexpr (FS.has(AVX) && ymm_sized<T> && sizeof(S) == 4) return _mm256_maskstore_ps(reinterpret_cast<float*>(p), mask, vcast<__m256>(v));

		else if constexpr (FS.has(AVX2) && xmm_sized<T> && any_i64<S>) return _mm_maskstore_epi64(reinterpret_cast<int64_t*>(p), mask, v);
		else if constexpr (FS.has(AVX2) && xmm_sized<T> && any_i32<S>) return _mm_maskstore_epi32(reinterpret_cast<int32_t*>(p), mask, v);
		else if constexpr (FS.has(AVX) && xmm_sized<T> && sizeof(S) == 8) return _mm_maskstore_pd(reinterpret_cast<double*>(p), mask, vcast<__m128d>(v));
		else if constexpr (FS.has(AVX) && xmm_sized<T> && sizeof(S) == 4) return _mm_maskstore_ps(reinterpret_cast<float*>(p), mask, vcast<__m128>(v));

		else if constexpr (sizeof(T) > 16)
		{
			store(v.lo(), p, mask.lo());
			store(v.hi(), sp + N / 2, mask.hi());
		}
		else
		{
			internals::scream();
			S* sp = static_cast<S*>(p);
			for (size_t i = 0; i < N; ++i) if (mask[i]) sp[i] = v[i];
		}

	}

	namespace internals
	{
		//x86 gather and scatter intrinsics only allow their Scale value to be 1, 2, 4 or 8.
		//This type takes in any Scale value and type of indices and calculates
		//optimal multiplier, new scale and canonical type for the indices
		//For example, a scale of 32 can be simplified to 8, since it's divisible by 8. 
		//Thus, input Scale value will be overridden with 8, and 
		//indices will need to be multiplied by 4 to preserve API-documented behavior
		//Fields:
		//indexMultiplier: the multiplier that indices must be multiplied by
		//extended_t: type that indices need to be converted to before the multiplication.
		//newScale: new scale value to be forwarded to gather or scatter intrinsic 
		template<size_t Scale, meta::any_int InputIndexT>
		struct GatherScatterScaleSanitizer
		{
			static inline constexpr size_t indexMultiplier = []() {for (auto& it : { 8,4,2,1 }) if (Scale % it == 0) return Scale / it; }();
			static inline constexpr size_t newScale = []() {for (auto& it : { 8,4,2,1 }) if (Scale % it == 0) return it; }();
			static constexpr inline bool fitsInto_i32 = []() {
				constexpr int64_t minVal = int64_t(std::numeric_limits<InputIndexT>::min()) * indexMultiplier;
				constexpr int64_t maxVal = int64_t(std::numeric_limits<InputIndexT>::max()) * indexMultiplier;
				return minVal >= std::numeric_limits<int32_t>::min() && maxVal <= std::numeric_limits<int32_t>::max();
				}();
			using extended_t = std::conditional_t<fitsInto_i32, int32_t, int64_t>;
		};
	}

	template<typename S, size_t N, size_t Scale, meta::any_int I>
	__forceinline void scatter(const SIMD_Vector<S, N>& v, void* base, const SIMD_Vector<I, N>& ind, const mask_t<S, N>& mask)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		//put everything up here to prevent else if chain breaks (since compilation gives useless errors by thinking unsanitized inputs surviving to native gathers
		using CanonicalIndex_t = std::conditional_t<(sizeof(I) <= 4), int32_t, int64_t>;
		using RetVec_t = SIMD_Vector<S, N>;
		using IndVec_t = SIMD_Vector<I, N>;
		constexpr size_t MaxSize = std::max(sizeof(RetVec_t), sizeof(IndVec_t));

		//convert index to __m128i/__m256i/__m512i to stop Clang from being a cry baby (it doesn't like index being non-intrinsic type and fails to compile)
		//or make it useless dummy if we need to split (doesn't work on MSVC for some reason, commenting out for now)
		using intr_t = typed_intrinsic_storage_t<I, N>;
		//std::conditional_t<MaxSize <= 64, intr_t, int> ni = MaxSize <= 64 ? vcast<intr_t>(ind) : 0;
		intr_t ni = ind;

		if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) scatter<S, N, Scale, I>(vcast<U>(v), base, ind, mask);
		//if scale is not native, emulate it by gathering with scale 1 and manually calculated byte offsets. 
		else if constexpr (Scale != 1 && Scale != 2 && Scale != 4 && Scale != 8)
		{
			using SN = GatherScatterScaleSanitizer<Scale, I>;
			return scatter<S, N, SN::newScale>(v, base, vcvt<typename SN::extended_t>(ind) * SN::indexMultiplier, mask);
		}

		//TODO: emulation of small int scatter (where elements gathered are small ints)
		else if constexpr (!std::is_same_v<I, CanonicalIndex_t>) return scatter<S, N, Scale>(v, base, vcvt<CanonicalIndex_t>(ind), mask);

		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && is_i64<I> && is_f64<S>) return _mm512_mask_i64scatter_pd(base, mask, ni, v, Scale);
		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && is_i64<I> && is_f32<S>) return _mm512_mask_i64scatter_ps(base, mask, ni, v, Scale);
		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && is_i64<I> && any_i64<S>) return _mm512_mask_i64scatter_epi64(base, mask, ni, v, Scale);
		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && is_i64<I> && any_i32<S>) return _mm512_mask_i64scatter_epi32(base, mask, ni, v, Scale);

		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && is_i32<I> && is_f64<S>) return _mm512_mask_i32scatter_pd(base, mask, ni, v, Scale);
		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && is_i32<I> && is_f32<S>) return _mm512_mask_i32scatter_ps(base, mask, ni, v, Scale);
		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && is_i32<I> && any_i64<S>) return _mm512_mask_i32scatter_epi64(base, mask, ni, v, Scale);
		else if constexpr (FS.has(AVX512_F) && is_zmm_size(MaxSize) && is_i32<I> && any_i32<S>) return _mm512_mask_i32scatter_epi32(base, mask, ni, v, Scale);

		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && is_ymm_size(MaxSize) && is_i64<I> && is_f64<S>) return _mm256_mask_i64scatter_pd(base, mask, ni, v, Scale);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && is_ymm_size(MaxSize) && is_i64<I> && is_f32<S>) return _mm256_mask_i64scatter_ps(base, mask, ni, v, Scale);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && is_ymm_size(MaxSize) && is_i64<I> && any_i64<S>) return _mm256_mask_i64scatter_epi64(base, mask, ni, v, Scale);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && is_ymm_size(MaxSize) && is_i64<I> && any_i32<S>) return _mm256_mask_i64scatter_epi32(base, mask, ni, v, Scale);

		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && is_ymm_size(MaxSize) && is_i32<I> && is_f64<S>) return _mm256_mask_i32scatter_pd(base, mask, ni, v, Scale);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && is_ymm_size(MaxSize) && is_i32<I> && is_f32<S>) return _mm256_mask_i32scatter_ps(base, mask, ni, v, Scale);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && is_ymm_size(MaxSize) && is_i32<I> && any_i64<S>) return _mm256_mask_i32scatter_epi64(base, mask, ni, v, Scale);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && is_ymm_size(MaxSize) && is_i32<I> && any_i32<S>) return _mm256_mask_i32scatter_epi32(base, mask, ni, v, Scale);

		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && is_xmm_size(MaxSize) && is_i64<I> && is_f64<S>) return _mm_mask_i64scatter_pd(base, mask, ni, v, Scale);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && is_xmm_size(MaxSize) && is_i64<I> && is_f32<S>) return _mm_mask_i64scatter_ps(base, mask, ni, v, Scale);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && is_xmm_size(MaxSize) && is_i64<I> && any_i64<S>) return _mm_mask_i64scatter_epi64(base, mask, ni, v, Scale);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && is_xmm_size(MaxSize) && is_i64<I> && any_i32<S>) return _mm_mask_i64scatter_epi32(base, mask, ni, v, Scale);

		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && is_xmm_size(MaxSize) && is_i32<I> && is_f64<S>) return _mm_mask_i32scatter_pd(base, mask, ni, v, Scale);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && is_xmm_size(MaxSize) && is_i32<I> && is_f32<S>) return _mm_mask_i32scatter_ps(base, mask, ni, v, Scale);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && is_xmm_size(MaxSize) && is_i32<I> && any_i64<S>) return _mm_mask_i32scatter_epi64(base, mask, ni, v, Scale);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && is_xmm_size(MaxSize) && is_i32<I> && any_i32<S>) return _mm_mask_i32scatter_epi32(base, mask, ni, v, Scale);
		else if constexpr (MaxSize > 16) //TODO: can make it 64-large, no scatters available in non-AVX512
		{
			scatter<S, N / 2, Scale, I>(v.lo(), base, ind.lo(), mask.lo());
			scatter<S, N / 2, Scale, I>(v.hi(), base, ind.hi(), mask.hi());
		}

		else
		{
			internals::scream();
			size_t addr = size_t(base);
			for (size_t i = 0; i < N; ++i) if (mask[i]) *(S*)(addr + Scale * ind[i]) = v[i];
		}

	}
	template<typename S, size_t N>
	__forceinline mask_t<S, N> cmp_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		using T = SIMD_Vector<S, N>;

		if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_i8<S>) return _mm512_cmpeq_epi8_mask(a, b);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_u8<S>) return _mm512_cmpeq_epu8_mask(a, b);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_i16<S>) return _mm512_cmpeq_epi16_mask(a, b);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_u16<S>) return _mm512_cmpeq_epu16_mask(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_i32<S>) return _mm512_cmpeq_epi32_mask(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_u32<S>) return _mm512_cmpeq_epu32_mask(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_i64<S>) return _mm512_cmpeq_epi64_mask(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_u64<S>) return _mm512_cmpeq_epu64_mask(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f32<S>) return _mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f64<S>) return _mm512_cmp_pd_mask(a, b, _CMP_EQ_OQ);

		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && ymm_sized<T> && is_i8<S>) return _mm256_cmpeq_epi8_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && ymm_sized<T> && is_u8<S>) return _mm256_cmpeq_epu8_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && ymm_sized<T> && is_i16<S>) return _mm256_cmpeq_epi16_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && ymm_sized<T> && is_u16<S>) return _mm256_cmpeq_epu16_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_i32<S>) return _mm256_cmpeq_epi32_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_u32<S>) return _mm256_cmpeq_epu32_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_i64<S>) return _mm256_cmpeq_epi64_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_u64<S>) return _mm256_cmpeq_epu64_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_f32<S>) return _mm256_cmp_ps_mask(a, b, _CMP_EQ_OQ);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_f64<S>) return _mm256_cmp_pd_mask(a, b, _CMP_EQ_OQ);

		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && xmm_sized<T> && is_i8<S>) return _mm_cmpeq_epi8_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && xmm_sized<T> && is_u8<S>) return _mm_cmpeq_epu8_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && xmm_sized<T> && is_i16<S>) return _mm_cmpeq_epi16_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && xmm_sized<T> && is_u16<S>) return _mm_cmpeq_epu16_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_i32<S>) return _mm_cmpeq_epi32_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_u32<S>) return _mm_cmpeq_epu32_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_i64<S>) return _mm_cmpeq_epi64_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_u64<S>) return _mm_cmpeq_epu64_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_f32<S>) return _mm_cmp_ps_mask(a, b, _CMP_EQ_OQ);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_f64<S>) return _mm_cmp_pd_mask(a, b, _CMP_EQ_OQ);

		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i64<S>) return _mm256_cmpeq_epi64(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i32<S>) return _mm256_cmpeq_epi32(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i16<S>) return _mm256_cmpeq_epi16(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i8<S>) return _mm256_cmpeq_epi8(a, b);
		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f64<S>) return _mm256_cmp_pd(a, b, _CMP_EQ_OQ);
		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f32<S>) return _mm256_cmp_ps(a, b, _CMP_EQ_OQ);

		else if constexpr (FS.has(SSE41) && xmm_sized<T> && any_i64<S>) return _mm_cmpeq_epi64(a, b);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && any_i32<S>) return _mm_cmpeq_epi32(a, b);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && any_i16<S>) return _mm_cmpeq_epi16(a, b);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && any_i8<S>) return _mm_cmpeq_epi8(a, b);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_f64<S>) return _mm_cmpeq_pd(a, b);
		else if constexpr (FS.has(SSE) && xmm_sized<T> && is_f32<S>) return _mm_cmpeq_ps(a, b);

		else if constexpr (sizeof(T) > 16) return { cmp_equal(a.lo(),b.lo()), cmp_equal(a.hi(),b.hi()) };
		else
		{
			internals::scream();
			typename SIMD_Vector<S, N>::MaskT ret = 0;
			for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] == b[i]);
			return ret;
		}
	}
	template<typename S, size_t N>
	__forceinline mask_t<S, N> cmp_not_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		using T = SIMD_Vector<S, N>;

		if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_i8<S>) return _mm512_cmpneq_epi8_mask(a, b);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_u8<S>) return _mm512_cmpneq_epu8_mask(a, b);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_i16<S>) return _mm512_cmpneq_epi16_mask(a, b);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_u16<S>) return _mm512_cmpneq_epu16_mask(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_i32<S>) return _mm512_cmpneq_epi32_mask(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_u32<S>) return _mm512_cmpneq_epu32_mask(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_i64<S>) return _mm512_cmpneq_epi64_mask(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_u64<S>) return _mm512_cmpneq_epu64_mask(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f32<S>) return _mm512_cmp_ps_mask(a, b, _CMP_NEQ_OQ);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f64<S>) return _mm512_cmp_pd_mask(a, b, _CMP_NEQ_OQ);
		else if constexpr (sizeof(T) > 32) return { cmp_not_equal(a.lo(),b.lo()), cmp_not_equal(a.hi(),b.hi()) };
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && ymm_sized<T> && is_i8<S>) return _mm256_cmpneq_epi8_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && ymm_sized<T> && is_u8<S>) return _mm256_cmpneq_epu8_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && ymm_sized<T> && is_i16<S>) return _mm256_cmpneq_epi16_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && ymm_sized<T> && is_u16<S>) return _mm256_cmpneq_epu16_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_i32<S>) return _mm256_cmpneq_epi32_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_u32<S>) return _mm256_cmpneq_epu32_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_i64<S>) return _mm256_cmpneq_epi64_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_u64<S>) return _mm256_cmpneq_epu64_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_f32<S>) return _mm256_cmp_ps_mask(a, b, _CMP_NEQ_OQ);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_f64<S>) return _mm256_cmp_pd_mask(a, b, _CMP_NEQ_OQ);

		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && xmm_sized<T> && is_i8<S>) return _mm_cmpneq_epi8_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && xmm_sized<T> && is_u8<S>) return _mm_cmpneq_epu8_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && xmm_sized<T> && is_i16<S>) return _mm_cmpneq_epi16_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && xmm_sized<T> && is_u16<S>) return _mm_cmpneq_epu16_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_i32<S>) return _mm_cmpneq_epi32_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_u32<S>) return _mm_cmpneq_epu32_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_i64<S>) return _mm_cmpneq_epi64_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_u64<S>) return _mm_cmpneq_epu64_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_f32<S>) return _mm_cmp_ps_mask(a, b, _CMP_NEQ_OQ);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_f64<S>) return _mm_cmp_pd_mask(a, b, _CMP_NEQ_OQ);

		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f64<S>) return _mm256_cmp_pd(a, b, _CMP_NEQ_OQ);
		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f32<S>) return _mm256_cmp_ps(a, b, _CMP_NEQ_OQ);

		else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_f64<S>) return _mm_cmpneq_pd(a, b);
		else if constexpr (FS.has(SSE) && xmm_sized<T> && is_f32<S>) return _mm_cmpneq_ps(a, b);

		else if (sizeof(T) <= 64 && any_int<S>) return ~cmp_equal(a, b); //don't hijack AVX512 cmp if too large

		else if constexpr (sizeof(T) > 16) return { cmp_not_equal(a.lo(),b.lo()), cmp_not_equal(a.hi(),b.hi()) };
		else
		{
			internals::scream();
			mask_t<S, N> ret = 0;
			for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] != b[i]);
			return ret;
		}

	}
	template<typename S, size_t N>
	__forceinline mask_t<S, N> cmp_less(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		using T = SIMD_Vector<S, N>;

		if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_i8<S>) return _mm512_cmplt_epi8_mask(a, b);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_u8<S>) return _mm512_cmplt_epu8_mask(a, b);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_i16<S>) return _mm512_cmplt_epi16_mask(a, b);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_u16<S>) return _mm512_cmplt_epu16_mask(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_i32<S>) return _mm512_cmplt_epi32_mask(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_u32<S>) return _mm512_cmplt_epu32_mask(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_i64<S>) return _mm512_cmplt_epi64_mask(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_u64<S>) return _mm512_cmplt_epu64_mask(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f32<S>) return _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f64<S>) return _mm512_cmp_pd_mask(a, b, _CMP_LT_OQ);

		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && ymm_sized<T> && is_i8<S>) return _mm256_cmplt_epi8_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && ymm_sized<T> && is_u8<S>) return _mm256_cmplt_epu8_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && ymm_sized<T> && is_i16<S>) return _mm256_cmplt_epi16_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && ymm_sized<T> && is_u16<S>) return _mm256_cmplt_epu16_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_i32<S>) return _mm256_cmplt_epi32_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_u32<S>) return _mm256_cmplt_epu32_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_i64<S>) return _mm256_cmplt_epi64_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_u64<S>) return _mm256_cmplt_epu64_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_f32<S>) return _mm256_cmp_ps_mask(a, b, _CMP_LT_OQ);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_f64<S>) return _mm256_cmp_pd_mask(a, b, _CMP_LT_OQ);

		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && xmm_sized<T> && is_i8<S>) return _mm_cmplt_epi8_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && xmm_sized<T> && is_u8<S>) return _mm_cmplt_epu8_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && xmm_sized<T> && is_i16<S>) return _mm_cmplt_epi16_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && xmm_sized<T> && is_u16<S>) return _mm_cmplt_epu16_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_i32<S>) return _mm_cmplt_epi32_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_u32<S>) return _mm_cmplt_epu32_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_i64<S>) return _mm_cmplt_epi64_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_u64<S>) return _mm_cmplt_epu64_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_f32<S>) return _mm_cmp_ps_mask(a, b, _CMP_LT_OQ);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_f64<S>) return _mm_cmp_pd_mask(a, b, _CMP_LT_OQ);

		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f64<S>) return _mm256_cmp_pd(a, b, _CMP_LT_OQ);
		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f32<S>) return _mm256_cmp_ps(a, b, _CMP_LT_OQ);

		else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_f64<S>) return _mm_cmplt_pd(a, b);
		else if constexpr (FS.has(SSE) && xmm_sized<T> && is_f32<S>) return _mm_cmplt_ps(a, b);

		else if constexpr (sizeof(T) <= 64 && any_int<S>) return cmp_greater(b, a); //swap order, but don't hijack too large vectors from AVX512 comparisons

		else if constexpr (sizeof(T) > 16) return { cmp_less(a.lo(),b.lo()), cmp_less(a.hi(),b.hi()) };
		else
		{
			internals::scream();
			typename SIMD_Vector<S, N>::MaskT ret = 0;
			for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] < b[i]);
			return ret;
		}
	}
	template<typename S, size_t N>
	__forceinline mask_t<S, N> cmp_less_or_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		using T = SIMD_Vector<S, N>;

		if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_i8<S>) return _mm512_cmple_epi8_mask(a, b);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_u8<S>) return _mm512_cmple_epu8_mask(a, b);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_i16<S>) return _mm512_cmple_epi16_mask(a, b);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_u16<S>) return _mm512_cmple_epu16_mask(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_i32<S>) return _mm512_cmple_epi32_mask(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_u32<S>) return _mm512_cmple_epu32_mask(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_i64<S>) return _mm512_cmple_epi64_mask(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_u64<S>) return _mm512_cmple_epu64_mask(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f32<S>) return _mm512_cmp_ps_mask(a, b, _CMP_LE_OQ);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f64<S>) return _mm512_cmp_pd_mask(a, b, _CMP_LE_OQ);

		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && ymm_sized<T> && is_i8<S>) return _mm256_cmple_epi8_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && ymm_sized<T> && is_u8<S>) return _mm256_cmple_epu8_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && ymm_sized<T> && is_i16<S>) return _mm256_cmple_epi16_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && ymm_sized<T> && is_u16<S>) return _mm256_cmple_epu16_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_i32<S>) return _mm256_cmple_epi32_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_u32<S>) return _mm256_cmple_epu32_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_i64<S>) return _mm256_cmple_epi64_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_u64<S>) return _mm256_cmple_epu64_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_f32<S>) return _mm256_cmp_ps_mask(a, b, _CMP_LE_OQ);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_f64<S>) return _mm256_cmp_pd_mask(a, b, _CMP_LE_OQ);

		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && xmm_sized<T> && is_i8<S>) return _mm_cmple_epi8_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && xmm_sized<T> && is_u8<S>) return _mm_cmple_epu8_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && xmm_sized<T> && is_i16<S>) return _mm_cmple_epi16_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && xmm_sized<T> && is_u16<S>) return _mm_cmple_epu16_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_i32<S>) return _mm_cmple_epi32_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_u32<S>) return _mm_cmple_epu32_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_i64<S>) return _mm_cmple_epi64_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_u64<S>) return _mm_cmple_epu64_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_f32<S>) return _mm_cmp_ps_mask(a, b, _CMP_LE_OQ);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_f64<S>) return _mm_cmp_pd_mask(a, b, _CMP_LE_OQ);

		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f64<S>) return _mm256_cmp_pd(a, b, _CMP_LE_OQ);
		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f32<S>) return _mm256_cmp_ps(a, b, _CMP_LE_OQ);

		else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_f64<S>) return _mm_cmple_pd(a, b);
		else if constexpr (FS.has(SSE) && xmm_sized<T> && is_f32<S>) return _mm_cmple_ps(a, b);

		else if (sizeof(T) <= 64 && any_int<S>) return ~cmp_greater(a, b); //don't hijack AVX512 cmp if too large

		else if constexpr (sizeof(T) > 16) return { cmp_less_or_equal(a.lo(),b.lo()), cmp_less_or_equal(a.hi(),b.hi()) };
		else
		{
			internals::scream();
			typename SIMD_Vector<S, N>::MaskT ret = 0;
			for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] <= b[i]);
			return ret;
		}
	}
	template<typename S, size_t N>
	__forceinline mask_t<S, N> cmp_greater(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		using T = SIMD_Vector<S, N>;

		if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_i8<S>) return _mm512_cmpgt_epi8_mask(a, b);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_u8<S>) return _mm512_cmpgt_epu8_mask(a, b);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_i16<S>) return _mm512_cmpgt_epi16_mask(a, b);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_u16<S>) return _mm512_cmpgt_epu16_mask(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_i32<S>) return _mm512_cmpgt_epi32_mask(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_u32<S>) return _mm512_cmpgt_epu32_mask(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_i64<S>) return _mm512_cmpgt_epi64_mask(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_u64<S>) return _mm512_cmpgt_epu64_mask(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f32<S>) return _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f64<S>) return _mm512_cmp_pd_mask(a, b, _CMP_GT_OQ);

		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && ymm_sized<T> && is_i8<S>) return _mm256_cmpgt_epi8_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && ymm_sized<T> && is_u8<S>) return _mm256_cmpgt_epu8_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && ymm_sized<T> && is_i16<S>) return _mm256_cmpgt_epi16_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && ymm_sized<T> && is_u16<S>) return _mm256_cmpgt_epu16_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_i32<S>) return _mm256_cmpgt_epi32_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_u32<S>) return _mm256_cmpgt_epu32_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_i64<S>) return _mm256_cmpgt_epi64_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_u64<S>) return _mm256_cmpgt_epu64_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_f32<S>) return _mm256_cmp_ps_mask(a, b, _CMP_GT_OQ);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_f64<S>) return _mm256_cmp_pd_mask(a, b, _CMP_GT_OQ);

		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && xmm_sized<T> && is_i8<S>) return _mm_cmpgt_epi8_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && xmm_sized<T> && is_u8<S>) return _mm_cmpgt_epu8_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && xmm_sized<T> && is_i16<S>) return _mm_cmpgt_epi16_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && xmm_sized<T> && is_u16<S>) return _mm_cmpgt_epu16_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_i32<S>) return _mm_cmpgt_epi32_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_u32<S>) return _mm_cmpgt_epu32_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_i64<S>) return _mm_cmpgt_epi64_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_u64<S>) return _mm_cmpgt_epu64_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_f32<S>) return _mm_cmp_ps_mask(a, b, _CMP_GT_OQ);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_f64<S>) return _mm_cmp_pd_mask(a, b, _CMP_GT_OQ);

		else if constexpr (FS.has(AVX2) && ymm_sized<T> && is_i64<S>) return _mm256_cmpgt_epi64(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && is_i32<S>) return _mm256_cmpgt_epi32(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && is_i16<S>) return _mm256_cmpgt_epi16(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && is_i8<S>) return _mm256_cmpgt_epi8(a, b);

		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f64<S>) return _mm256_cmp_pd(a, b, _CMP_GT_OQ);
		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f32<S>) return _mm256_cmp_ps(a, b, _CMP_GT_OQ);

		else if constexpr (FS.has(SSE42) && xmm_sized<T> && is_i64<S>) return _mm_cmpgt_epi64(a, b); //TODO: emulation for old
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_i32<S>) return _mm_cmpgt_epi32(a, b);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_i16<S>) return _mm_cmpgt_epi16(a, b);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_i8<S>) return _mm_cmpgt_epi8(a, b);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_f64<S>) return _mm_cmpgt_pd(a, b);
		else if constexpr (FS.has(SSE) && xmm_sized<T> && is_f32<S>) return _mm_cmpgt_ps(a, b);
		else if constexpr (std::is_unsigned_v<S>)
		{
			using I = typename ScalarTraits<S>::IntT;
			constexpr I xorv = I(1) << ((sizeof(I) * 8) - 1); //xor with 0x800..000 before comparison
			return cmp_greater(vcast<I>(a) ^ xorv, vcast<I>(b) ^ xorv);
		}

		else if constexpr (sizeof(T) > 16) return { cmp_greater(a.lo(),b.lo()), cmp_greater(a.hi(),b.hi()) };
		else
		{
			internals::scream();
			typename SIMD_Vector<S, N>::MaskT ret = 0;
			for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] > b[i]);
			return ret;
		}
	}
	template<typename S, size_t N>
	__forceinline mask_t<S, N> cmp_greater_or_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		using T = SIMD_Vector<S, N>;

		if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_i8<S>) return _mm512_cmpge_epi8_mask(a, b);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_u8<S>) return _mm512_cmpge_epu8_mask(a, b);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_i16<S>) return _mm512_cmpge_epi16_mask(a, b);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_u16<S>) return _mm512_cmpge_epu16_mask(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_i32<S>) return _mm512_cmpge_epi32_mask(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_u32<S>) return _mm512_cmpge_epu32_mask(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_i64<S>) return _mm512_cmpge_epi64_mask(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_u64<S>) return _mm512_cmpge_epu64_mask(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f32<S>) return _mm512_cmp_ps_mask(a, b, _CMP_GE_OQ);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f64<S>) return _mm512_cmp_pd_mask(a, b, _CMP_GE_OQ);

		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && ymm_sized<T> && is_i8<S>) return _mm256_cmpge_epi8_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && ymm_sized<T> && is_u8<S>) return _mm256_cmpge_epu8_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && ymm_sized<T> && is_i16<S>) return _mm256_cmpge_epi16_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && ymm_sized<T> && is_u16<S>) return _mm256_cmpge_epu16_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_i32<S>) return _mm256_cmpge_epi32_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_u32<S>) return _mm256_cmpge_epu32_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_i64<S>) return _mm256_cmpge_epi64_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_u64<S>) return _mm256_cmpge_epu64_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_f32<S>) return _mm256_cmp_ps_mask(a, b, _CMP_GE_OQ);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && ymm_sized<T> && is_f64<S>) return _mm256_cmp_pd_mask(a, b, _CMP_GE_OQ);

		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && xmm_sized<T> && is_i8<S>) return _mm_cmpge_epi8_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && xmm_sized<T> && is_u8<S>) return _mm_cmpge_epu8_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && xmm_sized<T> && is_i16<S>) return _mm_cmpge_epi16_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && xmm_sized<T> && is_u16<S>) return _mm_cmpge_epu16_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_i32<S>) return _mm_cmpge_epi32_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_u32<S>) return _mm_cmpge_epu32_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_i64<S>) return _mm_cmpge_epi64_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_u64<S>) return _mm_cmpge_epu64_mask(a, b);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_f32<S>) return _mm_cmp_ps_mask(a, b, _CMP_GE_OQ);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_F) && xmm_sized<T> && is_f64<S>) return _mm_cmp_pd_mask(a, b, _CMP_GE_OQ);

		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f64<S>) return _mm256_cmp_pd(a, b, _CMP_GE_OQ);
		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f32<S>) return _mm256_cmp_ps(a, b, _CMP_GE_OQ);

		else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_f64<S>) return _mm_cmpge_pd(a, b);
		else if constexpr (FS.has(SSE) && xmm_sized<T> && is_f32<S>) return _mm_cmpge_ps(a, b);

		else if constexpr (sizeof(T) <= 64 && any_int<S>) return ~cmp_greater(b, a); //order swap, not less. don't hijack too large vectors from AVX512 comparisons
		else if constexpr (sizeof(T) > 16) return { cmp_greater_or_equal(a.lo(),b.lo()), cmp_greater_or_equal(a.hi(),b.hi()) };
		else
		{
			typename SIMD_Vector<S, N>::MaskT ret = 0;
			for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] >= b[i]);
			return ret;
		}
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> abs(const SIMD_Vector<S, N>& a)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		using T = SIMD_Vector<S, N>;

#define AVXXY_SPLIT_ABS T{ abs(a.lo()), abs(a.hi()) }

		if constexpr (std::is_unsigned_v<S>) return a;

		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_i16<S>) return _mm512_abs_epi16(a);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_i8<S>) return _mm512_abs_epi8(a);

		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f64<S>) return _mm512_abs_pd(a);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f32<S>) return _mm512_abs_ps(a);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_i64<S>) return _mm512_abs_epi64(a);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_i32<S>) return _mm512_abs_epi32(a);
		else if constexpr (sizeof(T) > 64 && FS.has(AVX512_F) && (is_f32<S> || is_f64<S> || is_i64<S>)) return AVXXY_SPLIT_ABS;
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && is_i64<S>) return _mm256_abs_epi64(a);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && xmm_sized<T> && is_i64<S>) return _mm_abs_epi64(a);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && is_i64<S>) return AVXXY_SPLIT_ABS;

		else if constexpr (FS.has(AVX2) && ymm_sized<T> && is_i8<S>) return _mm256_abs_epi8(a);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && is_i16<S>) return _mm256_abs_epi16(a);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && is_i32<S>) return _mm256_abs_epi32(a);

		else if constexpr (FS.has(SSSE3) && xmm_sized<T> && is_i8<S>) return _mm_abs_epi8(a);
		else if constexpr (FS.has(SSSE3) && xmm_sized<T> && is_i16<S>) return _mm_abs_epi16(a);
		else if constexpr (FS.has(SSSE3) && xmm_sized<T> && is_i32<S>) return _mm_abs_epi32(a);

		else if constexpr (is_i64<S>) return mask_mov(a, a < 0, -a);
		//force sign bit to zero
		else if constexpr (is_f64<S>) return a & std::bit_cast<S>(~(uint64_t(1) << 63));
		else if constexpr (is_f32<S>) return a & std::bit_cast<S>(~(uint32_t(1) << 31));
		else if constexpr (is_fp16<S> || is_bf16<S>) return a & std::bit_cast<S>(~(uint16_t(1) << 15));

		else if constexpr (sizeof(T) > 16) return AVXXY_SPLIT_ABS;
		else
		{
			internals::scream();
			SIMD_Vector<S, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = std::abs(a[i]);
			return ret;
		}
#undef AVXXY_SPLIT_ABS
	}

	template<meta::any_float S, size_t N>
	__forceinline SIMD_Vector<S, N> floor(const SIMD_Vector<S, N>& a)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		using T = SIMD_Vector<S, N>;

		if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f64<S>) return _mm512_floor_pd(a);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f32<S>) return _mm512_floor_ps(a);

		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f64<S>) return _mm256_floor_pd(a);
		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f32<S>) return _mm256_floor_ps(a);

		else if constexpr (FS.has(SSE41) && xmm_sized<T> && is_f64<S>) return _mm_floor_pd(a);
		else if constexpr (FS.has(SSE41) && xmm_sized<T> && is_f32<S>) return _mm_floor_ps(a);
		else if constexpr (sizeof(T) > 16) return T{ floor(a.lo()), floor(a.hi()) };
		else
		{
			internals::scream();
			SIMD_Vector<S, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = std::floor(a[i]);
			return ret;
		}
	}
	template<meta::any_float S, size_t N>
	__forceinline SIMD_Vector<S, N> ceil(const SIMD_Vector<S, N>& a)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		using T = SIMD_Vector<S, N>;

		if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f64<S>) return _mm512_ceil_pd(a);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f32<S>) return _mm512_ceil_ps(a);

		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f64<S>) return _mm256_ceil_pd(a);
		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f32<S>) return _mm256_ceil_ps(a);

		else if constexpr (FS.has(SSE41) && xmm_sized<T> && is_f64<S>) return _mm_ceil_pd(a);
		else if constexpr (FS.has(SSE41) && xmm_sized<T> && is_f32<S>) return _mm_ceil_ps(a);

		else if constexpr (sizeof(T) > 16) return T{ ceil(a.lo()), ceil(a.hi()) };
		else
		{
			internals::scream();
			SIMD_Vector<S, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = std::ceil(a[i]);
			return ret;
		}
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> min(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		using T = SIMD_Vector<S, N>;

		if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_i16<S>) return _mm512_min_epi16(a, b);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_u16<S>) return _mm512_min_epu16(a, b);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_i8<S>) return _mm512_min_epi8(a, b);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_u8<S>) return _mm512_min_epu8(a, b);

		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f64<S>) return _mm512_min_pd(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f32<S>) return _mm512_min_ps(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_i64<S>) return _mm512_min_epi64(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_u64<S>) return _mm512_min_epu64(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_i32<S>) return _mm512_min_epi32(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_u32<S>) return _mm512_min_epu32(a, b);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && is_i64<S>) return _mm256_min_epi64(a, b);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && is_u64<S>) return _mm256_min_epu64(a, b);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && xmm_sized<T> && is_i64<S>) return _mm_min_epi64(a, b);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && xmm_sized<T> && is_u64<S>) return _mm_min_epu64(a, b);

		else if constexpr (FS.has(AVX2) && ymm_sized<T> && is_i8<S>) return _mm256_min_epi8(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && is_i16<S>) return _mm256_min_epi16(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && is_i32<S>) return _mm256_min_epi32(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && is_u8<S>) return _mm256_min_epu8(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && is_u16<S>) return _mm256_min_epu16(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && is_u32<S>) return _mm256_min_epu32(a, b);
		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f64<S>) return _mm256_min_pd(a, b);
		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f32<S>) return _mm256_min_ps(a, b);

		else if constexpr (FS.has(SSE41) && xmm_sized<T> && is_i8<S>) return _mm_min_epi8(a, b);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_i16<S>) return _mm_min_epi16(a, b);
		else if constexpr (FS.has(SSE41) && xmm_sized<T> && is_i32<S>) return _mm_min_epi32(a, b);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_u8<S>) return _mm_min_epu8(a, b);
		else if constexpr (FS.has(SSE41) && xmm_sized<T> && is_u16<S>) return _mm_min_epu16(a, b);
		else if constexpr (FS.has(SSE41) && xmm_sized<T> && is_u32<S>) return _mm_min_epu32(a, b);

		else if constexpr (sizeof(T) > 16) return T{ min(a.lo(), b.lo()), min(a.hi(),b.hi()) };
		else return mask_mov(a, b < a, b);

		/*
		else
		{
			internals::scream();
			SIMD_Vector<S, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = std::min(a[i], b[i]);
			return ret;
		}*/
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> max(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		using T = SIMD_Vector<S, N>;

		if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_i16<S>) return _mm512_max_epi16(a, b);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_u16<S>) return _mm512_max_epu16(a, b);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_i8<S>) return _mm512_max_epi8(a, b);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_u8<S>) return _mm512_max_epu8(a, b);

		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f64<S>) return _mm512_max_pd(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f32<S>) return _mm512_max_ps(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_i64<S>) return _mm512_max_epi64(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_u64<S>) return _mm512_max_epu64(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_i32<S>) return _mm512_max_epi32(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_u32<S>) return _mm512_max_epu32(a, b);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && is_i64<S>) return _mm256_max_epi64(a, b);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && is_u64<S>) return _mm256_max_epu64(a, b);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && xmm_sized<T> && is_i64<S>) return _mm_max_epi64(a, b);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && xmm_sized<T> && is_u64<S>) return _mm_max_epu64(a, b);

		else if constexpr (FS.has(AVX2) && ymm_sized<T> && is_i8<S>) return _mm256_max_epi8(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && is_i16<S>) return _mm256_max_epi16(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && is_i32<S>) return _mm256_max_epi32(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && is_u8<S>) return _mm256_max_epu8(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && is_u16<S>) return _mm256_max_epu16(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && is_u32<S>) return _mm256_max_epu32(a, b);
		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f64<S>) return _mm256_max_pd(a, b);
		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f32<S>) return _mm256_max_ps(a, b);

		else if constexpr (FS.has(SSE41) && xmm_sized<T> && is_i8<S>) return _mm_max_epi8(a, b);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_i16<S>) return _mm_max_epi16(a, b);
		else if constexpr (FS.has(SSE41) && xmm_sized<T> && is_i32<S>) return _mm_max_epi32(a, b);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_u8<S>) return _mm_max_epu8(a, b);
		else if constexpr (FS.has(SSE41) && xmm_sized<T> && is_u16<S>) return _mm_max_epu16(a, b);
		else if constexpr (FS.has(SSE41) && xmm_sized<T> && is_u32<S>) return _mm_max_epu32(a, b);

		else if constexpr (sizeof(T) > 16) return T{ max(a.lo(), b.lo()), max(a.hi(),b.hi()) };
		else return mask_mov(a, b > a, b);
		/*
		else
		{
			internals::scream();
			SIMD_Vector<S, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = std::max(a[i], b[i]);
			return ret;
		}*/
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> clamp(const SIMD_Vector<S, N>& val, const SIMD_Vector<S, N>& min, const SIMD_Vector<S, N>& max)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		return AVXXY_NAMESPACE::max(min, AVXXY_NAMESPACE::min(val, max));
	}


	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> unpacklo(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		using T = SIMD_Vector<S, N>;

		if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) return vcast<S>(unpacklo(vcast<U>(a), vcast<U>(b)));

		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && any_i16<S>) return _mm512_unpacklo_epi16(a, b);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && any_i8<S>) return _mm512_unpacklo_epi8(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f64<S>) return _mm512_unpacklo_pd(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f32<S>) return _mm512_unpacklo_ps(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && any_i64<S>) return _mm512_unpacklo_epi64(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && any_i32<S>) return _mm512_unpacklo_epi32(a, b);

		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i64<S>) return _mm256_unpacklo_epi64(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i32<S>) return _mm256_unpacklo_epi32(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i16<S>) return _mm256_unpacklo_epi16(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i8<S>) return _mm256_unpacklo_epi8(a, b);
		else if constexpr (FS.has(AVX) && ymm_sized<T> && sizeof(S) == 8) return T::fromBits(_mm256_unpacklo_pd(vcast<__m256d>(a), vcast<__m256d>(b)));
		else if constexpr (FS.has(AVX) && ymm_sized<T> && sizeof(S) == 4) return T::fromBits(_mm256_unpacklo_ps(vcast<__m256>(a), vcast<__m256>(b)));

		else if constexpr (FS.has(SSE2) && xmm_sized<T> && any_i64<S>) return _mm_unpacklo_epi64(a, b);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && any_i32<S>) return _mm_unpacklo_epi32(a, b);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && any_i16<S>) return _mm_unpacklo_epi16(a, b);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && any_i8<S>) return _mm_unpacklo_epi8(a, b);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_f64<S>) return _mm_unpacklo_pd(a, b);
		else if constexpr (FS.has(SSE) && xmm_sized<T> && sizeof(S) == 4) return T::fromBits(_mm_unpacklo_ps(vcast<__m128>(a), vcast<__m128>(b)));

		else if constexpr (sizeof(T) > 16) return T{ unpacklo(a.lo(),b.lo()), unpacklo(a.hi(),b.hi()) };
		else
		{
			internals::scream();
			return internals::unpack_base<S, N, true>(a, b);
		}
	}
	template<typename S, size_t N>
		requires meta::unpackhi_legal<S, N>
	__forceinline SIMD_Vector<S, N> unpackhi(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		using T = SIMD_Vector<S, N>;

		if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) return vcast<S>(unpackhi(vcast<U>(a), vcast<U>(b)));

		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && any_i16<S>) return _mm512_unpackhi_epi16(a, b);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && any_i8<S>) return _mm512_unpackhi_epi8(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f64<S>) return _mm512_unpackhi_pd(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f32<S>) return _mm512_unpackhi_ps(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && any_i64<S>) return _mm512_unpackhi_epi64(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && any_i32<S>) return _mm512_unpackhi_epi32(a, b);

		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i64<S>) return _mm256_unpackhi_epi64(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i32<S>) return _mm256_unpackhi_epi32(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i16<S>) return _mm256_unpackhi_epi16(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i8<S>) return _mm256_unpackhi_epi8(a, b);
		else if constexpr (FS.has(AVX) && ymm_sized<T> && sizeof(S) == 8) return T::fromBits(_mm256_unpackhi_pd(vcast<__m256d>(a), vcast<__m256d>(b)));
		else if constexpr (FS.has(AVX) && ymm_sized<T> && sizeof(S) == 4) return T::fromBits(_mm256_unpackhi_ps(vcast<__m256>(a), vcast<__m256>(b)));

		else if constexpr (FS.has(SSE2) && xmm_sized<T> && any_i64<S>) return _mm_unpackhi_epi64(a, b);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && any_i32<S>) return _mm_unpackhi_epi32(a, b);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && any_i16<S>) return _mm_unpackhi_epi16(a, b);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && any_i8<S>) return _mm_unpackhi_epi8(a, b);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_f64<S>) return _mm_unpackhi_pd(a, b);
		else if constexpr (FS.has(SSE) && xmm_sized<T> && sizeof(S) == 4) return T::fromBits(_mm_unpackhi_ps(vcast<__m128>(a), vcast<__m128>(b)));

		else if constexpr (sizeof(T) > 16) return T{ unpackhi(a.lo(),b.lo()), unpackhi(a.hi(),b.hi()) };
		else
		{
			internals::scream();
			return internals::unpack_base<S, N, false>(a, b);
		}
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> compress(const mask_t<S, N>& mask, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& src)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		using T = SIMD_Vector<S, N>;

		if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) return vcast<S>(compress(mask, vcast<U>(a), vcast<U>(src)));

		else if constexpr (FS.has(AVX512_VBMI2) && zmm_sized<T> && any_i16<S>) return _mm512_mask_compress_epi16(src, mask, a);
		else if constexpr (FS.has(AVX512_VBMI2) && zmm_sized<T> && any_i8<S>) return _mm512_mask_compress_epi8(src, mask, a);
		else if constexpr (FS.has(AVX512_VBMI2) && FS.has(AVX512_VL) && ymm_sized<T> && any_i16<S>) return _mm256_mask_compress_epi16(src, mask, a);
		else if constexpr (FS.has(AVX512_VBMI2) && FS.has(AVX512_VL) && ymm_sized<T> && any_i8<S>) return _mm256_mask_compress_epi8(src, mask, a);
		else if constexpr (FS.has(AVX512_VBMI2) && FS.has(AVX512_VL) && xmm_sized<T> && any_i16<S>) return _mm_mask_compress_epi16(src, mask, a);
		else if constexpr (FS.has(AVX512_VBMI2) && FS.has(AVX512_VL) && xmm_sized<T> && any_i8<S>) return _mm_mask_compress_epi8(src, mask, a);

		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f64<S>) return _mm512_mask_compress_pd(src, mask, a);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f32<S>) return _mm512_mask_compress_ps(src, mask, a);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && any_i64<S>) return _mm512_mask_compress_epi64(src, mask, a);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && any_i32<S>) return _mm512_mask_compress_epi32(src, mask, a);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && is_f64<S>) return _mm256_mask_compress_pd(src, mask, a);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && is_f32<S>) return _mm256_mask_compress_ps(src, mask, a);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && any_i64<S>) return _mm256_mask_compress_epi64(src, mask, a);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && any_i32<S>) return _mm256_mask_compress_epi32(src, mask, a);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && xmm_sized<T> && is_f64<S>) return _mm_mask_compress_pd(src, mask, a);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && xmm_sized<T> && is_f32<S>) return _mm_mask_compress_ps(src, mask, a);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && xmm_sized<T> && any_i64<S>) return _mm_mask_compress_epi64(src, mask, a);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && xmm_sized<T> && any_i32<S>) return _mm_mask_compress_epi32(src, mask, a);

		else if constexpr (FS.has(AVX512_F) && (zmm_sized<T> || ((ymm_sized<T> || xmm_sized<T>) && FS.has(AVX512_VL))) && any_small_int<S>) return vrtrunc<S>(compress(mask, vrzext<uint32_t>(a), vrzext<uint32_t>(src)));
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && sizeof(S) == 4)
		{
			auto permx_ind = vcvt<U>(SIMD_Vector<int8_t, N>(_mm_loadu_si64(&tables::compress_to_permx8[mask])));
			auto tmp = permx(a, permx_ind); //permx_ind is setup in such a way that is can be used both as index register and blend mask without extra conversions
			if constexpr (is_f32<S>) return _mm256_blendv_ps(tmp, src, _mm256_castsi256_ps(permx_ind));
			else return _mm256_blendv_epi8(vcast<__m256i>(tmp), src, permx_ind);
		}
		else if constexpr (FS.has(SSSE3) && xmm_sized<T> && sizeof(S) == 4)
		{
			uint32_t maskb = mask;
			const int8_t* table_ptr = tables::compress_dwords_pshufb.data() + (maskb * 16);
			//negative ind = pass through src
			__m128i ind = _mm_load_si128(reinterpret_cast<const __m128i*>(table_ptr));
			SIMD_Vector<int8_t, 16> shuf = _mm_shuffle_epi8(vcast<__m128i>(a), ind);
			return vcast<S>(mask_mov(shuf, ind, vcast<int8_t>(src)));
		}
		else if constexpr (sizeof(T) > 16)
		{
			T ret = src;
			auto cl = compress(mask.lo(), a.lo());
			auto ch = compress(mask.hi(), a.hi());
			size_t popcnt_lo = std::popcount((uint64_t)mask.lo());
			size_t popcnt_hi = std::popcount((uint64_t)mask.hi());

			static_assert(N <= 64);
			float* p = (float*)&ret;
			store(cl, p);
			U cm = (uint64_t(1) << popcnt_hi) - 1;
			store(ch, p + popcnt_lo, cm); //don't overwrite src remains
			return ret;
		}
		else
		{
			internals::scream();
			SIMD_Vector<S, N> ret;
			size_t j = 0;
			for (size_t i = 0; i < N; ++i) if (mask[i]) ret[j++] = a[i];
			for (; j < N; ++j) ret[j] = src[j];
			return ret;
		}
	}
	template<typename S, size_t N>
		requires (sizeof(S) * 8 >= N)
	__forceinline SIMD_Vector<typename meta::ScalarTraits<S>::UintT, N> conflict(const SIMD_Vector<S, N>& a)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		using T = SIMD_Vector<S, N>;
		//TODO: allow bigger CD?
		//TODO: can make conflict detection for smaller scalar types too
		//TODO: update this for new architecture and change return type to bits_to_uint_t<N>
		//TODO: should it even allow floating point types?
		//TODO: if yes, then add an integer routing
		if constexpr (FS.has(AVX512_CD) && zmm_sized<T> && sizeof(S) == 4) return _mm512_conflict_epi32(vcast<int32_t>(a));
		else if constexpr (FS.has(AVX512_CD) && zmm_sized<T> && sizeof(S) == 8) return _mm512_conflict_epi64(vcast<int64_t>(a));
		else if constexpr (FS.has(AVX512_CD) && FS.has(AVX512_VL) && ymm_sized<T> && sizeof(S) == 4) return _mm256_conflict_epi32(vcast<int32_t>(a));
		else if constexpr (FS.has(AVX512_CD) && FS.has(AVX512_VL) && ymm_sized<T> && sizeof(S) == 8) return _mm256_conflict_epi64(vcast<int64_t>(a));
		else if constexpr (FS.has(AVX512_CD) && FS.has(AVX512_VL) && xmm_sized<T> && sizeof(S) == 4) return _mm_conflict_epi32(vcast<int32_t>(a));
		else if constexpr (FS.has(AVX512_CD) && FS.has(AVX512_VL) && xmm_sized<T> && sizeof(S) == 8) return _mm_conflict_epi64(vcast<int64_t>(a));
		//TODO: >64 byte CD
		//TODO: emulations for CD
		else
		{
			using UV = SIMD_Vector<U, N>;
			UV ret;
			ret[0] = 0;
			for (size_t i = 1; i < N; ++i)
			{
				U acc = 0;
				for (size_t j = 0; j < i; ++j)
				{
					if (a[i] == a[j]) acc |= U(1) << j;
				}
				ret[i] = acc;
			}
			return ret;
		}

	}

	template<meta::vpopcnt_allowed S, size_t N>
	SIMD_Vector<typename meta::ScalarTraits<S>::UintT, N> vpopcnt(const SIMD_Vector<S, N>& a)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		using T = SIMD_Vector<S, N>;

		auto popcnt8 = [&]() {
			auto cs = vcast<uint8_t>(a);
			auto table = load_a<uint8_t, cs.LaneCount>(tables::popcnt_table_for_nibbles_as_epi8.data());
			auto lo_nib = cs & 15;
			auto hi_nib = vcast<decltype(cs)>(shift_right<4>(vcast<uint32_t>(cs))) & 15;
			return byte_shuffle(table, lo_nib) + byte_shuffle(table, hi_nib);
			};
		if constexpr (!any_int<S>) return vpopcnt(vcast<U>(a));
		else if constexpr (FS.has(AVX512_VPOPCNTDQ) && zmm_sized<T> && any_i64<S>) return _mm512_popcnt_epi64(a);
		else if constexpr (FS.has(AVX512_VPOPCNTDQ) && zmm_sized<T> && any_i32<S>) return _mm512_popcnt_epi32(a);
		else if constexpr (FS.has(AVX512_BITALG) && zmm_sized<T> && any_i16<S>) return _mm512_popcnt_epi16(a);
		else if constexpr (FS.has(AVX512_BITALG) && zmm_sized<T> && any_i8<S>) return _mm512_popcnt_epi8(a);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_VPOPCNTDQ) && ymm_sized<T> && any_i64<S>) return _mm256_popcnt_epi64(a);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_VPOPCNTDQ) && ymm_sized<T> && any_i32<S>) return _mm256_popcnt_epi32(a);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BITALG) && ymm_sized<T> && any_i16<S>) return _mm256_popcnt_epi16(a);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BITALG) && ymm_sized<T> && any_i8<S>) return _mm256_popcnt_epi8(a);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_VPOPCNTDQ) && xmm_sized<T> && any_i64<S>) return _mm_popcnt_epi64(a);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_VPOPCNTDQ) && xmm_sized<T> && any_i32<S>) return _mm_popcnt_epi32(a);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BITALG) && xmm_sized<T> && any_i16<S>) return _mm_popcnt_epi16(a);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BITALG) && xmm_sized<T> && any_i8<S>) return _mm_popcnt_epi8(a);

		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && any_i64<S>) return _mm512_sad_epu8(vpopcnt(vcast<uint8_t>(a)), _mm512_setzero_si512());
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && any_i32<S>) return _mm512_madd_epi16(vpopcnt(vcast<uint16_t>(a)), _mm512_set1_epi16(1));
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && any_i16<S>) return _mm512_maddubs_epi16(vpopcnt(vcast<uint8_t>(a)), _mm512_set1_epi8(1));
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && any_i8<S>) return popcnt8();

		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i64<S>) return _mm256_sad_epu8(vpopcnt(vcast<uint8_t>(a)), _mm256_setzero_si256());
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i32<S>) return _mm256_madd_epi16(vpopcnt(vcast<uint16_t>(a)), _mm256_set1_epi16(1));
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i16<S>) return _mm256_maddubs_epi16(vpopcnt(vcast<uint8_t>(a)), _mm256_set1_epi8(1));
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i8<S>) return popcnt8();

		else if constexpr (FS.has(SSSE3) && xmm_sized<T> && any_i64<S>) return _mm_sad_epu8(vpopcnt(vcast<uint8_t>(a)), _mm_setzero_si128());
		else if constexpr (FS.has(SSSE3) && xmm_sized<T> && any_i32<S>) return _mm_madd_epi16(vpopcnt(vcast<uint16_t>(a)), _mm_set1_epi16(1));
		else if constexpr (FS.has(SSSE3) && xmm_sized<T> && any_i16<S>) return _mm_maddubs_epi16(vpopcnt(vcast<uint8_t>(a)), _mm_set1_epi8(1));
		else if constexpr (FS.has(SSSE3) && xmm_sized<T> && any_i8<S>) return popcnt8();

		else if constexpr (sizeof(S) > 16) return { vpopcnt(a.lo()), vpopcnt(a.hi()) };
		else
		{
			T ret;
			for (size_t i = 0; i < N; ++i) ret[i] = std::popcount(std::bit_cast<U>(a[i]));
			return ret;
		}
	}

	template<typename S, size_t N>
	__forceinline mask_t<S, N> movemask(const SIMD_Vector<S, N>& v)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		//if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) return movemask(vcast<U>(v));
		//TODO: implement this for AVX512 by comparing with zero as integers

		mask_t<S, N> ret;
		using Tr = meta::ScalarTraits<S>;
		using U = Tr::UintT;
		for (size_t i = 0; i < N; ++i)
		{
			U sb = std::bit_cast<U>(v[i]) & Tr::SignMask;
			ret.setBit(i, sb);
		}
		return ret;
	}

	template<typename S, meta::ScalarSizeClassEnum C, size_t N>
	__forceinline SIMD_Vector<S, N> movm(const internals::SIMD_Mask<C, N>& mask)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		using T = SIMD_Vector<S, N>;
		if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) return vcast<S>(movm<U>(mask));

		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && any_i8<S>) return _mm512_movm_epi8(mask);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && any_i16<S>) return _mm512_movm_epi16(mask);
		else if constexpr (FS.has(AVX512_DQ) && zmm_sized<T> && any_i32<S>) return _mm512_movm_epi32(mask);
		else if constexpr (FS.has(AVX512_DQ) && zmm_sized<T> && any_i64<S>) return _mm512_movm_epi64(mask);

		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && ymm_sized<T> && any_i8<S>) return _mm256_movm_epi8(mask);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && ymm_sized<T> && any_i16<S>) return _mm256_movm_epi16(mask);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_DQ) && ymm_sized<T> && any_i32<S>) return _mm256_movm_epi32(mask);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_DQ) && ymm_sized<T> && any_i64<S>) return _mm256_movm_epi64(mask);

		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && xmm_sized<T> && any_i8<S>) return _mm_movm_epi8(mask);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && xmm_sized<T> && any_i16<S>) return _mm_movm_epi16(mask);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_DQ) && xmm_sized<T> && any_i32<S>) return _mm_movm_epi32(mask);
		else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_DQ) && xmm_sized<T> && any_i64<S>) return _mm_movm_epi64(mask);

		else if constexpr (sizeof(T) > 16) return { movm<S>(mask.lo()), movm<S>(mask.hi()) };
		else
		{
			internals::scream();
			SIMD_Vector<S, N> ret;
			using Tr = meta::ScalarTraits<S>;
			for (size_t i = 0; i < N; ++i)
			{
				typename Tr::UintT u = mask[i] ? Tr::AllOnesUint : 0;
				ret[i] = std::bit_cast<S>(u);
			}
			return ret;
		}

	}

	template<typename S, size_t N>
	SIMD_Vector<S, N> byte_shuffle(const SIMD_Vector<S, N>& a, const SIMD_Vector<uint8_t, N * sizeof(S)>& b)
	{
		using namespace meta;
		using namespace internals;
		using T = SIMD_Vector<uint8_t, N>;

		if constexpr (!is_u8<S>) return vcast<S>(byte_shuffle(vcast<uint8_t>(a), b));
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T>) return _mm512_shuffle_epi8(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T>) return _mm256_shuffle_epi8(a, b);
		else if constexpr (FS.has(SSSE3) && xmm_sized<T>) return _mm_shuffle_epi8(a, b);
		else if constexpr (sizeof(T) > 16) return { byte_shuffle(a.lo(),b.lo()), byte_shuffle(a.hi(),b.hi()) };
		else
		{
			T ret;
			for (size_t start = 0; start < N; start += 16)
				for (size_t i = 0; i < std::min<size_t>(N - start, 16); ++i)
					ret[start + i] = b[start + i] > 127 ? 0 : a[start + (b[i] & 15)];
			return ret;
		}
	}

	template<typename S, size_t N, size_t Scale, meta::any_int I>
	__forceinline SIMD_Vector<S, N> __gather_impl(const void* p, const SIMD_Vector<I, N>& ind, const mask_t<S,N>& mask, const SIMD_Vector<S, N>& src)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) return vcast<S>(__gather_impl<U, N, Scale, I>(p, ind, mask, vcast<U>(src)));
		else
		{
			//put everything up here to prevent else if chain breaks (since compilation gives useless errors by thinking unsanitized inputs surviving to native gathers
			using namespace meta;
			using CanonicalIndex_t = std::conditional_t<(sizeof(I) <= 4), int32_t, int64_t>;
			using RetVec_t = SIMD_Vector<S, N>;
			using IndVec_t = SIMD_Vector<I, N>;
			constexpr size_t MaxSize = std::max(sizeof(RetVec_t), sizeof(IndVec_t));

			//MSVC wants proper types, not void, Clang is more lenient
			using intr_base_ptr_t = std::conditional_t<is_f64<S>, double*,
				std::conditional_t<is_f32<S>, float*, int32_t*>>;
			const intr_base_ptr_t base = (const intr_base_ptr_t)p;

			//if scale is not native, emulate it by gathering with scale 1 and manually calculated byte offsets. 
			if constexpr (Scale != 1 && Scale != 2 && Scale != 4 && Scale != 8)
			{
				using SN = GatherScatterScaleSanitizer<Scale, I>;
				return gather<S, N, SN::newScale>(base, vcvt<typename SN::extended_t>(ind) * SN::indexMultiplier, mask, src);
			}

			//TODO: emulation of small int gathers (where elements gathered are small ints)
			else if constexpr (!std::is_same_v<I, CanonicalIndex_t>) return gather<S, N, Scale>(base, vcvt<CanonicalIndex_t>(ind), mask, src);

			//SANITIZATION DONE
			//if we get here, means that indices are already in good format (4-byte or 8-byte)
			//and scale is good too.
			//break up large gather into halves
			else if constexpr (MaxSize > 64) return RetVec_t{
				gather<S, N / 2, Scale, I>(base, ind.lo(), mask.lo(), src.lo()),
				gather<S, N / 2, Scale, I>(base, ind.hi(), mask.hi(), src.hi()) };

			else
			{
				if constexpr (FS.has(AVX512_F))
				{
					if constexpr (is_zmm_size(MaxSize))
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
					}
					if constexpr (FS.has(AVX512_VL))
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
						}
					}
				}
				if constexpr (FS.has(AVX2))
				{
					if constexpr (MaxSize > 32)
					{
						return {
							gather<S, N / 2, Scale, I>(base, ind.lo(), mask.lo(), src.lo()),
							gather<S, N / 2, Scale, I>(base, ind.hi(), mask.hi(), src.hi())
						};
					}
					else if constexpr (is_ymm_size(MaxSize))
					{
						//clang is a cry-baby with ind here for some reason, so force convert it. Pay attention to size!
						std::conditional_t<(meta::ymm_sized<IndVec_t>), __m256i, __m128i> ni = ind;
						if constexpr (is_i64<I> && is_f64<S>) return _mm256_mask_i64gather_pd(src, base, ni, mask, Scale);
						else if constexpr (is_i64<I> && is_f32<S>) return _mm256_mask_i64gather_ps(src, base, ni, mask, Scale);
						else if constexpr (is_i64<I> && any_i64<S>) return _mm256_mask_i64gather_epi64(src, base, ni, mask, Scale);
						else if constexpr (is_i64<I> && any_i32<S>) return _mm256_mask_i64gather_epi32(src, base, ni, mask, Scale);

						else if constexpr (is_i32<I> && is_f64<S>) return _mm256_mask_i32gather_pd(src, base, ni, mask, Scale);
						else if constexpr (is_i32<I> && is_f32<S>) return _mm256_mask_i32gather_ps(src, base, ni, mask, Scale);
						else if constexpr (is_i32<I> && any_i64<S>) return _mm256_mask_i32gather_epi64(src, base, ni, mask, Scale);
						else if constexpr (is_i32<I> && any_i32<S>) return _mm256_mask_i32gather_epi32(src, base, ni, mask, Scale);
					}
					else if constexpr (is_xmm_size(MaxSize))
					{
						__m128i ni = ind;
						if constexpr (is_i64<I> && is_f64<S>) return _mm_mask_i64gather_pd(src, base, ni, mask, Scale);
						else if constexpr (is_i64<I> && is_f32<S>) return _mm_mask_i64gather_ps(src, base, ni, mask, Scale);
						else if constexpr (is_i64<I> && any_i64<S>) return _mm_mask_i64gather_epi64(src, base, ni, mask, Scale);
						else if constexpr (is_i64<I> && any_i32<S>) return _mm_mask_i64gather_epi32(src, base, ni, mask, Scale);

						else if constexpr (is_i32<I> && is_f64<S>) return _mm_mask_i32gather_pd(src, base, ni, mask, Scale);
						else if constexpr (is_i32<I> && is_f32<S>) return _mm_mask_i32gather_ps(src, base, ni, mask, Scale);
						else if constexpr (is_i32<I> && any_i64<S>) return _mm_mask_i32gather_epi64(src, base, ni, mask, Scale);
						else if constexpr (is_i32<I> && any_i32<S>) return _mm_mask_i32gather_epi32(src, base, ni, mask, Scale);
					}
				}

				internals::scream();
				SIMD_Vector<S, N> ret;
				size_t addr = size_t(base);
				for (size_t i = 0; i < N; ++i) ret[i] = mask[i] ? *(const S*)(addr + Scale * ind[i]) : src[i];
				return ret;
			}
		}
	}

	template<typename Block, size_t... Idx, typename S, size_t N>
	//requires (meta::IsScalarType<Block> || meta::IsSimdVector<Block>)
	SIMD_Vector<S, N> permute(const SIMD_Vector<S, N>& a)
	{
		static_assert(meta::IsScalarType<Block> || meta::IsSimdVector<Block>, "permute block type must be scalar or vector");

		using namespace meta;
		using namespace internals;
		using T = SIMD_Vector<S, N>;

		constexpr size_t atomSize = sizeof(Block);
		constexpr size_t idxCount = sizeof...(Idx);
		constexpr size_t atomCount = sizeof(T) / atomSize;

		static_assert(sizeof(T) % atomSize == 0, "permute vector size must be divisible by block size");
		static_assert(atomCount == idxCount, "permute index count must match count of blocks in the input vector");

		T ret;
		constexpr size_t indices[] = { Idx... };

		constexpr bool indices_valid = []() {
			for (size_t i = 0; i < atomCount; ++i) if (indices[i] >= atomCount) return false;
			return true;
			}();
		static_assert(indices_valid, "permute block indices must be less than twice the block count in the input type");

		const auto* src = reinterpret_cast<const std::byte*>(&a);
		auto* dst = reinterpret_cast<std::byte*>(&ret);

		for (size_t i = 0; i < atomCount; ++i)
		{
			size_t ind = indices[i];
			memcpy(dst + i * atomSize, src + ind * atomSize, atomSize);
		}
		return ret;
	}
	template<typename Block, size_t ...Idx, typename S, size_t N>
	SIMD_Vector<S, N> permute2(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		static_assert(meta::IsScalarType<Block> || meta::IsSimdVector<Block>, "permute2 block type must be scalar or vector");

		using namespace meta;
		using namespace internals;
		using T = SIMD_Vector<S, N>;

		constexpr size_t atomSize = sizeof(Block);
		constexpr size_t idxCount = sizeof...(Idx);
		constexpr size_t atomCount = sizeof(T) / atomSize;

		static_assert(sizeof(T) % atomSize == 0, "permute2 vector size must be divisible by block size");
		static_assert(atomCount == idxCount, "permute2 index count must match count of blocks in the input vector");

		T ret;
		constexpr size_t indices[] = { Idx... };

		constexpr bool indices_valid = []() {
			for (size_t i = 0; i < atomCount; ++i) if (indices[i] >= atomCount * 2) return false;
			return true;
			}();
		static_assert(indices_valid, "permute2 block indices must be less than twice the block count in the input type");

		const auto* src1 = reinterpret_cast<const std::byte*>(&a);
		const auto* src2 = reinterpret_cast<const std::byte*>(&b);
		auto* dst = reinterpret_cast<std::byte*>(&ret);

		for (size_t i = 0; i < atomCount; ++i)
		{
			size_t ind = indices[i];
			const auto* src = ind < atomCount ? (src1 + ind * atomSize) : (src2 + (ind - atomCount) * atomSize);
			memcpy(dst + i * atomSize, src, atomSize);
		}
		return ret;
	}
}