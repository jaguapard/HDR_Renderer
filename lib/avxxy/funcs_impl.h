#pragma once
#include "funcs.h"
#include "SIMD_Vector.h"
#include "FeatureSet.h"
#include <source_location>


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






	//#define AVXXY_RUN(op) internals::Dispatcher::run<internals::op>
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> add(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace internals;
		using namespace meta;
		using T = SIMD_Vector<S, N>;

		if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f64<S>) return _mm512_add_pd(a, b);
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

		if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f64<S>) return _mm512_sub_pd(a, b);
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

		if constexpr (any_i8<S>) return vcvt<S>(mul(vcvt<canon_t>(a), vcvt<canon_t>(b))); //no 8 bit mul as of last AVX512, so emulate

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

		else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_f64<S>) return _mm_mul_pd(a, b);
		else if constexpr (FS.has(SSE) && xmm_sized<T> && is_f32<S>) return _mm_mul_ps(a, b);
		else if constexpr (FS.has(SSE41) && xmm_sized<T> && any_i32<S>) return _mm_mullo_epi32(a, b);
		else if constexpr (FS.has(SSE2) && xmm_sized<T> && any_i16<S>) return _mm_mullo_epi16(a, b);

		else if constexpr (FS.has(SSE2) && xmm_sized<T> && any_i64<S>) //TODO: check if it works. 256-bit version does
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

		if constexpr (any_i32<S>) return vcvt<S>(div(vcvt<double>(a), vcvt<double>(b))); //emulate 32 bit integer division via double precision division
		else if constexpr (any_small_int<S>) return vcvt<S>(div(vcvt<float>(a), vcvt<float>(b))); //emulate small integer division via single precision division

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
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T>) return _mm512_and_si512(vreinterpret_us<__m512i>(a), vreinterpret_us<__m512i>(b));

		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_int<S>) return _mm256_and_si256(vreinterpret_us<__m256i>(a), vreinterpret_us<__m256i>(b));
		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f64<S>) return _mm256_and_pd(vreinterpret_us<__m256d>(a), vreinterpret_us<__m256d>(b));
		else if constexpr (FS.has(AVX) && ymm_sized<T>) return _mm256_and_ps(vreinterpret_us<__m256>(a), vreinterpret_us<__m256>(b));

		else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_f64<S>) return _mm_and_pd(vreinterpret_us<__m128d>(a), vreinterpret_us<__m128d>(b));
		else if constexpr (FS.has(SSE2) && xmm_sized<T>) return _mm_and_si128(vreinterpret_us<__m128i>(a), vreinterpret_us<__m128i>(b));
		else if constexpr (FS.has(SSE) && xmm_sized<T>) return _mm_and_ps(vreinterpret_us<__m128>(a), vreinterpret_us<__m128>(b));

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
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T>) return _mm512_or_si512(vreinterpret_us<__m512i>(a), vreinterpret_us<__m512i>(b));

		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_int<S>) return _mm256_or_si256(vreinterpret_us<__m256i>(a), vreinterpret_us<__m256i>(b));
		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f64<S>) return _mm256_or_pd(vreinterpret_us<__m256d>(a), vreinterpret_us<__m256d>(b));
		else if constexpr (FS.has(AVX) && ymm_sized<T>) return _mm256_or_ps(vreinterpret_us<__m256>(a), vreinterpret_us<__m256>(b));

		else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_f64<S>) return _mm_or_pd(vreinterpret_us<__m128d>(a), vreinterpret_us<__m128d>(b));
		else if constexpr (FS.has(SSE2) && xmm_sized<T>) return _mm_or_si128(vreinterpret_us<__m128i>(a), vreinterpret_us<__m128i>(b));
		else if constexpr (FS.has(SSE) && xmm_sized<T>) return _mm_or_ps(vreinterpret_us<__m128>(a), vreinterpret_us<__m128>(b));

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
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T>) return _mm512_xor_si512(vreinterpret_us<__m512i>(a), vreinterpret_us<__m512i>(b));

		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_int<S>) return _mm256_xor_si256(vreinterpret_us<__m256i>(a), vreinterpret_us<__m256i>(b));
		else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f64<S>) return _mm256_xor_pd(vreinterpret_us<__m256d>(a), vreinterpret_us<__m256d>(b));
		else if constexpr (FS.has(AVX) && ymm_sized<T>) return _mm256_xor_ps(vreinterpret_us<__m256>(a), vreinterpret_us<__m256>(b));

		else if constexpr (FS.has(SSE2) && xmm_sized<T> && is_f64<S>) return _mm_xor_pd(vreinterpret_us<__m128d>(a), vreinterpret_us<__m128d>(b));
		else if constexpr (FS.has(SSE2) && xmm_sized<T>) return _mm_xor_si128(vreinterpret_us<__m128i>(a), vreinterpret_us<__m128i>(b));
		else if constexpr (FS.has(SSE) && xmm_sized<T>) return _mm_xor_ps(vreinterpret_us<__m128>(a), vreinterpret_us<__m128>(b));

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
		return logic_xor(a, std::bit_cast<S>(meta::ScalarTraits<S>::AllOnesUint));
		/*
		using U = typename ScalarTraits<S>::UintT;

		internals::scream();
		SIMD_Vector<S, N> ret;
		using T = typename meta::ScalarTraits<S>::UintT;
		for (size_t i = 0; i < N; ++i) ret[i] = std::bit_cast<S>(T(~std::bit_cast<T>(a[i])));
		return ret;*/
	}
	template<typename S, size_t N, typename I> requires (meta::any_int<S>&& meta::any_int<I>)
		__forceinline SIMD_Vector<S, N> shift_left(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& b)
	{
		using namespace internals;
		using namespace meta;
		using T = SIMD_Vector<S, N>;
		using canon_t = typename ScalarTraits<S>::UintT;

		if constexpr (!std::is_same_v<I, canon_t>) return shift_left(a, vcvt<canon_t>(b));

		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && any_i16<S>) return _mm512_sllv_epi16(a, b);
		else if constexpr (FS.has(AVX512_BW) && FS.has(AVX512_VL) && ymm_sized<T> && any_i16<S>) return _mm256_sllv_epi16(a, b);
		else if constexpr (FS.has(AVX512_BW) && FS.has(AVX512_VL) && xmm_sized<T> && any_i16<S>) return _mm_sllv_epi16(a, b);
		else if constexpr (FS.has(AVX512_BW) && FS.has(AVX512_VL) && any_i8<S>) return vrtrunc<S>(shift_left(vrzext<uint16_t>(a)));
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && any_i64<S>) return _mm512_sllv_epi64(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && any_i32<S>) return _mm512_sllv_epi32(a, b);

		else if constexpr (FS.has(AVX2) && sizeof(T) <= 32 && any_small_int<S>) //zero-extend small integers, shift and convert back. TODO: There could be a better way?
		{
			auto a32 = vrzext<uint32_t>(a);
			auto sh = shift_left(a32, b);
			return vrtrunc<S>(sh);
		}
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
	template<typename S, size_t N, typename I> requires (meta::any_int<S>&& meta::any_int<I>)
		__forceinline SIMD_Vector<S, N> shift_right(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& b)
	{
		using namespace internals;
		using namespace meta;
		using T = SIMD_Vector<S, N>;
		using canon_t = typename ScalarTraits<S>::UintT;

		if constexpr (!std::is_same_v<I, canon_t>) return shift_right(a, vcvt<canon_t>(b));

		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && any_i16<S>) return _mm512_srlv_epi16(a, b);
		else if constexpr (FS.has(AVX512_BW) && FS.has(AVX512_VL) && ymm_sized<T> && any_i16<S>) return _mm256_srlv_epi16(a, b);
		else if constexpr (FS.has(AVX512_BW) && FS.has(AVX512_VL) && xmm_sized<T> && any_i16<S>) return _mm_srlv_epi16(a, b);
		else if constexpr (FS.has(AVX512_BW) && FS.has(AVX512_VL) && any_i8<S>) return vrtrunc<S>(shift_right(vrzext<uint16_t>(a)));
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && any_i64<S>) return _mm512_srlv_epi64(a, b);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && any_i32<S>) return _mm512_srlv_epi32(a, b);

		else if constexpr (FS.has(AVX2) && sizeof(T) <= 32 && any_small_int<S>) //zero-extend small integers, shift and convert back. TODO: There could be a better way?
		{
			auto a32 = vrzext<uint32_t>(a);
			auto sh = shift_right(a32, b);
			return vrtrunc<S>(sh);
		}
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i64<S>) return _mm256_srlv_epi64(a, b);
		else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i32<S>) return _mm256_srlv_epi32(a, b);
		else if constexpr (FS.has(AVX2) && xmm_sized<T> && any_i64<S>) return _mm_srlv_epi64(a, b); //no shifts in SSE!
		else if constexpr (FS.has(AVX2) && xmm_sized<T> && any_i32<S>) return _mm_srlv_epi32(a, b);

		else
		{
			internals::scream();
			SIMD_Vector<S, N> ret;
			//using T = typename concepts::same_size_uint_t<S>::type;
			for (size_t i = 0; i < N; ++i) ret[i] = a[i] >> b[i];
			return ret;
		}

	}
	template<typename S, size_t N, typename I> requires (meta::any_int<I>)
	__forceinline SIMD_Vector<S, N> permx(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& ind)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		using canon_t = typename ScalarTraits<S>::UintT;
		using T = SIMD_Vector<S, N>;

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
			using X = std::conditional_t<is_f64<S>, float, uint32_t>;
			auto a32 = vcast<X>(a);
			__m256i ind32 = _mm256_castps_si256(_mm256_moveldup_ps(_mm256_castsi256_ps(ind)));//duplicate each low 32-bits of 64-bit index element into 32-bit lanes. Wrap around and power of 2 vector size limitation allow this to work.
			auto perm = permx(a32, _mm256_or_si256(_mm256_slli_epi32(ind32, 1), _mm256_setr_epi32(0, 1, 0, 1, 0, 1, 0, 1))); //index is multiplied by 2 and added to alterating 0, 1, emulating 64 bit behavior
			return vcast<S>(perm);
		}

		else if constexpr (FS.has(AVX) && xmm_sized<T> && sizeof(S) == 4) return _mm_permutevar_ps(vreinterpret_us<__m128>(a), ind);
		else if constexpr (FS.has(AVX) && xmm_sized<T> && sizeof(S) == 8) return _mm_permutevar_pd(vreinterpret_us<__m128d>(a), ind);

		//TODO: these may break with >127 bytes. Also check if they work at all
		else if constexpr (FS.has(SSSE3) && xmm_sized<T> && sizeof(S) == 1) return _mm_shuffle_epi8(a, ind & 0x7F); //discard sign bit to avoid unwanted zero-masking
		else if constexpr (FS.has(SSSE3) && xmm_sized<T> && sizeof(S) == 2)
		{
			__m128i ind2 = _mm_slli_epi16(ind, 1);
			__m128i db = _mm_shuffle_epi8(ind2, _mm_setr_epi8(0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14)); //duplicate low byte of each word
			__m128i ind3 = _mm_or_si128(db, _mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1));
			__m128i ind4 = _mm_and_si128(ind3, _mm_set1_epi8(0x7F));
			return T::from_bits_us(_mm_shuffle_epi8(vreinterpret_us<__m128i>(a), ind4));
		}
		else if constexpr (FS.has(SSSE3) && xmm_sized<T> && sizeof(S) == 4)
		{
			__m128i ind2 = _mm_slli_epi32(ind, 2);
			__m128i db = _mm_shuffle_epi8(ind2, _mm_setr_epi8(0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12)); //duplicate low byte of each dword
			__m128i ind3 = _mm_or_si128(db, _mm_setr_epi8(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3));
			__m128i ind4 = _mm_and_si128(ind3, _mm_set1_epi8(0x7F));
			return T::from_bits_us(_mm_shuffle_epi8(vreinterpret_us<__m128i>(a), ind4));
		}
		else if constexpr (FS.has(SSSE3) && xmm_sized<T> && sizeof(S) == 8)
		{
			__m128i ind2 = _mm_slli_epi64(ind, 3);
			__m128i db = _mm_shuffle_epi8(ind2, _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8)); //duplicate low byte of each qdword
			__m128i ind3 = _mm_or_si128(db, _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7));
			__m128i ind4 = _mm_and_si128(ind3, _mm_set1_epi8(0x7F));
			return T::from_bits_us(_mm_shuffle_epi8(vreinterpret_us<__m128i>(a), ind4));
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
	template<typename S, size_t N, typename I> requires (meta::any_int<I>)
	__forceinline SIMD_Vector<S, N> permx2(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b, const SIMD_Vector<I, N>& ind)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		using canon_t = typename ScalarTraits<S>::UintT;
		using T = SIMD_Vector<S, N>;

		if constexpr (!is_f64<S> && !is_f32<S> && !any_int<S>) return vcast<S>(permx2(vcast<U>(a), vcast<U>(b), ind));
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
		else if constexpr (true || sizeof(T) > 16) //TODO: seems to work, but very sus. Although, what else to do?
		{
			T pa = permx(a, ind);
			T pb = permx(b, ind);
			return mask_mov(pb, (ind & (2 * N - 1)) < N, pa);
		}
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

		if constexpr (!is_f32<S>) return sqrtd(vcvt<double>(a));
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


	template<typename To, size_t N, typename From>
	__forceinline SIMD_Vector<To, N> vcvt(const SIMD_Vector<From, N>& a)
	{
		using namespace meta;
		using namespace internals;
		using TV = SIMD_Vector<To, N>;
		using FV = SIMD_Vector<From, N>;
		constexpr size_t MaxSize = std::max(sizeof(TV), sizeof(FV));

		//Route all FP16 conversions to it's only friend - float
		if constexpr ((is_fp16<From> && !is_f32<To>) || (!is_f32<From> && is_fp16<To>)) return vcvt<To>(vcvt<float>(a));
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

		//TODO: add AVX2, AVX, SSE cvts
		else if constexpr (MaxSize > 16) return TV{ vcvt<To>(a.lo()), vcvt<To>(a.hi()) };
		else
		{
			internals::scream();
			SIMD_Vector<To, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = a[i];
			return ret;
		}
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
			using U = meta::ScalarTraits<S>::UintT;
			using U2 = meta::ScalarTraits<S2>::UintT;
			auto ex = vcvt<U2>(vcast<U>(a));
			return vcast<S2>(ex);
		}
	}

	template<typename S2, typename S, size_t N> requires (meta::IsScalarType<S2> && (sizeof(SIMD_Vector<S, N>) % sizeof(S2) == 0))
		__forceinline SIMD_Vector<S2, sizeof(SIMD_Vector<S, N>) / sizeof(S2)> vcast(const SIMD_Vector<S, N>& a)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		return vreinterpret_us<SIMD_Vector<S2, sizeof(SIMD_Vector<S, N>) / sizeof(S2)>>(a);
	}
	template<typename T, typename S, size_t N>
		requires (meta::IsSimdVector<T> && (sizeof(SIMD_Vector<S, N>) % sizeof(typename T::ScalarT) == 0) && sizeof(SIMD_Vector<S, N>) == sizeof(T))
	__forceinline T vcast(const SIMD_Vector<S, N>& a)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		return vreinterpret_us<T>(a);
	}


	template<typename T, typename S, size_t N> requires (sizeof(T) == sizeof(SIMD_Vector<S, N>))
		__forceinline T vreinterpret(const SIMD_Vector<S, N>& value)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		return vreinterpret_us<T>(value);
	}
	template<typename T, typename S, size_t N>
	__forceinline T vreinterpret_us(const SIMD_Vector<S, N>& value)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		T ret;
		memcpy(&ret, &value, std::min(sizeof(ret), sizeof(value)));
		return ret;
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
		else if constexpr (FS.has(AVX) && ymm_sized<T> && sizeof(S) == 8) return _mm256_blendv_pd(vreinterpret_us<__m256d>(ifBitClear), vreinterpret_us<__m256d>(ifBitSet), mask);
		else if constexpr (FS.has(AVX) && ymm_sized<T> && sizeof(S) == 4) return _mm256_blendv_ps(vreinterpret_us<__m256>(ifBitClear), vreinterpret_us<__m256>(ifBitSet), mask);

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

		auto ld = [&]() {
			if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f64<S>) return _mm512_loadu_pd(p);
			else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f32<S>) return _mm512_loadu_ps(p);
			else if constexpr (FS.has(AVX512_F) && zmm_sized<T>) return _mm512_loadu_si512(p);
			else if constexpr (FS.has(AVX) && ymm_sized<T> && any_int<S>) return _mm256_loadu_si256(reinterpret_cast<const __m256i_u*>(p));
			else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f64<S>) return _mm256_loadu_pd(reinterpret_cast<const double*>(p));
			else if constexpr (FS.has(AVX) && ymm_sized<T>) return _mm256_loadu_ps(reinterpret_cast<const float*>(p));
			else if constexpr (FS.has(SSE2) && xmm_sized<T> && any_int<S>) return _mm_loadu_si128(reinterpret_cast<const __m128i_u*>(p));
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
		return T::from_bits_us(ld());		
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
		return T::from_bits_us(ld());
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
			else if constexpr (FS.has(AVX) && ymm_sized<T> && is_f64<S>) return _mm256_maskload_pd(reinterpret_cast<const double*>(p), mask);
			else if constexpr (FS.has(AVX) && ymm_sized<T> && sizeof(S) == 4) return _mm256_maskload_ps(reinterpret_cast<const float*>(p), mask);

			else if constexpr (FS.has(AVX2) && xmm_sized<T> && any_i64<S>) return _mm_maskload_epi64(reinterpret_cast<const int64_t*>(p), mask);
			else if constexpr (FS.has(AVX2) && xmm_sized<T> && any_i32<S>) return _mm_maskload_epi64(reinterpret_cast<const int32_t*>(p), mask);
			else if constexpr (FS.has(AVX) && xmm_sized<T> && is_f64<S>) return _mm_maskload_pd(reinterpret_cast<const double*>(p), mask);
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
		else return T::from_bits_us(zload());
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

	template<typename S, size_t N, size_t Scale, typename I> requires (meta::any_int<I>)
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
		//std::conditional_t<MaxSize <= 64, intr_t, int> ni = MaxSize <= 64 ? vreinterpret_us<intr_t>(ind) : 0;
		intr_t ni = ind;

		if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) scatter<S, N, Scale, I>(vcast<U>(v), base, ind, mask);
		//if scale is not native, emulate it by gathering with scale 1 and manually calculated byte offsets. 
		//TODO: Can optimize a little by checking if Scale*maxint(I) fits into smaller sizes
		else if constexpr (Scale != 1 && Scale != 2 && Scale != 4 && Scale != 8) return scatter<S, N, 1>(v, base, vcvt<int64_t>(ind) * Scale, mask);

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
			typename SIMD_Vector<S, N>::MaskT ret = 0;
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
		if constexpr (std::is_unsigned_v<S>) return a;
		//TODO: can add fallback for FP types by forcing sign bit to zero? Even if they're unsupported (i.e. FP16 on SSE)

		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_i16<S>) return _mm512_abs_epi16(a);
		else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && is_i8<S>) return _mm512_abs_epi8(a);

		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f64<S>) return _mm512_abs_pd(a);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f32<S>) return _mm512_abs_ps(a);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_i64<S>) return _mm512_abs_epi64(a);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_i32<S>) return _mm512_abs_epi32(a);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && ymm_sized<T> && is_i64<S>) return _mm256_abs_epi64(a);
		else if constexpr (FS.has(AVX512_F) && FS.has(AVX512_VL) && xmm_sized<T> && is_i64<S>) return _mm_abs_epi64(a);
		
		
		else if constexpr (sizeof(T) > 16) return T{ abs(a.lo()), abs(a.hi()) };
		else
		{
			internals::scream();
			SIMD_Vector<S, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = std::abs(a[i]);
			return ret;
		}
	}

	template<typename S, size_t N>
		requires (meta::any_float<S>)
	__forceinline SIMD_Vector<S, N> floor(const SIMD_Vector<S, N>& a)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		using T = SIMD_Vector<S, N>;
		
		if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f64<S>) return _mm512_floor_pd(a);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f32<S>) return _mm512_floor_ps(a);
		else if constexpr (sizeof(T) > 16) return T{ floor(a.lo()), floor(a.hi()) };
		else 
		{
			internals::scream();
			SIMD_Vector<S, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = std::floor(a[i]);
			return ret;
		}
	}
	template<typename S, size_t N>
		requires (meta::any_float<S>)
	__forceinline SIMD_Vector<S, N> ceil(const SIMD_Vector<S, N>& a)
	{
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		using T = SIMD_Vector<S, N>;

		if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f64<S>) return _mm512_ceil_pd(a);
		else if constexpr (FS.has(AVX512_F) && zmm_sized<T> && is_f32<S>) return _mm512_ceil_ps(a);
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

		else if constexpr (sizeof(T) > 16) return T{ min(a.lo(), b.lo()), min(a.hi(),b.hi()) };
		else
		{
			internals::scream();
			SIMD_Vector<S, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = std::min(a[i], b[i]);
			return ret;
		}
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

		else if constexpr (sizeof(T) > 16) return T{ max(a.lo(), b.lo()), max(a.hi(),b.hi()) };
		else
		{
			internals::scream();
			SIMD_Vector<S, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = std::max(a[i], b[i]);
			return ret;
		}
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

		else if constexpr (sizeof(T) > 16) return T{ unpacklo(a.lo(),b.lo()), unpacklo(a.hi(),b.hi()) };
		else
		{
			internals::scream();
			return internals::unpack_base<S, N, true>(a, b);
		}
	}
	template<typename S, size_t N>
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
		//TODO: compress emulation for bytes and words by extending for AVX512 F
		//TODO: compress splitting via overlapping stores
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
		if constexpr (FS.has(AVX512_CD) && zmm_sized<T> && sizeof(S) == 4) return _mm512_conflict_epi32(vcast<SIMD_Vector<int32_t, N>>(a));
		else if constexpr (FS.has(AVX512_CD) && zmm_sized<T> && sizeof(S) == 8) return _mm512_conflict_epi64(vcast<SIMD_Vector<int64_t, N>>(a));
		else if constexpr (FS.has(AVX512_CD) && FS.has(AVX512_VL) && ymm_sized<T> && sizeof(S) == 4) return _mm256_conflict_epi32(vcast<SIMD_Vector<int32_t, N>>(a));
		else if constexpr (FS.has(AVX512_CD) && FS.has(AVX512_VL) && ymm_sized<T> && sizeof(S) == 8) return _mm256_conflict_epi64(vcast<SIMD_Vector<int64_t, N>>(a));
		else if constexpr (FS.has(AVX512_CD) && FS.has(AVX512_VL) && xmm_sized<T> && sizeof(S) == 4) return _mm_conflict_epi32(vcast<SIMD_Vector<int32_t, N>>(a));
		else if constexpr (FS.has(AVX512_CD) && FS.has(AVX512_VL) && xmm_sized<T> && sizeof(S) == 8) return _mm_conflict_epi64(vcast<SIMD_Vector<int64_t, N>>(a));
		//TODO: >64 byte CD

		else
		{
			using UV = SIMD_Vector<U, N>;
			UV ret;
			for (size_t i = 0; i < N; ++i)
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
	__forceinline SIMD_Vector<S, N> movm(const SIMD_Mask<C, N>& mask)
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

	template<typename S, size_t N, size_t Scale, typename I>
	__forceinline SIMD_Vector<S, N> __gather_impl(const void* p, const SIMD_Vector<I, N>& ind, const typename SIMD_Vector<S, N>::MaskT& mask, const SIMD_Vector<S, N>& src)
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
			//TODO: Can optimize a little by checking if Scale*maxint(I) fits into smaller sizes
			if constexpr (Scale != 1 && Scale != 2 && Scale != 4 && Scale != 8) return gather<S, N, 1>(base, vcvt<int64_t>(ind) * Scale, mask, src);

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
}