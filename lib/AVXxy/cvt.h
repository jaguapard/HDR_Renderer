#pragma once
#include "SIMD_Vector.h"
#include "funcs.h"

namespace AVXXY_NAMESPACE
{
	//Various internal functions of the library that are not meant to be used by the users.
	namespace internals
	{
		//This function is used internally and lacks upstream input sanitization. If you're looking for vector conversions, use cvt instead
		template<typename To, size_t N, typename From>
		__forceinline SIMD_Vector<To, N> zmm_cvt(const SIMD_Vector<From, N>& value)
		requires (inRange(std::max(sizeof(SIMD_Vector<From, N>), sizeof(SIMD_Vector<To, N>)), 33, 64))
		{
			using TV = SIMD_Vector<To, N>;
			//Truncation is REQUIRED for FP -> int
			if constexpr (std::is_same_v<From, double> && std::is_same_v<To, uint64_t>)  return TV(_mm512_cvttpd_epu64(value));
			else if constexpr (std::is_same_v<From, double> && std::is_same_v<To, int64_t>)  return TV(_mm512_cvttpd_epi64(value));
			else if constexpr (std::is_same_v<From, double> && std::is_same_v<To, float>)  return TV(_mm512_cvtpd_ps(value));
			else if constexpr (std::is_same_v<From, double> && std::is_same_v<To, uint32_t>)  return TV(_mm512_cvttpd_epu32(value));
			else if constexpr (std::is_same_v<From, double> && std::is_same_v<To, int32_t>)  return TV(_mm512_cvttpd_epi32(value));

			else if constexpr (std::is_same_v<From, float> && std::is_same_v<To, double>) return TV(_mm512_cvtps_pd(value));
			else if constexpr (std::is_same_v<From, float> && std::is_same_v<To, uint64_t>) return TV(_mm512_cvttps_epu64(value));
			else if constexpr (std::is_same_v<From, float> && std::is_same_v<To, int64_t>) return TV(_mm512_cvttps_epi64(value));
			else if constexpr (std::is_same_v<From, float> && std::is_same_v<To, uint32_t>) return TV(_mm512_cvttps_epu32(value));
			else if constexpr (std::is_same_v<From, float> && std::is_same_v<To, int32_t>) return TV(_mm512_cvttps_epi32(value));
			//TODO: FP16 conversion
			//else if constexpr (std::is_same_v<From, float16_t> && std::is_same_v<To, float>) return TV(_mm512_cvtph_ps(value));

			//only float conversions care about the signedness
			else if constexpr (std::is_same_v<From, uint64_t> && std::is_same_v<To, double>) return TV(_mm512_cvtepu64_pd(value));
			else if constexpr (std::is_same_v<From, uint64_t> && std::is_same_v<To, float>) return TV(_mm512_cvtepu64_ps(value));
			else if constexpr (std::is_same_v<From, int64_t> && std::is_same_v<To, double>) return TV(_mm512_cvtepi64_pd(value));
			else if constexpr (std::is_same_v<From, int64_t> && std::is_same_v<To, float>) return TV(_mm512_cvtepi64_ps(value));
			else if constexpr (std::is_same_v<From, uint32_t> && std::is_same_v<To, double>) return TV(_mm512_cvtepu32_pd(value));
			else if constexpr (std::is_same_v<From, uint32_t> && std::is_same_v<To, float>) return TV(_mm512_cvtepu32_ps(value));
			else if constexpr (std::is_same_v<From, int32_t> && std::is_same_v<To, double>) return TV(_mm512_cvtepi32_pd(value));
			else if constexpr (std::is_same_v<From, int32_t> && std::is_same_v<To, float>) return TV(_mm512_cvtepi32_ps(value));

			//integer conversion zone
			else if constexpr (std::is_integral_v<From> && std::is_integral_v<To>)
			{
				//conversion to smaller ints doesn't care about sign
				if constexpr (sizeof(From) == 8 && sizeof(To) == 4) return TV(_mm512_cvtepi64_epi32(value));
				else if constexpr (sizeof(From) == 8 && sizeof(To) == 2) return TV(_mm512_cvtepi64_epi16(value));
				else if constexpr (sizeof(From) == 8 && sizeof(To) == 1) return TV(_mm512_cvtepi64_epi8(value));
				else if constexpr (sizeof(From) == 4 && sizeof(To) == 2) return TV(_mm512_cvtepi32_epi16(value));
				else if constexpr (sizeof(From) == 4 && sizeof(To) == 1) return TV(_mm512_cvtepi32_epi8(value));
				else if constexpr (sizeof(From) == 2 && sizeof(To) == 1) return TV(_mm512_cvtepi16_epi8(value));

				//conversion to bigger DOES care about sign
				else if constexpr (std::is_same_v<From, uint8_t> && sizeof(To) == 8) return TV(_mm512_cvtepu8_epi64(value));
				else if constexpr (std::is_same_v<From, uint8_t> && sizeof(To) == 4) return TV(_mm512_cvtepu8_epi32(value));
				else if constexpr (std::is_same_v<From, uint8_t> && sizeof(To) == 2) return TV(_mm512_cvtepu8_epi16(value));
				else if constexpr (std::is_same_v<From, int8_t> && sizeof(To) == 8) return TV(_mm512_cvtepi8_epi64(value));
				else if constexpr (std::is_same_v<From, int8_t> && sizeof(To) == 4) return TV(_mm512_cvtepi8_epi32(value));
				else if constexpr (std::is_same_v<From, int8_t> && sizeof(To) == 2) return TV(_mm512_cvtepi8_epi16(value));

				else if constexpr (std::is_same_v<From, uint16_t> && sizeof(To) == 8) return TV(_mm512_cvtepu16_epi64(value));
				else if constexpr (std::is_same_v<From, uint16_t> && sizeof(To) == 4) return TV(_mm512_cvtepu16_epi32(value));
				else if constexpr (std::is_same_v<From, int16_t> && sizeof(To) == 8) return TV(_mm512_cvtepi16_epi64(value));
				else if constexpr (std::is_same_v<From, int16_t> && sizeof(To) == 4) return TV(_mm512_cvtepi16_epi32(value));

				else if constexpr (std::is_same_v<From, uint32_t> && sizeof(To) == 8) return TV(_mm512_cvtepu32_epi64(value));
				else if constexpr (std::is_same_v<From, int32_t> && sizeof(To) == 8) return TV(_mm512_cvtepi32_epi64(value));
			}
			else static_assert(false, "Unsupported arguments for SIMD_Vector zmm_cvt");
	}


	//This function is used internally and lacks upstream input sanitization. If you're looking for vector conversions, use cvt instead
	template<typename To, size_t N, typename From>
	__forceinline SIMD_Vector<To, N> ymm_cvt(const SIMD_Vector<From, N>& value)
	requires (inRange(std::max(sizeof(SIMD_Vector<From, N>), sizeof(SIMD_Vector<To, N>)), 17, 32))
	{
		using TV = SIMD_Vector<To, N>;
		//Truncation is REQUIRED for FP -> int
		if constexpr (std::is_same_v<From, double> && std::is_same_v<To, uint64_t>)  return TV(_mm256_cvttpd_epu64(value));
		else if constexpr (std::is_same_v<From, double> && std::is_same_v<To, int64_t>)  return TV(_mm256_cvttpd_epi64(value));
		else if constexpr (std::is_same_v<From, double> && std::is_same_v<To, float>)  return TV(_mm256_cvtpd_ps(value));
		else if constexpr (std::is_same_v<From, double> && std::is_same_v<To, uint32_t>)  return TV(_mm256_cvttpd_epu32(value));
		else if constexpr (std::is_same_v<From, double> && std::is_same_v<To, int32_t>)  return TV(_mm256_cvttpd_epi32(value));

		else if constexpr (std::is_same_v<From, float> && std::is_same_v<To, double>) return TV(_mm256_cvtps_pd(value));
		else if constexpr (std::is_same_v<From, float> && std::is_same_v<To, uint64_t>) return TV(_mm256_cvttps_epu64(value));
		else if constexpr (std::is_same_v<From, float> && std::is_same_v<To, int64_t>) return TV(_mm256_cvttps_epi64(value));
		else if constexpr (std::is_same_v<From, float> && std::is_same_v<To, uint32_t>) return TV(_mm256_cvttps_epu32(value));
		else if constexpr (std::is_same_v<From, float> && std::is_same_v<To, int32_t>) return TV(_mm256_cvttps_epi32(value));
		//else if constexpr (std::is_same_v<From, float16_t> && std::is_same_v<To, float>) return TV(_mm256_cvtph_ps(value));

		//only float conversions care about the signedness
		else if constexpr (std::is_same_v<From, uint64_t> && std::is_same_v<To, double>) return TV(_mm256_cvtepu64_pd(value));
		else if constexpr (std::is_same_v<From, uint64_t> && std::is_same_v<To, float>) return TV(_mm256_cvtepu64_ps(value));
		else if constexpr (std::is_same_v<From, int64_t> && std::is_same_v<To, double>) return TV(_mm256_cvtepi64_pd(value));
		else if constexpr (std::is_same_v<From, int64_t> && std::is_same_v<To, float>) return TV(_mm256_cvtepi64_ps(value));
		else if constexpr (std::is_same_v<From, uint32_t> && std::is_same_v<To, double>) return TV(_mm256_cvtepu32_pd(value));
		else if constexpr (std::is_same_v<From, uint32_t> && std::is_same_v<To, float>) return TV(_mm256_cvtepu32_ps(value));
		else if constexpr (std::is_same_v<From, int32_t> && std::is_same_v<To, double>) return TV(_mm256_cvtepi32_pd(value));
		else if constexpr (std::is_same_v<From, int32_t> && std::is_same_v<To, float>) return TV(_mm256_cvtepi32_ps(value));

		//integer conversion zone
		else if constexpr (std::is_integral_v<From> && std::is_integral_v<To>)
		{
			//conversion to smaller ints doesn't care about sign
			if constexpr (sizeof(From) == 8 && sizeof(To) == 4) return TV(_mm256_cvtepi64_epi32(value));
			else if constexpr (sizeof(From) == 8 && sizeof(To) == 2) return TV(_mm256_cvtepi64_epi16(value));
			else if constexpr (sizeof(From) == 8 && sizeof(To) == 1) return TV(_mm256_cvtepi64_epi8(value));
			else if constexpr (sizeof(From) == 4 && sizeof(To) == 2) return TV(_mm256_cvtepi32_epi16(value));
			else if constexpr (sizeof(From) == 4 && sizeof(To) == 1) return TV(_mm256_cvtepi32_epi8(value));
			else if constexpr (sizeof(From) == 2 && sizeof(To) == 1) return TV(_mm256_cvtepi16_epi8(value));

			//conversion to bigger DOES care about sign
			else if constexpr (std::is_same_v<From, uint8_t> && sizeof(To) == 8) return TV(_mm256_cvtepu8_epi64(value));
			else if constexpr (std::is_same_v<From, uint8_t> && sizeof(To) == 4) return TV(_mm256_cvtepu8_epi32(value));
			else if constexpr (std::is_same_v<From, uint8_t> && sizeof(To) == 2) return TV(_mm256_cvtepu8_epi16(value));
			else if constexpr (std::is_same_v<From, int8_t> && sizeof(To) == 8) return TV(_mm256_cvtepi8_epi64(value));
			else if constexpr (std::is_same_v<From, int8_t> && sizeof(To) == 4) return TV(_mm256_cvtepi8_epi32(value));
			else if constexpr (std::is_same_v<From, int8_t> && sizeof(To) == 2) return TV(_mm256_cvtepi8_epi16(value));

			else if constexpr (std::is_same_v<From, uint16_t> && sizeof(To) == 8) return TV(_mm256_cvtepu16_epi64(value));
			else if constexpr (std::is_same_v<From, uint16_t> && sizeof(To) == 4) return TV(_mm256_cvtepu16_epi32(value));
			else if constexpr (std::is_same_v<From, int16_t> && sizeof(To) == 8) return TV(_mm256_cvtepi16_epi64(value));
			else if constexpr (std::is_same_v<From, int16_t> && sizeof(To) == 4) return TV(_mm256_cvtepi16_epi32(value));

			else if constexpr (std::is_same_v<From, uint32_t> && sizeof(To) == 8) return TV(_mm256_cvtepu32_epi64(value));
			else if constexpr (std::is_same_v<From, int32_t> && sizeof(To) == 8) return TV(_mm256_cvtepi32_epi64(value));
		}
		else static_assert(false, "Unsupported arguments for SIMD_Vector zmm_cvt");
	}
	//This function is used internally and lacks upstream input sanitization. If you're looking for vector conversions, use cvt instead
	template<typename To, size_t N, typename From>
	__forceinline SIMD_Vector<To, N> xmm_cvt(const SIMD_Vector<From, N>& value)
	requires (std::max(sizeof(SIMD_Vector<From, N>), sizeof(SIMD_Vector<To, N>)) <= 16)
	{
		using TV = SIMD_Vector<To, N>;
		//Truncation is REQUIRED for FP -> int
		if constexpr (std::is_same_v<From, double> && std::is_same_v<To, uint64_t>)  return TV(_mm_cvttpd_epu64(value));
		else if constexpr (std::is_same_v<From, double> && std::is_same_v<To, int64_t>)  return TV(_mm_cvttpd_epi64(value));
		else if constexpr (std::is_same_v<From, double> && std::is_same_v<To, float>)  return TV(_mm_cvtpd_ps(value));
		else if constexpr (std::is_same_v<From, double> && std::is_same_v<To, uint32_t>)  return TV(_mm_cvttpd_epu32(value));
		else if constexpr (std::is_same_v<From, double> && std::is_same_v<To, int32_t>)  return TV(_mm_cvttpd_epi32(value));

		else if constexpr (std::is_same_v<From, float> && std::is_same_v<To, double>) return TV(_mm_cvtps_pd(value));
		else if constexpr (std::is_same_v<From, float> && std::is_same_v<To, uint64_t>) return TV(_mm_cvttps_epu64(value));
		else if constexpr (std::is_same_v<From, float> && std::is_same_v<To, int64_t>) return TV(_mm_cvttps_epi64(value));
		else if constexpr (std::is_same_v<From, float> && std::is_same_v<To, uint32_t>) return TV(_mm_cvttps_epu32(value));
		else if constexpr (std::is_same_v<From, float> && std::is_same_v<To, int32_t>) return TV(_mm_cvttps_epi32(value));
		//else if constexpr (std::is_same_v<From, float16_t> && std::is_same_v<To, float>) return TV(_mm_cvtph_ps(value));

		//only float conversions care about the signedness
		else if constexpr (std::is_same_v<From, uint64_t> && std::is_same_v<To, double>) return TV(_mm_cvtepu64_pd(value));
		else if constexpr (std::is_same_v<From, uint64_t> && std::is_same_v<To, float>) return TV(_mm_cvtepu64_ps(value));
		else if constexpr (std::is_same_v<From, int64_t> && std::is_same_v<To, double>) return TV(_mm_cvtepi64_pd(value));
		else if constexpr (std::is_same_v<From, int64_t> && std::is_same_v<To, float>) return TV(_mm_cvtepi64_ps(value));
		else if constexpr (std::is_same_v<From, uint32_t> && std::is_same_v<To, double>) return TV(_mm_cvtepu32_pd(value));
		else if constexpr (std::is_same_v<From, uint32_t> && std::is_same_v<To, float>) return TV(_mm_cvtepu32_ps(value));
		else if constexpr (std::is_same_v<From, int32_t> && std::is_same_v<To, double>) return TV(_mm_cvtepi32_pd(value));
		else if constexpr (std::is_same_v<From, int32_t> && std::is_same_v<To, float>) return TV(_mm_cvtepi32_ps(value));

		//integer conversion zone
		else if constexpr (std::is_integral_v<From> && std::is_integral_v<To>)
		{
			//conversion to smaller ints doesn't care about sign
			if constexpr (sizeof(From) == 8 && sizeof(To) == 4) return TV(_mm_cvtepi64_epi32(value));
			else if constexpr (sizeof(From) == 8 && sizeof(To) == 2) return TV(_mm_cvtepi64_epi16(value));
			else if constexpr (sizeof(From) == 8 && sizeof(To) == 1) return TV(_mm_cvtepi64_epi8(value));
			else if constexpr (sizeof(From) == 4 && sizeof(To) == 2) return TV(_mm_cvtepi32_epi16(value));
			else if constexpr (sizeof(From) == 4 && sizeof(To) == 1) return TV(_mm_cvtepi32_epi8(value));
			else if constexpr (sizeof(From) == 2 && sizeof(To) == 1) return TV(_mm_cvtepi16_epi8(value));

			//conversion to bigger DOES care about sign
			else if constexpr (std::is_same_v<From, uint8_t> && sizeof(To) == 8) return TV(_mm_cvtepu8_epi64(value));
			else if constexpr (std::is_same_v<From, uint8_t> && sizeof(To) == 4) return TV(_mm_cvtepu8_epi32(value));
			else if constexpr (std::is_same_v<From, uint8_t> && sizeof(To) == 2) return TV(_mm_cvtepu8_epi16(value));
			else if constexpr (std::is_same_v<From, int8_t> && sizeof(To) == 8) return TV(_mm_cvtepi8_epi64(value));
			else if constexpr (std::is_same_v<From, int8_t> && sizeof(To) == 4) return TV(_mm_cvtepi8_epi32(value));
			else if constexpr (std::is_same_v<From, int8_t> && sizeof(To) == 2) return TV(_mm_cvtepi8_epi16(value));

			else if constexpr (std::is_same_v<From, uint16_t> && sizeof(To) == 8) return TV(_mm_cvtepu16_epi64(value));
			else if constexpr (std::is_same_v<From, uint16_t> && sizeof(To) == 4) return TV(_mm_cvtepu16_epi32(value));
			else if constexpr (std::is_same_v<From, int16_t> && sizeof(To) == 8) return TV(_mm_cvtepi16_epi64(value));
			else if constexpr (std::is_same_v<From, int16_t> && sizeof(To) == 4) return TV(_mm_cvtepi16_epi32(value));

			else if constexpr (std::is_same_v<From, uint32_t> && sizeof(To) == 8) return TV(_mm_cvtepu32_epi64(value));
			else if constexpr (std::is_same_v<From, int32_t> && sizeof(To) == 8) return TV(_mm_cvtepi32_epi64(value));
		}
		else static_assert(false, "Unsupported arguments for SIMD_Vector zmm_cvt");
	}
	}



	template<typename To, size_t N, typename From>
	__forceinline SIMD_Vector<To, N> cvt(const SIMD_Vector<From, N>& value)
	{
		using FromVector = SIMD_Vector<From, N>; using ToVector = SIMD_Vector<To, N>;
		constexpr size_t MaxSize = std::max(sizeof(FromVector), sizeof(ToVector));
		if constexpr (std::is_same_v<To, From>) return value; //if something tries to convert type to itself, it's just a NOP, so value can be passed back immediately
		else if constexpr (sizeof(To) == sizeof(From) && std::is_integral_v<To> && std::is_integral_v<From>) return reinterpret<To>(value); //same sized integers, just reinterpret
		else if constexpr (MaxSize > 64) return concat(cvt<To>(extract<0, 2>(value)), cvt<To>(extract<1, 2>(value))); //break up too large vectors to halves

		//if constexpr (N == 1) return insert<0,1,To,1>({},  )
		//small integers have no direct path to floating point conversions, so route them through 32-bit integers of samed signedness
		//from small integer to double or float
		else if constexpr (std::is_integral_v<From> && sizeof(From) < 4 && (std::is_same_v<To, double> || std::is_same_v<To, float>))
		{
			if constexpr (std::is_signed_v<From>) return cvt<To>(cvt<int32_t>(value));
			if constexpr (std::is_unsigned_v<From>) return cvt<To>(cvt<uint32_t>(value));
		}
		//same for inverse conversions. From double or float to small integer
		else if constexpr ((std::is_same_v<From, double> || std::is_same_v<From, float>) && std::is_integral_v<To> && sizeof(To) < 4)
		{
			if constexpr (std::is_signed_v<To>) return cvt<To>(cvt<int32_t>(value));
			if constexpr (std::is_unsigned_v<To>) return cvt<To>(cvt<uint32_t>(value));
		}

		//Now that special cases are out of the way, actual intrinsics can be used.
		else if constexpr (inRange(MaxSize, 33, 64)) return internals::zmm_cvt<To>(value);
		else if constexpr (inRange(MaxSize, 17, 32)) return internals::ymm_cvt<To>(value);
		else if constexpr (MaxSize <= 16) return internals::xmm_cvt<To>(value);
		else static_assert(false, "Unsupported arguments for SIMD_Vector cvt");
	}
}