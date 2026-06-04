#pragma once
#include "funcs.h"
#include <string>
#include "cvt.h"
#include <bit>
#include <iostream>
#include "capabilities.h"
namespace AVXXY_NAMESPACE
{
	//DO NOT forget elses, compilation is funky with constexpr trees, it's not like usual control flow that can easily return from the middle of a function

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> add(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using T = SIMD_Vector<S, N>;
		constexpr auto c = capabilities::current;
		constexpr bool hasVectorVersion = (c.SSE && (std::is_same_v<S, float>)) || c.SSE2;
		//can't unite this. Every case must be a separate branch, since template resolution doesn't early-return (it does at runtime, but not at compile-time). Thus, have fully expanded decisions
		//to ensure nothing every falls out without returning
		if constexpr (inRange(sizeof(T), 0, 16) && c.SSE && std::is_same_v<S, float>) return _mm_add_ps(a, b);
		else if constexpr (inRange(sizeof(T), 0, 16) && c.SSE2 && std::is_same_v<S, double>) return _mm_add_pd(a, b);
		else if constexpr (inRange(sizeof(T), 0, 16) && c.SSE2 && (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>)) return _mm_add_epi64(a, b);
		else if constexpr (inRange(sizeof(T), 0, 16) && c.SSE2 && (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>)) return _mm_add_epi32(a, b);
		else if constexpr (inRange(sizeof(T), 0, 16) && c.SSE2 && (std::is_same_v<S, int16_t> || std::is_same_v<S, uint16_t>)) return _mm_add_epi16(a, b);
		else if constexpr (inRange(sizeof(T), 0, 16) && c.SSE2 && (std::is_same_v<S, int8_t> || std::is_same_v<S, uint8_t>)) return _mm_add_epi8(a, b);
		else if constexpr (inRange(sizeof(T), 17, 32) && c.AVX && std::is_same_v<S, float>) return _mm256_add_ps(a, b);
		else if constexpr (inRange(sizeof(T), 17, 32) && c.AVX && std::is_same_v<S, double>) return _mm256_add_pd(a, b);
		//TODO: can replace these with FP emulation for old AVX, but SSE is probably still faster
		else if constexpr (inRange(sizeof(T), 17, 32) && c.AVX2 && (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>)) return _mm256_add_epi64(a, b);
		else if constexpr (inRange(sizeof(T), 17, 32) && c.AVX2 && (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>)) return _mm256_add_epi32(a, b);
		else if constexpr (inRange(sizeof(T), 17, 32) && c.AVX2 && (std::is_same_v<S, int16_t> || std::is_same_v<S, uint16_t>)) return _mm256_add_epi16(a, b);
		else if constexpr (inRange(sizeof(T), 17, 32) && c.AVX2 && (std::is_same_v<S, int8_t> || std::is_same_v<S, uint8_t>)) return _mm256_add_epi8(a, b);
		else if constexpr (inRange(sizeof(T), 33, 64) && c.AVX512.F && (std::is_same_v<S, double>)) return _mm512_add_pd(a, b);
		else if constexpr (inRange(sizeof(T), 33, 64) && c.AVX512.F && (std::is_same_v<S, float>)) return _mm512_add_ps(a, b);
		else if constexpr (inRange(sizeof(T), 33, 64) && c.AVX512.F && (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>)) return _mm512_add_epi64(a, b);
		else if constexpr (inRange(sizeof(T), 33, 64) && c.AVX512.F && (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>)) return _mm512_add_epi32(a, b);
		else if constexpr (inRange(sizeof(T), 33, 64) && c.AVX512.BW && (std::is_same_v<S, int16_t> || std::is_same_v<S, uint16_t>)) return _mm512_add_epi16(a, b);
		else if constexpr (inRange(sizeof(T), 33, 64) && c.AVX512.BW && (std::is_same_v<S, int8_t> || std::is_same_v<S, uint8_t>)) return _mm512_add_epi8(a, b);
		else if constexpr (hasVectorVersion) return concat(add(a.lo, b.lo), add(a.hi, b.hi));
		else
		{
			T ret;
			for (size_t i = 0; i < N; ++i) ret[i] = a[i] + b[i];
			return ret;
		}
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> sub(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using T = SIMD_Vector<S, N>;
		constexpr auto c = capabilities::current;
		constexpr bool hasVectorVersion = (c.SSE && (std::is_same_v<S, float>)) || c.SSE2;
		if constexpr (inRange(sizeof(T), 0, 16) && c.SSE && std::is_same_v<S, float>) return _mm_sub_ps(a, b);
		else if constexpr (inRange(sizeof(T), 0, 16) && c.SSE2 && std::is_same_v<S, double>) return _mm_sub_pd(a, b);
		else if constexpr (inRange(sizeof(T), 0, 16) && c.SSE2 && (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>)) return _mm_sub_epi64(a, b);
		else if constexpr (inRange(sizeof(T), 0, 16) && c.SSE2 && (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>)) return _mm_sub_epi32(a, b);
		else if constexpr (inRange(sizeof(T), 0, 16) && c.SSE2 && (std::is_same_v<S, int16_t> || std::is_same_v<S, uint16_t>)) return _mm_sub_epi16(a, b);
		else if constexpr (inRange(sizeof(T), 0, 16) && c.SSE2 && (std::is_same_v<S, int8_t> || std::is_same_v<S, uint8_t>)) return _mm_sub_epi8(a, b);
		else if constexpr (inRange(sizeof(T), 17, 32) && c.AVX && std::is_same_v<S, float>) return _mm256_sub_ps(a, b);
		else if constexpr (inRange(sizeof(T), 17, 32) && c.AVX && std::is_same_v<S, double>) return _mm256_sub_pd(a, b);
		else if constexpr (inRange(sizeof(T), 17, 32) && c.AVX2 && (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>)) return _mm256_sub_epi64(a, b);
		else if constexpr (inRange(sizeof(T), 17, 32) && c.AVX2 && (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>)) return _mm256_sub_epi32(a, b);
		else if constexpr (inRange(sizeof(T), 17, 32) && c.AVX2 && (std::is_same_v<S, int16_t> || std::is_same_v<S, uint16_t>)) return _mm256_sub_epi16(a, b);
		else if constexpr (inRange(sizeof(T), 17, 32) && c.AVX2 && (std::is_same_v<S, int8_t> || std::is_same_v<S, uint8_t>)) return _mm256_sub_epi8(a, b);
		else if constexpr (inRange(sizeof(T), 33, 64) && c.AVX512.F && (std::is_same_v<S, double>)) return _mm512_sub_pd(a, b);
		else if constexpr (inRange(sizeof(T), 33, 64) && c.AVX512.F && (std::is_same_v<S, float>)) return _mm512_sub_ps(a, b);
		else if constexpr (inRange(sizeof(T), 33, 64) && c.AVX512.F && (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>)) return _mm512_sub_epi64(a, b);
		else if constexpr (inRange(sizeof(T), 33, 64) && c.AVX512.F && (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>)) return _mm512_sub_epi32(a, b);
		else if constexpr (inRange(sizeof(T), 33, 64) && c.AVX512.BW && (std::is_same_v<S, int16_t> || std::is_same_v<S, uint16_t>)) return _mm512_sub_epi16(a, b);
		else if constexpr (inRange(sizeof(T), 33, 64) && c.AVX512.BW && (std::is_same_v<S, int8_t> || std::is_same_v<S, uint8_t>)) return _mm512_sub_epi8(a, b);
		else if constexpr (hasVectorVersion) return concat(sub(a.lo, b.lo), sub(a.hi, b.hi));
		else
		{
			T ret;
			for (size_t i = 0; i < N; ++i) ret[i] = a[i] - b[i];
			return ret;
		}
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> mul(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using T = SIMD_Vector<S, N>;
		if constexpr (std::is_same_v<S, int8_t>) return vec_cvt<int8_t>(mul(vec_cvt<int16_t>(a), vec_cvt<int16_t>(b)));
		else if constexpr (std::is_same_v<S, uint8_t>) return vec_cvt<uint8_t>(mul(vec_cvt<uint16_t>(a), vec_cvt<uint16_t>(b)));
		else if constexpr (sizeof(T) > 64) return concat(mul(a.lo, b.lo), mul(a.hi, b.hi));
		else if (capabilities::current.AVX512.BW)
		{
			if constexpr (inRange(sizeof(T), 33, 64))
			{
				if constexpr (std::is_same_v<S, float>) return _mm512_mul_ps(a, b);
				if constexpr (std::is_same_v<S, double>) return _mm512_mul_pd(a, b);
				if constexpr (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>) return _mm512_mullo_epi64(a, b);
				if constexpr (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>) return _mm512_mullo_epi32(a, b);
				if constexpr (std::is_same_v<S, int16_t> || std::is_same_v<S, uint16_t>) return _mm512_mullo_epi16(a, b);
			}
			else if constexpr (inRange(sizeof(T), 17, 32))
			{
				if constexpr (std::is_same_v<S, float>) return _mm256_mul_ps(a, b);
				if constexpr (std::is_same_v<S, double>) return _mm256_mul_pd(a, b);
				if constexpr (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>) return _mm256_mullo_epi64(a, b);
				if constexpr (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>) return _mm256_mullo_epi32(a, b);
				if constexpr (std::is_same_v<S, int16_t> || std::is_same_v<S, uint16_t>) return _mm256_mullo_epi16(a, b);
			}
			else if constexpr (inRange(sizeof(T), 0, 16))
			{
				if constexpr (std::is_same_v<S, float>) return _mm_mul_ps(a, b);
				if constexpr (std::is_same_v<S, double>) return _mm_mul_pd(a, b);
				if constexpr (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>) return _mm_mullo_epi64(a, b);
				if constexpr (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>) return _mm_mullo_epi32(a, b);
				if constexpr (std::is_same_v<S, int16_t> || std::is_same_v<S, uint16_t>) return _mm_mullo_epi16(a, b);
			}
		}
		else
		{
			T ret;
			for (size_t i = 0; i < N; ++i) ret[i] = a[i] * b[i];
			return ret;
		}
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N * 2> concat(const SIMD_Vector<S, N>& to, const SIMD_Vector<S, N>& what)
	{
		SIMD_Vector<S, N * 2> ret;
		ret.lo = to;
		ret.hi = what;
		return ret;
	}

	template<size_t N>
		requires (N * 2 <= 64)
	__forceinline SIMD_Mask<N * 2> concat_masks(const SIMD_Mask<N>& to, const SIMD_Mask<N>& what)
	{
		using T = SIMD_Mask<N * 2>;
		using U = typename T::UintType;
		U ret = to;
		ret |= U(what) << N;
		return ret;
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> div(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using T = SIMD_Vector<S, N>;
		constexpr size_t sz = sizeof(T);
		if constexpr (std::is_integral_v<S> && sizeof(S) <= 2) return vec_cvt<S>(div(vec_cvt<float>(a), vec_cvt<float>(b))); //emulate division for 8 and 16 bit ints via float division
		else if constexpr (std::is_integral_v<S> && sizeof(S) == 4) return vec_cvt<S>(div(vec_cvt<double>(a), vec_cvt<double>(b))); //for 32 bit ints via double division
		//64 bit integers are too large, can't easily weasel our way out. TODO: implement 64-bit int div
		else if constexpr (sz > 64) return concat(div(lower_half(a), lower_half(b)), div(upper_half(a), upper_half(b)));
		else if constexpr (capabilities::current == capabilities::zen4)
		{
			if constexpr (inRange(sz, 33, 64))
			{
				if constexpr (std::is_same_v<S, float>) return T(_mm512_div_ps(a, b));
				if constexpr (std::is_same_v<S, double>) return T(_mm512_div_pd(a, b));
			}
			else if constexpr (inRange(sz, 17, 32))
			{
				if constexpr (std::is_same_v<S, float>) return T(_mm256_div_ps(a, b));
				if constexpr (std::is_same_v<S, double>) return T(_mm256_div_pd(a, b));
			}
			else if constexpr (inRange(sz, 0, 16))
			{
				if constexpr (std::is_same_v<S, float>) return T(_mm_div_ps(a, b));
				if constexpr (std::is_same_v<S, double>) return T(_mm_div_pd(a, b));
			}
		}
		else
		{
			SIMD_Vector<S, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = a[i] / b[i];
			return ret;
		}
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> logic_and(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using T = SIMD_Vector<S, N>;
		if constexpr (capabilities::current == capabilities::zen4)
		{
			if constexpr (sizeof(T) > 64) return concat(logic_and(a.lo, b.lo), logic_and(a.hi, b.hi));
			else if constexpr (inRange(sizeof(T), 33, 64))
			{
				if constexpr (std::is_integral_v<S>) return _mm512_and_si512(a, b);
				if constexpr (std::is_same_v<S, float>) return _mm512_and_ps(a, b);
				if constexpr (std::is_same_v<S, double>) return _mm512_and_pd(a, b);
			}
			else if constexpr (inRange(sizeof(T), 17, 32))
			{
				if constexpr (std::is_integral_v<S>) return _mm256_and_si256(a, b);
				if constexpr (std::is_same_v<S, float>) return _mm256_and_ps(a, b);
				if constexpr (std::is_same_v<S, double>) return _mm256_and_pd(a, b);
			}
			else if constexpr (inRange(sizeof(T), 0, 16))
			{
				if constexpr (std::is_integral_v<S>) return _mm_and_si128(a, b);
				if constexpr (std::is_same_v<S, float>) return _mm_and_ps(a, b);
				if constexpr (std::is_same_v<S, double>) return _mm_and_pd(a, b);
			}
		}
		else
		{
			T ret;
			for (size_t i = 0; i < N; ++i)
			{
				if constexpr (std::is_same_v<S, double>) ret[i] = std::bit_cast<double>(std::bit_cast<uint64_t>(a[i]) & std::bit_cast<uint64_t>(b[i]));
				else if constexpr (std::is_same_v<S, float>) ret[i] = std::bit_cast<float>(std::bit_cast<uint32_t>(a[i]) & std::bit_cast<uint32_t>(b[i]));
				else ret[i] = a[i] & b[i];
			}
			return ret;
		}
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> logic_or(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using T = SIMD_Vector<S, N>;
		if constexpr (capabilities::current == capabilities::zen4)
		{
			if constexpr (sizeof(T) > 64) return concat(logic_or(a.lo, b.lo), logic_or(a.hi, b.hi));
			else if constexpr (inRange(sizeof(T), 33, 64))
			{
				if constexpr (std::is_integral_v<S>) return _mm512_or_si512(a, b);
				if constexpr (std::is_same_v<S, float>) return _mm512_or_ps(a, b);
				if constexpr (std::is_same_v<S, double>) return _mm512_or_pd(a, b);
			}
			else if constexpr (inRange(sizeof(T), 17, 32))
			{
				if constexpr (std::is_integral_v<S>) return _mm256_or_si256(a, b);
				if constexpr (std::is_same_v<S, float>) return _mm256_or_ps(a, b);
				if constexpr (std::is_same_v<S, double>) return _mm256_or_pd(a, b);
			}
			else if constexpr (inRange(sizeof(T), 0, 16))
			{
				if constexpr (std::is_integral_v<S>) return _mm_or_si128(a, b);
				if constexpr (std::is_same_v<S, float>) return _mm_or_ps(a, b);
				if constexpr (std::is_same_v<S, double>) return _mm_or_pd(a, b);
			}
		}
		else
		{
			T ret;
			for (size_t i = 0; i < N; ++i)
			{
				if constexpr (std::is_same_v<S, double>) ret[i] = std::bit_cast<double>(std::bit_cast<uint64_t>(a[i]) | std::bit_cast<uint64_t>(b[i]));
				else if constexpr (std::is_same_v<S, float>) ret[i] = std::bit_cast<float>(std::bit_cast<uint32_t>(a[i]) | std::bit_cast<uint32_t>(b[i]));
				else ret[i] = a[i] | b[i];
			}
			return ret;
		}
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> logic_xor(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using T = SIMD_Vector<S, N>;
		if constexpr (capabilities::current == capabilities::zen4)
		{
			if constexpr (sizeof(T) > 64) return concat(logic_xor(a.lo, b.lo), logic_xor(a.hi, b.hi));
			else if constexpr (inRange(sizeof(T), 33, 64))
			{
				if constexpr (std::is_integral_v<S>) return _mm512_xor_si512(a, b);
				if constexpr (std::is_same_v<S, float>) return _mm512_xor_ps(a, b);
				if constexpr (std::is_same_v<S, double>) return _mm512_xor_pd(a, b);
			}
			else if constexpr (inRange(sizeof(T), 17, 32))
			{
				if constexpr (std::is_integral_v<S>) return _mm256_xor_si256(a, b);
				if constexpr (std::is_same_v<S, float>) return _mm256_xor_ps(a, b);
				if constexpr (std::is_same_v<S, double>) return _mm256_xor_pd(a, b);
			}
			else if constexpr (inRange(sizeof(T), 0, 16))
			{
				if constexpr (std::is_integral_v<S>) return _mm_xor_si128(a, b);
				if constexpr (std::is_same_v<S, float>) return _mm_xor_ps(a, b);
				if constexpr (std::is_same_v<S, double>) return _mm_xor_pd(a, b);
			}
		}
		else
		{
			T ret;
			for (size_t i = 0; i < N; ++i)
			{
				if constexpr (std::is_same_v<S, double>) ret[i] = std::bit_cast<double>(std::bit_cast<uint64_t>(a[i]) ^ std::bit_cast<uint64_t>(b[i]));
				else if constexpr (std::is_same_v<S, float>) ret[i] = std::bit_cast<float>(std::bit_cast<uint32_t>(a[i]) ^ std::bit_cast<uint32_t>(b[i]));
				else ret[i] = a[i] ^ b[i];
			}
			return ret;
		}
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> logic_not(const SIMD_Vector<S, N>& a)
	{
		using T = SIMD_Vector<S, N>;
		if constexpr (capabilities::current == capabilities::zen4)
		{
			if constexpr (sizeof(T) > 64) return concat(logic_not(a.lo), logic_not(a.hi));
			else if constexpr (inRange(sizeof(T), 33, 64))
			{
				if constexpr (std::is_integral_v<S>) return _mm512_xor_si512(a, _mm512_set1_epi32(0xFFFFFFFF));
				if constexpr (std::is_same_v<S, float>) return _mm512_xor_ps(a, _mm512_set1_ps(std::bit_cast<float>(0xFFFFFFFF)));
				if constexpr (std::is_same_v<S, double>) return _mm512_xor_pd(a, _mm512_set1_pd(std::bit_cast<double>(0xFFFFFFFFFFFFFFFF)));
			}
			else if constexpr (inRange(sizeof(T), 17, 32))
			{
				if constexpr (std::is_integral_v<S>) return _mm256_xor_si256(a, _mm256_set1_epi32(0xFFFFFFFF));
				if constexpr (std::is_same_v<S, float>) return _mm256_xor_ps(a, _mm256_set1_ps(std::bit_cast<float>(0xFFFFFFFF)));
				if constexpr (std::is_same_v<S, double>) return _mm256_xor_pd(a, _mm256_set1_pd(std::bit_cast<double>(0xFFFFFFFFFFFFFFFF)));
			}
			else if constexpr (inRange(sizeof(T), 0, 16))
			{
				if constexpr (std::is_integral_v<S>) return _mm_xor_si128(a, _mm_set1_epi32(0xFFFFFFFF));
				if constexpr (std::is_same_v<S, float>) return _mm_xor_ps(a, _mm_set1_ps(std::bit_cast<float>(0xFFFFFFFF)));
				if constexpr (std::is_same_v<S, double>) return _mm_xor_pd(a, _mm_set1_pd(std::bit_cast<double>(0xFFFFFFFFFFFFFFFF)));
			}
		}
		else
		{
			SIMD_Vector<S, N> ret;
			for (size_t i = 0; i < N; ++i)
			{
				if constexpr (std::is_same_v<S, double>) ret[i] = std::bit_cast<double>(~std::bit_cast<uint64_t>(a[i]));
				else if constexpr (std::is_same_v<S, float>) ret[i] = std::bit_cast<float>(~std::bit_cast<uint32_t>(a[i]));
				else ret[i] = ~a[i];
			}
			return ret;
		}
	}

	template<typename S, size_t N, typename I>
		requires (std::is_integral_v<I>&& std::is_integral_v<S>)
	__forceinline SIMD_Vector<S, N> shift_left(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& amount)
	{
		using T = SIMD_Vector<S, N>;
		if constexpr (capabilities::current == capabilities::zen4)
		{
			if constexpr (sizeof(T) > 64) return concat(shift_left(a.lo, amount.lo), shift_left(a.hi, amount.hi));
			if constexpr (sizeof(S) == 1) //no 8 bit shifts in x86!
			{
				auto alo = vec_cvt<uint16_t>(vec_cvt<uint8_t>(a.lo)); //treat as unsigned bytes to avoid sign spillover
				auto ahi = vec_cvt<uint16_t>(vec_cvt<uint8_t>(a.hi));
				auto shlo = vec_cvt<S>(shift_left(alo, amount.lo));
				auto shhi = vec_cvt<S>(shift_left(ahi, amount.hi));
				return concat(shlo, shhi);
			}

			auto xa = vec_cvt<typename T::IntScalarType>(amount);
			if constexpr (inRange(sizeof(T), 33, 64))
			{
				if constexpr (sizeof(S) == 8) return _mm512_sllv_epi64(a, xa);
				if constexpr (sizeof(S) == 4) return _mm512_sllv_epi32(a, xa);
				if constexpr (sizeof(S) == 2) return _mm512_sllv_epi16(a, xa);
			}
			else if constexpr (inRange(sizeof(T), 17, 32))
			{
				if constexpr (sizeof(S) == 8) return _mm256_sllv_epi64(a, xa);
				if constexpr (sizeof(S) == 4) return _mm256_sllv_epi32(a, xa);
				if constexpr (sizeof(S) == 2) return _mm256_sllv_epi16(a, xa);
			}
			else if constexpr (inRange(sizeof(T), 0, 16))
			{
				if constexpr (sizeof(S) == 8) return _mm_sllv_epi64(a, xa);
				if constexpr (sizeof(S) == 4) return _mm_sllv_epi32(a, xa);
				if constexpr (sizeof(S) == 2) return _mm_sllv_epi16(a, xa);
			}
		}
		else
		{
			T ret;
			for (size_t i = 0; i < N; ++i) ret[i] = a[i] << amount[i];
			return ret;
		}
	}
	template<typename S, size_t N, typename I>
		requires (std::is_integral_v<I>&& std::is_integral_v<S>)
	__forceinline SIMD_Vector<S, N> shift_right(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& amount)
	{
		using T = SIMD_Vector<S, N>;
		if constexpr (capabilities::current == capabilities::zen4)
		{
			if constexpr (sizeof(T) > 64) return concat(shift_right(a.lo, amount.lo), shift_right(a.hi, amount.hi));
			else if constexpr (sizeof(S) == 1) //no 8 bit shifts in x86!
			{
				auto alo = vec_cvt<uint16_t>(vec_cvt<uint8_t>(a.lo)); //treat as unsigned bytes to avoid sign spillover
				auto ahi = vec_cvt<uint16_t>(vec_cvt<uint8_t>(a.hi));
				auto shlo = vec_cvt<S>(shift_right(alo, amount.lo));
				auto shhi = vec_cvt<S>(shift_right(ahi, amount.hi));
				return concat(shlo, shhi);
			}
			else {

				auto xa = vec_cvt<typename T::IntScalarType>(amount);
				if constexpr (inRange(sizeof(T), 33, 64))
				{
					if constexpr (sizeof(S) == 8) return _mm512_srlv_epi64(a, xa);
					if constexpr (sizeof(S) == 4) return _mm512_srlv_epi32(a, xa);
					if constexpr (sizeof(S) == 2) return _mm512_srlv_epi16(a, xa);
				}
				else if constexpr (inRange(sizeof(T), 17, 32))
				{
					if constexpr (sizeof(S) == 8) return _mm256_srlv_epi64(a, xa);
					if constexpr (sizeof(S) == 4) return _mm256_srlv_epi32(a, xa);
					if constexpr (sizeof(S) == 2) return _mm256_srlv_epi16(a, xa);
				}
				else if constexpr (inRange(sizeof(T), 0, 16))
				{
					if constexpr (sizeof(S) == 8) return _mm_srlv_epi64(a, xa);
					if constexpr (sizeof(S) == 4) return _mm_srlv_epi32(a, xa);
					if constexpr (sizeof(S) == 2) return _mm_srlv_epi16(a, xa);
				}
				else static_assert(always_false_v<S>, "invalid args shift_right");
			}
		}
		else
		{
			T ret;
			for (size_t i = 0; i < N; ++i) ret[i] = a[i] >> amount[i];
			return ret;
		}
	}

	template<typename S, size_t N, typename I>
	[[gnu::target("avx512vbmi")]] //todo: change this later
	__forceinline SIMD_Vector<S, N> permx(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& ind)
		requires (sizeof(SIMD_Vector<S, N>) <= 64) //for now, don't emulate, just route to native permutex
	{
		using T = SIMD_Vector<S, N>;
		if constexpr (capabilities::current == capabilities::zen4)
		{
			auto xind = vec_cvt<typename T::IntScalarType>(ind); //TODO: when implementing larger permutes: can overflow
			if constexpr (inRange(sizeof(T), 33, 64))
			{
				if constexpr (std::is_same_v<S, double>) return _mm512_permutexvar_pd(xind, a);
				if constexpr (std::is_same_v<S, float>) return _mm512_permutexvar_ps(xind, a);
				if constexpr (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>) return _mm512_permutexvar_epi64(xind, a);
				if constexpr (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>) return _mm512_permutexvar_epi32(xind, a);
				if constexpr (std::is_same_v<S, int16_t> || std::is_same_v<S, uint16_t>) return _mm512_permutexvar_epi16(xind, a);
				if constexpr (std::is_same_v<S, int8_t> || std::is_same_v<S, uint8_t>) return _mm512_permutexvar_epi8(xind, a);
			}
			else if constexpr (inRange(sizeof(T), 17, 32))
			{
				if constexpr (std::is_same_v<S, double>) return _mm256_permutexvar_pd(xind, a);
				if constexpr (std::is_same_v<S, float>) return _mm256_permutexvar_ps(xind, a);
				if constexpr (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>) return _mm256_permutexvar_epi64(xind, a);
				if constexpr (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>) return _mm256_permutexvar_epi32(xind, a);
				if constexpr (std::is_same_v<S, int16_t> || std::is_same_v<S, uint16_t>) return _mm256_permutexvar_epi16(xind, a);
				if constexpr (std::is_same_v<S, int8_t> || std::is_same_v<S, uint8_t>) return _mm256_permutexvar_epi8(xind, a);
			}
			else if constexpr (inRange(sizeof(T), 0, 16))
			{
				if constexpr (std::is_same_v<S, double>) return _mm_permutexvar_pd(xind, a);
				if constexpr (std::is_same_v<S, float>) return _mm_permutexvar_ps(xind, a);
				if constexpr (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>) return _mm_permutexvar_epi64(xind, a);
				if constexpr (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>) return _mm_permutexvar_epi32(xind, a);
				if constexpr (std::is_same_v<S, int16_t> || std::is_same_v<S, uint16_t>) return _mm_permutexvar_epi16(xind, a);
				if constexpr (std::is_same_v<S, int8_t> || std::is_same_v<S, uint8_t>) return _mm_permutexvar_epi8(xind, a);
			}
		}
		else
		{
			T ret;
			for (size_t i = 0; i < N; ++i) ret[i] = a[ind[i] & (N - 1)];
			return ret;
		}
	}

	template<typename S, size_t N, typename I>
	[[gnu::target("avx512vbmi")]] //todo: change this later
	__forceinline SIMD_Vector<S, N> permx2(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& ind, const SIMD_Vector<S, N>& b)
		requires (sizeof(SIMD_Vector<S, N>) <= 64)
	{
		using T = SIMD_Vector<S, N>;
		if constexpr (capabilities::current == capabilities::zen4)
		{
			auto xind = vec_cvt<typename T::IntScalarType>(ind);
			if constexpr (inRange(sizeof(T), 33, 64))
			{
				if constexpr (std::is_same_v<S, double>) return _mm512_permutex2var_pd(a, xind, b);
				if constexpr (std::is_same_v<S, float>) return _mm512_permutex2var_ps(a, xind, b);
				if constexpr (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>) return _mm512_permutex2var_epi64(a, xind, b);
				if constexpr (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>) return _mm512_permutex2var_epi32(a, xind, b);
				if constexpr (std::is_same_v<S, int16_t> || std::is_same_v<S, uint16_t>) return _mm512_permutex2var_epi16(a, xind, b);
				if constexpr (std::is_same_v<S, int8_t> || std::is_same_v<S, uint8_t>) return _mm512_permutex2var_epi8(a, xind, b);
			}
			else if constexpr (inRange(sizeof(T), 17, 32))
			{
				if constexpr (std::is_same_v<S, double>) return _mm256_permutex2var_pd(a, xind, b);
				if constexpr (std::is_same_v<S, float>) return _mm256_permutex2var_ps(a, xind, b);
				if constexpr (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>) return _mm256_permutex2var_epi64(a, xind, b);
				if constexpr (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>) return _mm256_permutex2var_epi32(a, xind, b);
				if constexpr (std::is_same_v<S, int16_t> || std::is_same_v<S, uint16_t>) return _mm256_permutex2var_epi16(a, xind, b);
				if constexpr (std::is_same_v<S, int8_t> || std::is_same_v<S, uint8_t>) return _mm256_permutex2var_epi8(a, xind, b);
			}
			else if constexpr (inRange(sizeof(T), 0, 16))
			{
				if constexpr (std::is_same_v<S, double>) return _mm_permutex2var_pd(a, xind, b);
				if constexpr (std::is_same_v<S, float>) return _mm_permutex2var_ps(a, xind, b);
				if constexpr (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>) return _mm_permutex2var_epi64(a, xind, b);
				if constexpr (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>) return _mm_permutex2var_epi32(a, xind, b);
				if constexpr (std::is_same_v<S, int16_t> || std::is_same_v<S, uint16_t>) return _mm_permutex2var_epi16(a, xind, b);
				if constexpr (std::is_same_v<S, int8_t> || std::is_same_v<S, uint8_t>) return _mm_permutex2var_epi8(a, xind, b);
			}
		}
		else
		{
			T ret;
			for (size_t i = 0; i < N; ++i)
			{
				auto j = ind[i] & (2 * N - 1);
				ret[i] = j < N ? a[j] : b[j - N];
			}
			return ret;
		}
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N / 2> upper_half(const SIMD_Vector<S, N>& a)
	{
		static_assert(N % 2 == 0, "upper_half is called on an odd element count vector");
		return a.hi;
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N / 2> lower_half(const SIMD_Vector<S, N>& a)
	{
		static_assert(N % 2 == 0, "lower_half is called on an odd element count vector");
		return a.lo;
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<float, N> sqrtf(const SIMD_Vector<S, N>& a)
	{
		auto x = vec_cvt<float>(a);
		if constexpr (capabilities::current == capabilities::zen4)
		{
			if constexpr (sizeof(x) > 64) return concat(sqrtf(x.lo), sqrtf(x.hi));
			else if constexpr (inRange(sizeof(x), 33, 64)) return _mm512_sqrt_ps(x);
			else if constexpr (inRange(sizeof(x), 17, 32)) return _mm256_sqrt_ps(x);
			else if constexpr (inRange(sizeof(x), 0, 16)) return _mm_sqrt_ps(x);
		}
		else
		{
			SIMD_Vector<float, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = std::sqrtf(a[i]);
			return ret;
		}
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<double, N> sqrtd(const SIMD_Vector<S, N>& a)
	{
		auto x = vec_cvt<double>(a);
		if constexpr (capabilities::current == capabilities::zen4)
		{
			if constexpr (sizeof(x) > 64) return concat(sqrtd(x.lo), sqrtd(x.hi));
			else if constexpr (inRange(sizeof(x), 33, 64)) return _mm512_sqrt_pd(x);
			else if constexpr (inRange(sizeof(x), 17, 32)) return _mm256_sqrt_pd(x);
			else if constexpr (inRange(sizeof(x), 0, 16)) return _mm_sqrt_pd(x);
		}
		else
		{
			SIMD_Vector<float, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = std::sqrt<double>(a[i]);
			return ret;
		}
	}

	/*
	template<typename S, size_t N1, size_t N2, size_t N3, size_t N4>
	SIMD_Vector<S, N1 + N2 + N3 + N4> concat(const SIMD_Vector<S, N1>& v0, const SIMD_Vector<S, N2>& v1, const SIMD_Vector<S, N3>& v2, const SIMD_Vector<S, N4>& v3)
	{
		return concat(concat(v0, v1), concat(v2, v3));
	}

	template<typename S, size_t N1, size_t N2, size_t N3, size_t N4, size_t N5, size_t N6, size_t N7, size_t N8>
	SIMD_Vector<S, N1 + N2 + N3 + N4 + N5 + N6 + N7 + N8> concat(const SIMD_Vector<S, N1>& v0, const SIMD_Vector<S, N2>& v1, const SIMD_Vector<S, N3>& v2, const SIMD_Vector<S, N4>& v3, const SIMD_Vector<S, N5>& v4, const SIMD_Vector<S, N6>& v5, const SIMD_Vector<S, N7>& v6, const SIMD_Vector<S, N8>& v7)
	{
		return concat(concat(v0, v1, v2, v3), concat(v4, v5, v6, v7));
	}*/

	template<typename T, typename S, size_t N>
	__forceinline T reinterpret(const SIMD_Vector<S, N>& value)
	{
		T ret;
		std::memcpy(&ret, &value, std::min(sizeof(ret), sizeof(value)));
		return ret;
	}

	template<size_t Part, size_t PartCount, typename S, size_t N>
	__forceinline SIMD_Vector<S, N / PartCount> extract(const SIMD_Vector<S, N>& value)
	{
		static_assert(N % PartCount == 0, "extract: number elements not divisible by part count");
		static_assert(Part < PartCount, "extract: part number to extract is greater or equal to part count");
		static_assert(isPowerOf2(PartCount));

		if constexpr (PartCount == 1) return value;
		else if constexpr (PartCount == 2) return Part == 0 ? value.lo : value.hi;
		else
		{
			//test: extract<6, 8> should return 8th right in before the last element.
			// |0,1,2,3,4,5,6,7| <-- 6 here
			// pivot 4 == 8/2. 6 is NOT < 4:
			// |0,1,2,3|4,5,6,7| <-- inspect upper half, deduct N/2, 6-4=2, correct
			//         |0,1,2,3| <-- as seen from next level extract. It gets index 2
			// pivot 2 == 4/2. 2 in NOT < 2, so inspect upper. 2 (previous index) - 2 = 0
			//             |0,1| <-- N==2, Part == 0, return lo, correct
			//halve vector each time, i.e. each step returns extract on one of the halves

			constexpr size_t HalfParts = PartCount / 2;
			if constexpr (Part < HalfParts) return extract<Part, HalfParts>(value.lo);
			else return extract<Part - HalfParts, HalfParts>(value.hi);
		}
	}

	template<size_t Part, size_t N2, typename S, size_t N>
	__forceinline SIMD_Vector<S, N> insert(const SIMD_Vector<S, N>& to, const SIMD_Vector<S, N2>& what)
	{
		static_assert(N2 <= N, "insert: attempting to insert value to a vector that is smaller than it");
		static_assert(N % N2 == 0, "insert: attempting to insert value to a vector which has size not divisible by insertee's size");
		static_assert(Part < N / N2, "insert: attempting to insert value past the target vector's bounds");
		static_assert(isPowerOf2(N), "insert: N is not power of 2");
		static_assert(isPowerOf2(N2), "insert: N2 is not power of 2");

		if constexpr (N == N2) return what;
		SIMD_Vector<S, N> ret;
		if constexpr (N == N2 * 2)
		{
			if constexpr (Part == 0) { ret.lo = what; ret.hi = to.hi; }
			else { ret.lo = to.lo; ret.hi = what; }
			return ret;
		}
		/* //TODO: implement this
		else
		{
			constexpr size_t HalfParts = N / 2;
			if constexpr (Part < HalfParts) return extract<Part, HalfParts>(value.lo);
			else return extract<Part - HalfParts, HalfParts>(value.hi);
		}*/
		ret = to;
		size_t addr = size_t(&ret) + sizeof(what) * Part;
		memcpy((void*)addr, &what, sizeof(what));
		return ret;
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> mask_mov(const SIMD_Vector<S, N>& ifBitClear, const typename SIMD_Vector<S, N>::MaskType& mask, const SIMD_Vector<S, N>& ifBitSet)
	{
		using T = SIMD_Vector<S, N>;
		if constexpr (sizeof(T) > 64) return concat(mask_mov(ifBitClear.lo, mask.lo(), ifBitSet.lo), mask_mov(ifBitClear.hi, mask.hi(), ifBitSet.hi));
		else if constexpr (capabilities::current.AVX512.F)
		{
			if constexpr (inRange(sizeof(T), 33, 64))
			{
				if constexpr (std::is_same_v<S, double>) return _mm512_mask_mov_pd(ifBitClear, mask, ifBitSet);
				if constexpr (std::is_same_v<S, float>) return _mm512_mask_mov_ps(ifBitClear, mask, ifBitSet);
				if constexpr (std::is_same_v<S, uint64_t> || std::is_same_v<S, int64_t>) return _mm512_mask_mov_epi64(ifBitClear, mask, ifBitSet);
				if constexpr (std::is_same_v<S, uint32_t> || std::is_same_v<S, int32_t>) return _mm512_mask_mov_epi32(ifBitClear, mask, ifBitSet);
				if constexpr (std::is_same_v<S, uint16_t> || std::is_same_v<S, int16_t>) return _mm512_mask_mov_epi16(ifBitClear, mask, ifBitSet);
				if constexpr (std::is_same_v<S, uint8_t> || std::is_same_v<S, int8_t>) return _mm512_mask_mov_epi8(ifBitClear, mask, ifBitSet);
			}
			else if constexpr (inRange(sizeof(T), 17, 32))
			{
				if constexpr (std::is_same_v<S, double>) return _mm256_mask_mov_pd(ifBitClear, mask, ifBitSet);
				if constexpr (std::is_same_v<S, float>) return _mm256_mask_mov_ps(ifBitClear, mask, ifBitSet);
				if constexpr (std::is_same_v<S, uint64_t> || std::is_same_v<S, int64_t>) return _mm256_mask_mov_epi64(ifBitClear, mask, ifBitSet);
				if constexpr (std::is_same_v<S, uint32_t> || std::is_same_v<S, int32_t>) return _mm256_mask_mov_epi32(ifBitClear, mask, ifBitSet);
				if constexpr (std::is_same_v<S, uint16_t> || std::is_same_v<S, int16_t>) return _mm256_mask_mov_epi16(ifBitClear, mask, ifBitSet);
				if constexpr (std::is_same_v<S, uint8_t> || std::is_same_v<S, int8_t>) return _mm256_mask_mov_epi8(ifBitClear, mask, ifBitSet);
			}
			else if constexpr (inRange(sizeof(T), 0, 16))
			{
				if constexpr (std::is_same_v<S, double>) return _mm_mask_mov_pd(ifBitClear, mask, ifBitSet);
				if constexpr (std::is_same_v<S, float>) return _mm_mask_mov_ps(ifBitClear, mask, ifBitSet);
				if constexpr (std::is_same_v<S, uint64_t> || std::is_same_v<S, int64_t>) return _mm_mask_mov_epi64(ifBitClear, mask, ifBitSet);
				if constexpr (std::is_same_v<S, uint32_t> || std::is_same_v<S, int32_t>) return _mm_mask_mov_epi32(ifBitClear, mask, ifBitSet);
				if constexpr (std::is_same_v<S, uint16_t> || std::is_same_v<S, int16_t>) return _mm_mask_mov_epi16(ifBitClear, mask, ifBitSet);
				if constexpr (std::is_same_v<S, uint8_t> || std::is_same_v<S, int8_t>) return _mm_mask_mov_epi8(ifBitClear, mask, ifBitSet);
			}
		}
		else
		{
			SIMD_Vector<S, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = mask[i] ? ifBitSet[i] : ifBitClear[i];
			return ret;
		}
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> maskz_mov(const typename SIMD_Vector<S, N>::MaskType& mask, const SIMD_Vector<S, N>& ifBitSet)
	{
		using T = SIMD_Vector<S, N>;
		if constexpr (capabilities::current == capabilities::zen4)
		{
			if constexpr (sizeof(T) > 64) return concat(maskz_mov(mask.lo(), ifBitSet.lo), maskz_mov(mask.hi(), ifBitSet.hi));
			else if constexpr (inRange(sizeof(T), 33, 64))
			{
				if constexpr (std::is_same_v<S, double>) return _mm512_maskz_mov_pd(mask, ifBitSet);
				if constexpr (std::is_same_v<S, float>) return _mm512_maskz_mov_ps(mask, ifBitSet);
				if constexpr (std::is_same_v<S, uint64_t> || std::is_same_v<S, int64_t>) return _mm512_maskz_mov_epi64(mask, ifBitSet);
				if constexpr (std::is_same_v<S, uint32_t> || std::is_same_v<S, int32_t>) return _mm512_maskz_mov_epi32(mask, ifBitSet);
				if constexpr (std::is_same_v<S, uint16_t> || std::is_same_v<S, int16_t>) return _mm512_maskz_mov_epi16(mask, ifBitSet);
				if constexpr (std::is_same_v<S, uint8_t> || std::is_same_v<S, int8_t>) return _mm512_maskz_mov_epi8(mask, ifBitSet);
			}
			else if constexpr (inRange(sizeof(T), 17, 32))
			{
				if constexpr (std::is_same_v<S, double>) return _mm256_maskz_mov_pd(mask, ifBitSet);
				if constexpr (std::is_same_v<S, float>) return _mm256_maskz_mov_ps(mask, ifBitSet);
				if constexpr (std::is_same_v<S, uint64_t> || std::is_same_v<S, int64_t>) return _mm256_maskz_mov_epi64(mask, ifBitSet);
				if constexpr (std::is_same_v<S, uint32_t> || std::is_same_v<S, int32_t>) return _mm256_maskz_mov_epi32(mask, ifBitSet);
				if constexpr (std::is_same_v<S, uint16_t> || std::is_same_v<S, int16_t>) return _mm256_maskz_mov_epi16(mask, ifBitSet);
				if constexpr (std::is_same_v<S, uint8_t> || std::is_same_v<S, int8_t>) return _mm256_maskz_mov_epi8(mask, ifBitSet);
			}
			else if constexpr (inRange(sizeof(T), 0, 16))
			{
				if constexpr (std::is_same_v<S, double>) return _mm_maskz_mov_pd(mask, ifBitSet);
				if constexpr (std::is_same_v<S, float>) return _mm_maskz_mov_ps(mask, ifBitSet);
				if constexpr (std::is_same_v<S, uint64_t> || std::is_same_v<S, int64_t>) return _mm_maskz_mov_epi64(mask, ifBitSet);
				if constexpr (std::is_same_v<S, uint32_t> || std::is_same_v<S, int32_t>) return _mm_maskz_mov_epi32(mask, ifBitSet);
				if constexpr (std::is_same_v<S, uint16_t> || std::is_same_v<S, int16_t>) return _mm_maskz_mov_epi16(mask, ifBitSet);
				if constexpr (std::is_same_v<S, uint8_t> || std::is_same_v<S, int8_t>) return _mm_maskz_mov_epi8(mask, ifBitSet);
			}
		}
		else
		{
			SIMD_Vector<S, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = mask[i] ? ifBitSet[i] : 0;
			return ret;
		}
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> blend(const typename SIMD_Vector<S, N>::MaskType& mask, const SIMD_Vector<S, N>& ifBitClear, const SIMD_Vector<S, N>& ifBitSet)
	{
		return mask_mov(ifBitClear, mask, ifBitSet);
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> load(const void* p, const typename SIMD_Vector<S, N>::MaskType& mask, const SIMD_Vector<S, N>& src)
	{
		using T = SIMD_Vector<S, N>;
		constexpr size_t sz = sizeof(T);
		if constexpr (capabilities::current.AVX512.F)
		{
			if constexpr (sz > 64)
			{
				size_t addr = size_t(p) + sz / 2;
				return concat(load(p, mask.lo(), src.lo), load((const void*)(addr), mask.hi(), src.hi));
			}
			else if constexpr (inRange(sz, 33, 64))
			{
				if constexpr (std::is_same_v<S, int8_t> || std::is_same_v<S, uint8_t>) return _mm512_mask_loadu_epi8(src, mask, p);
				if constexpr (std::is_same_v<S, int16_t> || std::is_same_v<S, uint16_t>) return _mm512_mask_loadu_epi16(src, mask, p);
				if constexpr (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>) return _mm512_mask_loadu_epi32(src, mask, p);
				if constexpr (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>) return _mm512_mask_loadu_epi64(src, mask, p);
				if constexpr (std::is_same_v<S, float>) return _mm512_mask_loadu_ps(src, mask, p);
				if constexpr (std::is_same_v<S, double>) return _mm512_mask_loadu_pd(src, mask, p);
			}
			else if constexpr (inRange(sz, 17, 32))
			{
				if constexpr (std::is_same_v<S, int8_t> || std::is_same_v<S, uint8_t>) return _mm256_mask_loadu_epi8(src, mask, p);
				if constexpr (std::is_same_v<S, int16_t> || std::is_same_v<S, uint16_t>) return _mm256_mask_loadu_epi16(src, mask, p);
				if constexpr (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>) return _mm256_mask_loadu_epi32(src, mask, p);
				if constexpr (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>) return _mm256_mask_loadu_epi64(src, mask, p);
				if constexpr (std::is_same_v<S, float>) return _mm256_mask_loadu_ps(src, mask, p);
				if constexpr (std::is_same_v<S, double>) return _mm256_mask_loadu_pd(src, mask, p);
			}
			else if constexpr (inRange(sz, 0, 16))
			{
				if constexpr (std::is_same_v<S, int8_t> || std::is_same_v<S, uint8_t>) return _mm_mask_loadu_epi8(src, mask, p);
				if constexpr (std::is_same_v<S, int16_t> || std::is_same_v<S, uint16_t>) return _mm_mask_loadu_epi16(src, mask, p);
				if constexpr (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>) return _mm_mask_loadu_epi32(src, mask, p);
				if constexpr (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>) return _mm_mask_loadu_epi64(src, mask, p);
				if constexpr (std::is_same_v<S, float>) return _mm_mask_loadu_ps(src, mask, p);
				if constexpr (std::is_same_v<S, double>) return _mm_mask_loadu_pd(src, mask, p);
			}
		}
		else
		{
			//TODO: move before recursion
			SIMD_Vector<S, N> ret;
			const S* sp = (const S*)p;
			for (size_t i = 0; i < N; ++i)
				ret[i] = mask[i] ? sp[i] : src[i];
			return ret;
		}
		//else static_assert(always_false_v<S>, "Unsupported arguments for SIMD_Vector load.");
	}

	template<typename S, size_t N>
	__forceinline void store(const SIMD_Vector<S, N>& v, void* p, const typename SIMD_Vector<S, N>::MaskType& mask)
	{
		using T = SIMD_Vector<S, N>;
		constexpr size_t sz = sizeof(T);
		if constexpr (capabilities::current.AVX512.F)
		{
			if constexpr (sz > 64)
			{
				size_t addr = size_t(p) + sz / 2;
				store(v.lo, p, mask.lo());
				store(v.hi, (void*)addr, mask.hi());
			}
			else if constexpr (inRange(sz, 33, 64))
			{
				if constexpr (std::is_same_v<S, int8_t> || std::is_same_v<S, uint8_t>) return _mm512_mask_storeu_epi8(p, mask, v);
				if constexpr (std::is_same_v<S, int16_t> || std::is_same_v<S, uint16_t>) return _mm512_mask_storeu_epi16(p, mask, v);
				if constexpr (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>) return _mm512_mask_storeu_epi32(p, mask, v);
				if constexpr (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>) return _mm512_mask_storeu_epi64(p, mask, v);
				if constexpr (std::is_same_v<S, float>) return _mm512_mask_storeu_ps(p, mask, v);
				if constexpr (std::is_same_v<S, double>) return _mm512_mask_storeu_pd(p, mask, v);
			}
			else if constexpr (inRange(sz, 17, 32))
			{
				if constexpr (std::is_same_v<S, int8_t> || std::is_same_v<S, uint8_t>) return _mm256_mask_storeu_epi8(p, mask, v);
				if constexpr (std::is_same_v<S, int16_t> || std::is_same_v<S, uint16_t>) return _mm256_mask_storeu_epi16(p, mask, v);
				if constexpr (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>) return _mm256_mask_storeu_epi32(p, mask, v);
				if constexpr (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>) return _mm256_mask_storeu_epi64(p, mask, v);
				if constexpr (std::is_same_v<S, float>) return _mm256_mask_storeu_ps(p, mask, v);
				if constexpr (std::is_same_v<S, double>) return _mm256_mask_storeu_pd(p, mask, v);
			}
			else if constexpr (inRange(sz, 0, 16))
			{
				if constexpr (std::is_same_v<S, int8_t> || std::is_same_v<S, uint8_t>) return _mm_mask_storeu_epi8(p, mask, v);
				if constexpr (std::is_same_v<S, int16_t> || std::is_same_v<S, uint16_t>) return _mm_mask_storeu_epi16(p, mask, v);
				if constexpr (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>) return _mm_mask_storeu_epi32(p, mask, v);
				if constexpr (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>) return _mm_mask_storeu_epi64(p, mask, v);
				if constexpr (std::is_same_v<S, float>) return _mm_mask_storeu_ps(p, mask, v);
				if constexpr (std::is_same_v<S, double>) return _mm_mask_storeu_pd(p, mask, v);
			}
		}
		else
		{
			//TODO: move this scalar fallback on top, before recursive splitting
			S* sp = (S*)p;
			for (size_t i = 0; i < N; ++i)
			{
				if (mask[i]) sp[i] = v[i];
			}
		}
		//else static_assert(always_false_v<S>, "Unsupported arguments for SIMD_Vector load.");
	}

	template<typename S, size_t N, size_t Scale, typename I>
		requires (std::is_integral_v<I> && sizeof(I) <= 8)
	__forceinline SIMD_Vector<S, N> gather(const void* base, const SIMD_Vector<I, N>& ind, const typename SIMD_Vector<S, N>::MaskType& mask, const SIMD_Vector<S, N>& src)
	{
		if constexpr (capabilities::current == capabilities::zen4)
		{
			if constexpr (sizeof(S) < 4)
			{
				//TODO: make aligned version that skips this check
				size_t baseAddr = size_t(base);
				SIMD_Vector<S, N> ret;
				size_t modAddr = baseAddr % 4;
				if (modAddr == 0)
				{
					constexpr size_t indShift = sizeof(S) == 2 ? 1 : 2;
					auto dwords = gather<int32_t, N, Scale>(base, ind >> indShift, mask);
					auto shifts = ind & (indShift == 1 ? 1 : 3);
					auto prefab = dwords >> (shifts << 3); //shift into proper places
					ret = vec_cvt<S>(prefab);
				}
				else [[unlikely]] //unaligned pointer, nothing we can do but read scalarly
				{
					// x86 provides gathers only for 32-bit and 64-bit elements.
					// Emulating byte/word gathers via widened accesses is not universally safe:
					// a valid 1- or 2-byte element may reside at the end of a mapped page,
					// while a widened 4-byte read crosses into an unmapped page and faults.
					//
					// Without additional guarantees (tail padding or known-safe overread),
					// exact-width scalar loads are the only fully-correct implementation.
					SIMD_Vector<S, N> ret;
					for (size_t i = 0; i < N; ++i)
					{
						if (!mask[i]) continue;
						size_t a = baseAddr + Scale * ind[i];
						ret[i] = *((const S*)(a));
					}
				}
				return mask_mov(src, mask, ret);
			}

			using CanonicalIndex_t = std::conditional_t<(sizeof(I) <= 4), int32_t, int64_t>;
			//progressively sanitize inputs. There're no gathers for small indices, so first, need to extend them.
			//TODO: can unite this check with Scale and use smaller indices?
			if constexpr (!std::is_same_v<I, CanonicalIndex_t>) return gather<S, N, Scale>(base, vec_cvt<CanonicalIndex_t>(ind), mask, src);

			//if we get here, means that indices are already in good format (4-byte or 8-byte)
			using RetVec_t = SIMD_Vector<S, N>;
			using IndVec_t = SIMD_Vector<I, N>;
			constexpr size_t MaxSize = std::max(sizeof(RetVec_t), sizeof(IndVec_t));

			//break up large gather into halves
			if constexpr (MaxSize > 64) return concat(
				gather<S, N / 2, Scale, I>(base, ind.lo, mask.lo(), src.lo),
				gather<S, N / 2, Scale, I>(base, ind.hi, mask.hi(), src.hi));

			//if scale is natively supported, we may already gather if inputs are clean enough, or sanitize further.
			//This is the main gather engine, everything funnels into here
			if constexpr (Scale == 1 || Scale == 2 || Scale == 4 || Scale == 8)
			{
				//dispatch to native gathers, since small gathers all non-native gathers already processed
				if constexpr (sizeof(S) == 4 || sizeof(S) == 8)
				{
					if constexpr (inRange(MaxSize, 33, 64))
					{
						if constexpr (std::is_same_v<I, int64_t> && std::is_same_v<S, double>) return _mm512_mask_i64gather_pd(src, mask, ind, base, Scale);
						if constexpr (std::is_same_v<I, int64_t> && std::is_same_v<S, float>) return _mm512_mask_i64gather_ps(src, mask, ind, base, Scale);
						if constexpr (std::is_same_v<I, int64_t> && (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>)) return  _mm512_mask_i64gather_epi64(src, mask, ind, base, Scale);
						if constexpr (std::is_same_v<I, int64_t> && (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>)) return _mm512_mask_i64gather_epi32(src, mask, ind, base, Scale);

						if constexpr (std::is_same_v<I, int32_t> && std::is_same_v<S, double>) return _mm512_mask_i32gather_pd(src, mask, ind, base, Scale);
						if constexpr (std::is_same_v<I, int32_t> && std::is_same_v<S, float>) return _mm512_mask_i32gather_ps(src, mask, reinterpret<__m512i>(ind), base, Scale);
						if constexpr (std::is_same_v<I, int32_t> && (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>)) return  _mm512_mask_i32gather_epi64(src, mask, ind, base, Scale);
						if constexpr (std::is_same_v<I, int32_t> && (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>)) return  _mm512_mask_i32gather_epi32(src, mask, ind, base, Scale);
					}
					//TODO: when making older gather versions, they use mask vectors instead of registers and _mm256_mask_* instead of mmask
					else if constexpr (inRange(MaxSize, 17, 32))
					{
						if constexpr (std::is_same_v<I, int64_t> && std::is_same_v<S, double>) return _mm256_mmask_i64gather_pd(src, mask, ind, base, Scale);
						if constexpr (std::is_same_v<I, int64_t> && std::is_same_v<S, float>) return _mm256_mmask_i64gather_ps(src, mask, ind, base, Scale);
						if constexpr (std::is_same_v<I, int64_t> && (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>)) return  _mm256_mmask_i64gather_epi64(src, mask, ind, base, Scale);
						if constexpr (std::is_same_v<I, int64_t> && (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>)) return  _mm256_mmask_i64gather_epi32(src, mask, ind, base, Scale);

						if constexpr (std::is_same_v<I, int32_t> && std::is_same_v<S, double>) return _mm256_mmask_i32gather_pd(src, mask, ind, base, Scale);
						if constexpr (std::is_same_v<I, int32_t> && std::is_same_v<S, float>) return _mm256_mmask_i32gather_ps(src, mask, ind, base, Scale);
						if constexpr (std::is_same_v<I, int32_t> && (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>)) return _mm256_mmask_i32gather_epi64(src, mask, ind, base, Scale);
						if constexpr (std::is_same_v<I, int32_t> && (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>)) return _mm256_mmask_i32gather_epi32(src, mask, ind, base, Scale);
					}
					else if constexpr (inRange(MaxSize, 0, 16))
					{
						if constexpr (std::is_same_v<I, int64_t> && std::is_same_v<S, double>) return _mm_mmask_i64gather_pd(src, mask, ind, base, Scale);
						if constexpr (std::is_same_v<I, int64_t> && std::is_same_v<S, float>) return _mm_mmask_i64gather_ps(src, mask, ind, base, Scale);
						if constexpr (std::is_same_v<I, int64_t> && (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>)) return _mm_mmask_i64gather_epi64(src, mask, ind, base, Scale);
						if constexpr (std::is_same_v<I, int64_t> && (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>)) return _mm_mmask_i64gather_epi32(src, mask, ind, base, Scale);

						if constexpr (std::is_same_v<I, int32_t> && std::is_same_v<S, double>) return  _mm_mmask_i32gather_pd(src, mask, ind, base, Scale);
						if constexpr (std::is_same_v<I, int32_t> && std::is_same_v<S, float>) return _mm_mmask_i32gather_ps(src, mask, ind, base, Scale);
						if constexpr (std::is_same_v<I, int32_t> && (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>)) return _mm_mmask_i32gather_epi64(src, mask, ind, base, Scale);
						if constexpr (std::is_same_v<I, int32_t> && (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>)) return _mm_mmask_i32gather_epi32(src, mask, ind, base, Scale);
					}
					else static_assert(always_false_v<S>, "Native gather case got to failure branch. This should never happen");
				}
			}
			else
			{
				//if scale is not native, emulate it by gathering with scale 1 and manually calculated byte offsets. 
				//TODO: Can optimize a little by checking if Scale*maxint(I) fits into smaller sizes
				return gather<S, N, 1>(base, mul(vec_cvt<int64_t>(ind), Scale), mask, src);
			}
		}
		else
		{
			SIMD_Vector<S, N> ret;
			size_t baseAddr = size_t(base);
			for (size_t i = 0; i < N; ++i) ret[i] = mask[i] ? *(const S*)(baseAddr + Scale * ind[i]) : src[i];
			return ret;
		}
	}

	template<typename S, size_t N, size_t Scale, typename I>
	__forceinline void scatter(const SIMD_Vector<S, N>& vec, void* base, const SIMD_Vector<I, N>& ind, const typename SIMD_Vector<S, N>::MaskType& mask)
	{
		if constexpr (capabilities::current == capabilities::zen4)
		{
			if constexpr (sizeof(S) < 4)
			{
				//Scalar fallback for small elements. In theory, you can gather, replace needed bytes and scatter back, but that's a whole other can of worms.
				//Plus, that may not actually be faster than just scalarizing, especially on CPUs with bad gather
				size_t baseAddr = size_t(base);
				for (size_t i = 0; i < N; ++i)
				{
					if (!mask[i]) continue;
					size_t a = baseAddr + Scale * ind[i];
					*((S*)(a)) = vec[i];
				}
				return;
			}

			if constexpr (!(Scale == 1 || Scale == 2 || Scale == 4 || Scale == 8)) return scatter<S, N, 1>(vec, base, vec_cvt<int64_t>(ind) * Scale, mask);

			using CanonInd_t = std::conditional_t<sizeof(I) <= 4, int32_t, int64_t>;
			if (!std::is_same_v<CanonInd_t, I>) return scatter<S, N, Scale>(vec, base, vec_cvt<CanonInd_t>(ind), mask);

			using IndVec_t = SIMD_Vector<I, N>;
			using RetVec_t = SIMD_Vector<S, N>;
			constexpr size_t MaxSize = std::max(sizeof(RetVec_t), sizeof(IndVec_t));
			if constexpr (MaxSize > 64)
			{
				scatter(vec.lo, base, ind.lo, mask);
				scatter(vec.hi, base, ind.hi, mask);
				return;
			}
			else if constexpr (inRange(MaxSize, 33, 64))
			{
				if constexpr (std::is_same_v<I, int64_t> && std::is_same_v<S, double>) return _mm512_mask_i64scatter_pd(base, mask, ind, vec, Scale);
				if constexpr (std::is_same_v<I, int64_t> && std::is_same_v<S, float>) return _mm512_mask_i64scatter_ps(base, mask, ind, vec, Scale);
				if constexpr (std::is_same_v<I, int64_t> && (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>)) return _mm512_mask_i64scatter_epi64(base, mask, ind, vec, Scale);
				if constexpr (std::is_same_v<I, int64_t> && (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>)) return _mm512_mask_i64scatter_epi32(base, mask, ind, vec, Scale);

				if constexpr (std::is_same_v<I, int32_t> && std::is_same_v<S, double>) return _mm512_mask_i32scatter_pd(base, mask, ind, vec, Scale);
				if constexpr (std::is_same_v<I, int32_t> && std::is_same_v<S, float>) return _mm512_mask_i32scatter_ps(base, mask, ind, vec, Scale);
				if constexpr (std::is_same_v<I, int32_t> && (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>)) return _mm512_mask_i32scatter_epi64(base, mask, ind, vec, Scale);
				if constexpr (std::is_same_v<I, int32_t> && (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>)) return _mm512_mask_i32scatter_epi32(base, mask, ind, vec, Scale);
			}
			else if constexpr (inRange(MaxSize, 17, 32))
			{
				if constexpr (std::is_same_v<I, int64_t> && std::is_same_v<S, double>) return _mm256_mask_i64scatter_pd(base, mask, ind, vec, Scale);
				if constexpr (std::is_same_v<I, int64_t> && std::is_same_v<S, float>) return _mm256_mask_i64scatter_ps(base, mask, ind, vec, Scale);
				if constexpr (std::is_same_v<I, int64_t> && (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>)) return _mm256_mask_i64scatter_epi64(base, mask, ind, vec, Scale);
				if constexpr (std::is_same_v<I, int64_t> && (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>)) return _mm256_mask_i64scatter_epi32(base, mask, ind, vec, Scale);

				if constexpr (std::is_same_v<I, int32_t> && std::is_same_v<S, double>) return _mm256_mask_i32scatter_pd(base, mask, ind, vec, Scale);
				if constexpr (std::is_same_v<I, int32_t> && std::is_same_v<S, float>) return _mm256_mask_i32scatter_ps(base, mask, ind, vec, Scale);
				if constexpr (std::is_same_v<I, int32_t> && (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>)) return _mm256_mask_i32scatter_epi64(base, mask, ind, vec, Scale);
				if constexpr (std::is_same_v<I, int32_t> && (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>)) return _mm256_mask_i32scatter_epi32(base, mask, ind, vec, Scale);
			}
			else if constexpr (inRange(MaxSize, 0, 16))
			{
				if constexpr (std::is_same_v<I, int64_t> && std::is_same_v<S, double>) return _mm_mask_i64scatter_pd(base, mask, ind, vec, Scale);
				if constexpr (std::is_same_v<I, int64_t> && std::is_same_v<S, float>) return _mm_mask_i64scatter_ps(base, mask, ind, vec, Scale);
				if constexpr (std::is_same_v<I, int64_t> && (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>)) return _mm_mask_i64scatter_epi64(base, mask, ind, vec, Scale);
				if constexpr (std::is_same_v<I, int64_t> && (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>)) return _mm_mask_i64scatter_epi32(base, mask, ind, vec, Scale);

				if constexpr (std::is_same_v<I, int32_t> && std::is_same_v<S, double>) return _mm_mask_i32scatter_pd(base, mask, ind, vec, Scale);
				if constexpr (std::is_same_v<I, int32_t> && std::is_same_v<S, float>) return _mm_mask_i32scatter_ps(base, mask, ind, vec, Scale);
				if constexpr (std::is_same_v<I, int32_t> && (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>)) return _mm_mask_i32scatter_epi64(base, mask, ind, vec, Scale);
				if constexpr (std::is_same_v<I, int32_t> && (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>)) return _mm_mask_i32scatter_epi32(base, mask, ind, vec, Scale);
			}
		}
		else
		{
			size_t baseAddr = size_t(base);
			for (size_t i = 0; i < N; ++i) if (mask[i]) *(S*)(baseAddr + Scale * ind[i]) = vec[i];
			return;
		}
	}

	template<typename S, size_t N>
		requires (N <= 64)
	__forceinline typename SIMD_Vector<S, N>::MaskType cmp_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using T = SIMD_Vector<S, N>;
		if constexpr (capabilities::current == capabilities::zen4)
		{
			if constexpr (sizeof(T) > 64) return concat_masks(cmp_equal(a.lo, b.lo), cmp_equal(a.hi, b.hi));
			else if constexpr (inRange(sizeof(T), 33, 64))
			{
				if constexpr (std::is_same_v<S, double>) return _mm512_cmp_pd_mask(a, b, _CMP_EQ_OQ);
				else if constexpr (std::is_same_v<S, float>) return _mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ);
				else if constexpr (std::is_same_v<S, int64_t>) return _mm512_cmpeq_epi64_mask(a, b);
				else if constexpr (std::is_same_v<S, uint64_t>) return _mm512_cmpeq_epu64_mask(a, b);
				else if constexpr (std::is_same_v<S, int32_t>) return _mm512_cmpeq_epi32_mask(a, b);
				else if constexpr (std::is_same_v<S, uint32_t>) return _mm512_cmpeq_epu32_mask(a, b);
				else if constexpr (std::is_same_v<S, int16_t>) return _mm512_cmpeq_epi16_mask(a, b);
				else if constexpr (std::is_same_v<S, uint16_t>) return _mm512_cmpeq_epu16_mask(a, b);
				else if constexpr (std::is_same_v<S, int8_t>) return _mm512_cmpeq_epi8_mask(a, b);
				else if constexpr (std::is_same_v<S, uint8_t>) return _mm512_cmpeq_epu8_mask(a, b);
			}
			else if constexpr (inRange(sizeof(T), 17, 32))
			{
				if constexpr (std::is_same_v<S, double>) return _mm256_cmp_pd_mask(a, b, _CMP_EQ_OQ);
				else if constexpr (std::is_same_v<S, float>) return _mm256_cmp_ps_mask(a, b, _CMP_EQ_OQ);
				else if constexpr (std::is_same_v<S, int64_t>) return _mm256_cmpeq_epi64_mask(a, b);
				else if constexpr (std::is_same_v<S, uint64_t>) return _mm256_cmpeq_epu64_mask(a, b);
				else if constexpr (std::is_same_v<S, int32_t>) return _mm256_cmpeq_epi32_mask(a, b);
				else if constexpr (std::is_same_v<S, uint32_t>) return _mm256_cmpeq_epu32_mask(a, b);
				else if constexpr (std::is_same_v<S, int16_t>) return _mm256_cmpeq_epi16_mask(a, b);
				else if constexpr (std::is_same_v<S, uint16_t>) return _mm256_cmpeq_epu16_mask(a, b);
				else if constexpr (std::is_same_v<S, int8_t>) return _mm256_cmpeq_epi8_mask(a, b);
				else if constexpr (std::is_same_v<S, uint8_t>) return _mm256_cmpeq_epu8_mask(a, b);
			}
			else if constexpr (inRange(sizeof(T), 0, 16))
			{
				if constexpr (std::is_same_v<S, double>) return _mm_cmp_pd_mask(a, b, _CMP_EQ_OQ);
				else if constexpr (std::is_same_v<S, float>) return _mm_cmp_ps_mask(a, b, _CMP_EQ_OQ);
				else if constexpr (std::is_same_v<S, int64_t>) return _mm_cmpeq_epi64_mask(a, b);
				else if constexpr (std::is_same_v<S, uint64_t>) return _mm_cmpeq_epu64_mask(a, b);
				else if constexpr (std::is_same_v<S, int32_t>) return _mm_cmpeq_epi32_mask(a, b);
				else if constexpr (std::is_same_v<S, uint32_t>) return _mm_cmpeq_epu32_mask(a, b);
				else if constexpr (std::is_same_v<S, int16_t>) return _mm_cmpeq_epi16_mask(a, b);
				else if constexpr (std::is_same_v<S, uint16_t>) return _mm_cmpeq_epu16_mask(a, b);
				else if constexpr (std::is_same_v<S, int8_t>) return _mm_cmpeq_epi8_mask(a, b);
				else if constexpr (std::is_same_v<S, uint8_t>) return _mm_cmpeq_epu8_mask(a, b);
			}
		}
		else
		{
			typename SIMD_Vector<S, N>::MaskType ret = 0;
			for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] == b[i]);
			return ret;
		}
	}
	template<typename S, size_t N>
		requires (N <= 64)
	__forceinline typename SIMD_Vector<S, N>::MaskType cmp_not_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using T = SIMD_Vector<S, N>;
		if constexpr (capabilities::current == capabilities::zen4)
		{
			if constexpr (sizeof(T) > 64) return concat_masks(cmp_not_equal(a.lo, b.lo), cmp_not_equal(a.hi, b.hi));
			else if constexpr (inRange(sizeof(T), 33, 64))
			{
				if constexpr (std::is_same_v<S, double>) return _mm512_cmp_pd_mask(a, b, _CMP_NEQ_OQ);
				else if constexpr (std::is_same_v<S, float>) return _mm512_cmp_ps_mask(a, b, _CMP_NEQ_OQ);
				else if constexpr (std::is_same_v<S, int64_t>) return _mm512_cmpneq_epi64_mask(a, b);
				else if constexpr (std::is_same_v<S, uint64_t>) return _mm512_cmpneq_epu64_mask(a, b);
				else if constexpr (std::is_same_v<S, int32_t>) return _mm512_cmpneq_epi32_mask(a, b);
				else if constexpr (std::is_same_v<S, uint32_t>) return _mm512_cmpneq_epu32_mask(a, b);
				else if constexpr (std::is_same_v<S, int16_t>) return _mm512_cmpneq_epi16_mask(a, b);
				else if constexpr (std::is_same_v<S, uint16_t>) return _mm512_cmpneq_epu16_mask(a, b);
				else if constexpr (std::is_same_v<S, int8_t>) return _mm512_cmpneq_epi8_mask(a, b);
				else if constexpr (std::is_same_v<S, uint8_t>) return _mm512_cmpneq_epu8_mask(a, b);
			}
			else if constexpr (inRange(sizeof(T), 17, 32))
			{
				if constexpr (std::is_same_v<S, double>) return _mm256_cmp_pd_mask(a, b, _CMP_NEQ_OQ);
				else if constexpr (std::is_same_v<S, float>) return _mm256_cmp_ps_mask(a, b, _CMP_NEQ_OQ);
				else if constexpr (std::is_same_v<S, int64_t>) return _mm256_cmpneq_epi64_mask(a, b);
				else if constexpr (std::is_same_v<S, uint64_t>) return _mm256_cmpneq_epu64_mask(a, b);
				else if constexpr (std::is_same_v<S, int32_t>) return _mm256_cmpneq_epi32_mask(a, b);
				else if constexpr (std::is_same_v<S, uint32_t>) return _mm256_cmpneq_epu32_mask(a, b);
				else if constexpr (std::is_same_v<S, int16_t>) return _mm256_cmpneq_epi16_mask(a, b);
				else if constexpr (std::is_same_v<S, uint16_t>) return _mm256_cmpneq_epu16_mask(a, b);
				else if constexpr (std::is_same_v<S, int8_t>) return _mm256_cmpneq_epi8_mask(a, b);
				else if constexpr (std::is_same_v<S, uint8_t>) return _mm256_cmpneq_epu8_mask(a, b);
			}
			else if constexpr (inRange(sizeof(T), 0, 16))
			{
				if constexpr (std::is_same_v<S, double>) return _mm_cmp_pd_mask(a, b, _CMP_NEQ_OQ);
				else if constexpr (std::is_same_v<S, float>) return _mm_cmp_ps_mask(a, b, _CMP_NEQ_OQ);
				else if constexpr (std::is_same_v<S, int64_t>) return _mm_cmpneq_epi64_mask(a, b);
				else if constexpr (std::is_same_v<S, uint64_t>) return _mm_cmpneq_epu64_mask(a, b);
				else if constexpr (std::is_same_v<S, int32_t>) return _mm_cmpneq_epi32_mask(a, b);
				else if constexpr (std::is_same_v<S, uint32_t>) return _mm_cmpneq_epu32_mask(a, b);
				else if constexpr (std::is_same_v<S, int16_t>) return _mm_cmpneq_epi16_mask(a, b);
				else if constexpr (std::is_same_v<S, uint16_t>) return _mm_cmpneq_epu16_mask(a, b);
				else if constexpr (std::is_same_v<S, int8_t>) return _mm_cmpneq_epi8_mask(a, b);
				else if constexpr (std::is_same_v<S, uint8_t>) return _mm_cmpneq_epu8_mask(a, b);
			}
		}
		else
		{
			typename SIMD_Vector<S, N>::MaskType ret = 0;
			for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] != b[i]);
			return ret;
		}
	}
	template<typename S, size_t N>
		requires (N <= 64)
	__forceinline typename SIMD_Vector<S, N>::MaskType cmp_greater(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using T = SIMD_Vector<S, N>;
		if constexpr (capabilities::current == capabilities::zen4)
		{
			if constexpr (sizeof(T) > 64) return concat_masks(cmp_greater(a.lo, b.lo), cmp_greater(a.hi, b.hi));
			else if constexpr (inRange(sizeof(T), 33, 64))
			{
				if constexpr (std::is_same_v<S, double>) return _mm512_cmp_pd_mask(a, b, _CMP_GT_OQ);
				else if constexpr (std::is_same_v<S, float>) return _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ);
				else if constexpr (std::is_same_v<S, int64_t>) return _mm512_cmpgt_epi64_mask(a, b);
				else if constexpr (std::is_same_v<S, uint64_t>) return _mm512_cmpgt_epu64_mask(a, b);
				else if constexpr (std::is_same_v<S, int32_t>) return _mm512_cmpgt_epi32_mask(a, b);
				else if constexpr (std::is_same_v<S, uint32_t>) return _mm512_cmpgt_epu32_mask(a, b);
				else if constexpr (std::is_same_v<S, int16_t>) return _mm512_cmpgt_epi16_mask(a, b);
				else if constexpr (std::is_same_v<S, uint16_t>) return _mm512_cmpgt_epu16_mask(a, b);
				else if constexpr (std::is_same_v<S, int8_t>) return _mm512_cmpgt_epi8_mask(a, b);
				else if constexpr (std::is_same_v<S, uint8_t>) return _mm512_cmpgt_epu8_mask(a, b);
			}
			else if constexpr (inRange(sizeof(T), 17, 32))
			{
				if constexpr (std::is_same_v<S, double>) return _mm256_cmp_pd_mask(a, b, _CMP_GT_OQ);
				else if constexpr (std::is_same_v<S, float>) return _mm256_cmp_ps_mask(a, b, _CMP_GT_OQ);
				else if constexpr (std::is_same_v<S, int64_t>) return _mm256_cmpgt_epi64_mask(a, b);
				else if constexpr (std::is_same_v<S, uint64_t>) return _mm256_cmpgt_epu64_mask(a, b);
				else if constexpr (std::is_same_v<S, int32_t>) return _mm256_cmpgt_epi32_mask(a, b);
				else if constexpr (std::is_same_v<S, uint32_t>) return _mm256_cmpgt_epu32_mask(a, b);
				else if constexpr (std::is_same_v<S, int16_t>) return _mm256_cmpgt_epi16_mask(a, b);
				else if constexpr (std::is_same_v<S, uint16_t>) return _mm256_cmpgt_epu16_mask(a, b);
				else if constexpr (std::is_same_v<S, int8_t>) return _mm256_cmpgt_epi8_mask(a, b);
				else if constexpr (std::is_same_v<S, uint8_t>) return _mm256_cmpgt_epu8_mask(a, b);
			}
			else if constexpr (inRange(sizeof(T), 0, 16))
			{
				if constexpr (std::is_same_v<S, double>) return _mm_cmp_pd_mask(a, b, _CMP_GT_OQ);
				else if constexpr (std::is_same_v<S, float>) return _mm_cmp_ps_mask(a, b, _CMP_GT_OQ);
				else if constexpr (std::is_same_v<S, int64_t>) return _mm_cmpgt_epi64_mask(a, b);
				else if constexpr (std::is_same_v<S, uint64_t>) return _mm_cmpgt_epu64_mask(a, b);
				else if constexpr (std::is_same_v<S, int32_t>) return _mm_cmpgt_epi32_mask(a, b);
				else if constexpr (std::is_same_v<S, uint32_t>) return _mm_cmpgt_epu32_mask(a, b);
				else if constexpr (std::is_same_v<S, int16_t>) return _mm_cmpgt_epi16_mask(a, b);
				else if constexpr (std::is_same_v<S, uint16_t>) return _mm_cmpgt_epu16_mask(a, b);
				else if constexpr (std::is_same_v<S, int8_t>) return _mm_cmpgt_epi8_mask(a, b);
				else if constexpr (std::is_same_v<S, uint8_t>) return _mm_cmpgt_epu8_mask(a, b);
			}
		}
		else
		{
			typename SIMD_Vector<S, N>::MaskType ret = 0;
			for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] > b[i]);
			return ret;
		}
	}
	template<typename S, size_t N>
		requires (N <= 64)
	__forceinline typename SIMD_Vector<S, N>::MaskType cmp_greater_or_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using T = SIMD_Vector<S, N>;
		if constexpr (capabilities::current == capabilities::zen4)
		{
			if constexpr (sizeof(T) > 64) return concat_masks(cmp_greater_or_equal(a.lo, b.lo), cmp_greater_or_equal(a.hi, b.hi));
			else if constexpr (inRange(sizeof(T), 33, 64))
			{
				if constexpr (std::is_same_v<S, double>) return _mm512_cmp_pd_mask(a, b, _CMP_GE_OQ);
				else if constexpr (std::is_same_v<S, float>) return _mm512_cmp_ps_mask(a, b, _CMP_GE_OQ);
				else if constexpr (std::is_same_v<S, int64_t>) return _mm512_cmpge_epi64_mask(a, b);
				else if constexpr (std::is_same_v<S, uint64_t>) return _mm512_cmpge_epu64_mask(a, b);
				else if constexpr (std::is_same_v<S, int32_t>) return _mm512_cmpge_epi32_mask(a, b);
				else if constexpr (std::is_same_v<S, uint32_t>) return _mm512_cmpge_epu32_mask(a, b);
				else if constexpr (std::is_same_v<S, int16_t>) return _mm512_cmpge_epi16_mask(a, b);
				else if constexpr (std::is_same_v<S, uint16_t>) return _mm512_cmpge_epu16_mask(a, b);
				else if constexpr (std::is_same_v<S, int8_t>) return _mm512_cmpge_epi8_mask(a, b);
				else if constexpr (std::is_same_v<S, uint8_t>) return _mm512_cmpge_epu8_mask(a, b);
			}
			else if constexpr (inRange(sizeof(T), 17, 32))
			{
				if constexpr (std::is_same_v<S, double>) return _mm256_cmp_pd_mask(a, b, _CMP_GE_OQ);
				else if constexpr (std::is_same_v<S, float>) return _mm256_cmp_ps_mask(a, b, _CMP_GE_OQ);
				else if constexpr (std::is_same_v<S, int64_t>) return _mm256_cmpge_epi64_mask(a, b);
				else if constexpr (std::is_same_v<S, uint64_t>) return _mm256_cmpge_epu64_mask(a, b);
				else if constexpr (std::is_same_v<S, int32_t>) return _mm256_cmpge_epi32_mask(a, b);
				else if constexpr (std::is_same_v<S, uint32_t>) return _mm256_cmpge_epu32_mask(a, b);
				else if constexpr (std::is_same_v<S, int16_t>) return _mm256_cmpge_epi16_mask(a, b);
				else if constexpr (std::is_same_v<S, uint16_t>) return _mm256_cmpge_epu16_mask(a, b);
				else if constexpr (std::is_same_v<S, int8_t>) return _mm256_cmpge_epi8_mask(a, b);
				else if constexpr (std::is_same_v<S, uint8_t>) return _mm256_cmpge_epu8_mask(a, b);
			}
			else if constexpr (inRange(sizeof(T), 0, 16))
			{
				if constexpr (std::is_same_v<S, double>) return _mm_cmp_pd_mask(a, b, _CMP_GE_OQ);
				else if constexpr (std::is_same_v<S, float>) return _mm_cmp_ps_mask(a, b, _CMP_GE_OQ);
				else if constexpr (std::is_same_v<S, int64_t>) return _mm_cmpge_epi64_mask(a, b);
				else if constexpr (std::is_same_v<S, uint64_t>) return _mm_cmpge_epu64_mask(a, b);
				else if constexpr (std::is_same_v<S, int32_t>) return _mm_cmpge_epi32_mask(a, b);
				else if constexpr (std::is_same_v<S, uint32_t>) return _mm_cmpge_epu32_mask(a, b);
				else if constexpr (std::is_same_v<S, int16_t>) return _mm_cmpge_epi16_mask(a, b);
				else if constexpr (std::is_same_v<S, uint16_t>) return _mm_cmpge_epu16_mask(a, b);
				else if constexpr (std::is_same_v<S, int8_t>) return _mm_cmpge_epi8_mask(a, b);
				else if constexpr (std::is_same_v<S, uint8_t>) return _mm_cmpge_epu8_mask(a, b);
			}
		}
		else
		{
			typename SIMD_Vector<S, N>::MaskType ret = 0;
			for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] >= b[i]);
			return ret;
		}
	}
	template<typename S, size_t N>
		requires (N <= 64)
	__forceinline SIMD_Vector<S, N> mask2vec(const SIMD_Mask<N>& mask)
	{
		constexpr size_t RetSize = sizeof(S) * N;
		if constexpr (capabilities::current == capabilities::zen4)
		{
			if constexpr (RetSize > 64) return concat(mask2vec(mask.lo()), mask2vec(mask.hi()));
			else if constexpr (inRange(RetSize, 33, 64))
			{
				if constexpr (sizeof(S) == 8) return std::bit_cast<typename reg512<S>::type>(_mm512_movm_epi64(mask));
				if constexpr (sizeof(S) == 4) return std::bit_cast<typename reg512<S>::type>(_mm512_movm_epi32(mask));
				if constexpr (sizeof(S) == 2) return std::bit_cast<typename reg512<S>::type>(_mm512_movm_epi16(mask));
				if constexpr (sizeof(S) == 1) return std::bit_cast<typename reg512<S>::type>(_mm512_movm_epi8(mask));
			}
			else if constexpr (inRange(RetSize, 17, 32))
			{
				if constexpr (sizeof(S) == 8) return std::bit_cast<typename reg256<S>::type>(_mm256_movm_epi64(mask));
				if constexpr (sizeof(S) == 4) return std::bit_cast<typename reg256<S>::type>(_mm256_movm_epi32(mask));
				if constexpr (sizeof(S) == 2) return std::bit_cast<typename reg256<S>::type>(_mm256_movm_epi16(mask));
				if constexpr (sizeof(S) == 1) return std::bit_cast<typename reg256<S>::type>(_mm256_movm_epi8(mask));
			}
			else if constexpr (inRange(RetSize, 0, 16))
			{
				if constexpr (sizeof(S) == 8) return std::bit_cast<typename reg128<S>::type>(_mm_movm_epi64(mask));
				if constexpr (sizeof(S) == 4) return std::bit_cast<typename reg128<S>::type>(_mm_movm_epi32(mask));
				if constexpr (sizeof(S) == 2) return std::bit_cast<typename reg128<S>::type>(_mm_movm_epi16(mask));
				if constexpr (sizeof(S) == 1) return std::bit_cast<typename reg128<S>::type>(_mm_movm_epi8(mask));
			}
		}
		else
		{
			SIMD_Vector<S, N> ret;
			mask.explode<S>(ret.arr.data(), std::bit_cast<S>(~typename SIMD_Vector<S, N>::IntScalarType(0)), std::bit_cast<S>(0));
			return ret;
		}
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> abs(const SIMD_Vector<S, N>& a)
	{
		if constexpr (std::is_unsigned_v<S>) return a;
		using T = SIMD_Vector<S, N>;
		if constexpr (capabilities::current == capabilities::zen4)
		{
			if constexpr (sizeof(T) > 64) return concat(abs(a.lo), abs(a.hi));
			else if constexpr (inRange(sizeof(T), 33, 64))
			{
				if constexpr (std::is_same_v<S, double>) return _mm512_abs_pd(a);
				if constexpr (std::is_same_v<S, float>) return _mm512_abs_ps(a);
				if constexpr (std::is_same_v<S, int64_t>) return _mm512_abs_epi64(a);
				if constexpr (std::is_same_v<S, int32_t>) return _mm512_abs_epi32(a);
				if constexpr (std::is_same_v<S, int16_t>) return _mm512_abs_epi16(a);
				if constexpr (std::is_same_v<S, int8_t>) return _mm512_abs_epi8(a);
			}
			else if constexpr (inRange(sizeof(T), 17, 32))
			{
				//no abs for pd and ps, but there is for ints, lol. Cut off sign bit manually
				if constexpr (std::is_same_v<S, double>) return a & std::bit_cast<double>(0x7FFFFFFFFFFFFFFF);
				if constexpr (std::is_same_v<S, float>) return a & std::bit_cast<float>(0x7FFFFFFF);
				if constexpr (std::is_same_v<S, int64_t>) return _mm256_abs_epi64(a);
				if constexpr (std::is_same_v<S, int32_t>) return _mm256_abs_epi32(a);
				if constexpr (std::is_same_v<S, int16_t>) return _mm256_abs_epi16(a);
				if constexpr (std::is_same_v<S, int8_t>) return _mm256_abs_epi8(a);
			}
			else if constexpr (inRange(sizeof(T), 17, 32))
			{
				//no abs for pd and ps, but there is for ints, lol. Cut off sign bit manually
				if constexpr (std::is_same_v<S, double>) return a & std::bit_cast<double>(0x7FFFFFFFFFFFFFFF);
				if constexpr (std::is_same_v<S, float>) return a & std::bit_cast<float>(0x7FFFFFFF);
				if constexpr (std::is_same_v<S, int64_t>) return _mm_abs_epi64(a);
				if constexpr (std::is_same_v<S, int32_t>) return _mm_abs_epi32(a);
				if constexpr (std::is_same_v<S, int16_t>) return _mm_abs_epi16(a);
				if constexpr (std::is_same_v<S, int8_t>) return _mm_abs_epi8(a);
			}
		}
		else
		{
			T ret;
			for (size_t i = 0; i < N; ++i) ret[i] = std::abs(a[i]);
			return ret;
		}
	}

	template<typename S, size_t N>
		requires (std::is_floating_point_v<S>)
	__forceinline SIMD_Vector<S, N> floor(const SIMD_Vector<S, N>& a)
	{
		using T = SIMD_Vector<S, N>;
		if constexpr (capabilities::current == capabilities::zen4)
		{
			if constexpr (sizeof(T) > 64) return concat(floor(a.lo), floor(a.hi));
			else if constexpr (inRange(sizeof(T), 33, 64))
			{
				if constexpr (std::is_same_v<S, double>) return _mm512_floor_pd(a);
				if constexpr (std::is_same_v<S, float>) return _mm512_floor_ps(a);
			}
			else if constexpr (inRange(sizeof(T), 17, 32))
			{
				if constexpr (std::is_same_v<S, double>) return _mm256_floor_pd(a);
				if constexpr (std::is_same_v<S, float>) return _mm256_floor_ps(a);
			}
			else if constexpr (inRange(sizeof(T), 0, 16))
			{
				if constexpr (std::is_same_v<S, double>) return _mm_floor_pd(a);
				if constexpr (std::is_same_v<S, float>) return _mm_floor_ps(a);
			}
		}
		else
		{
			T ret;
			for (size_t i = 0; i < N; ++i) ret[i] = std::floor(a[i]);
			return ret;
		}
	}

	template<typename S, size_t N>
		requires (std::is_floating_point_v<S>)
	__forceinline SIMD_Vector<S, N> ceil(const SIMD_Vector<S, N>& a)
	{
		using T = SIMD_Vector<S, N>;
		if constexpr (capabilities::current == capabilities::zen4)
		{
			if constexpr (sizeof(T) > 64) return concat(ceil(a.lo), ceil(a.hi));
			else if constexpr (inRange(sizeof(T), 33, 64))
			{
				if constexpr (std::is_same_v<S, double>) return _mm512_ceil_pd(a);
				if constexpr (std::is_same_v<S, float>) return _mm512_ceil_ps(a);
			}
			else if constexpr (inRange(sizeof(T), 17, 32))
			{
				if constexpr (std::is_same_v<S, double>) return _mm256_ceil_pd(a);
				if constexpr (std::is_same_v<S, float>) return _mm256_ceil_ps(a);
			}
			else if constexpr (inRange(sizeof(T), 0, 16))
			{
				if constexpr (std::is_same_v<S, double>) return _mm_ceil_pd(a);
				if constexpr (std::is_same_v<S, float>) return _mm_ceil_ps(a);
			}
		}
		else
		{
			T ret;
			for (size_t i = 0; i < N; ++i) ret[i] = std::ceil(a[i]);
			return ret;
		}
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> min(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using T = SIMD_Vector<S, N>;
		if constexpr (capabilities::current == capabilities::zen4)
		{
			if constexpr (sizeof(T) > 64) return concat(min(a.lo, b.lo), min(a.hi, b.hi));
			else if constexpr (inRange(sizeof(T), 33, 64))
			{
				if constexpr (std::is_same_v<S, double>) return _mm512_min_pd(a, b);
				if constexpr (std::is_same_v<S, float>) return _mm512_min_ps(a, b);
				if constexpr (std::is_same_v<S, uint64_t>) return _mm512_min_epu64(a, b);
				if constexpr (std::is_same_v<S, uint32_t>) return _mm512_min_epu32(a, b);
				if constexpr (std::is_same_v<S, uint16_t>) return _mm512_min_epu16(a, b);
				if constexpr (std::is_same_v<S, uint8_t>) return _mm512_min_epu8(a, b);
				if constexpr (std::is_same_v<S, int64_t>) return _mm512_min_epi64(a, b);
				if constexpr (std::is_same_v<S, int32_t>) return _mm512_min_epi32(a, b);
				if constexpr (std::is_same_v<S, int16_t>) return _mm512_min_epi16(a, b);
				if constexpr (std::is_same_v<S, int8_t>) return _mm512_min_epi8(a, b);
			}
			else if constexpr (inRange(sizeof(T), 17, 32))
			{
				if constexpr (std::is_same_v<S, double>) return _mm256_min_pd(a, b);
				if constexpr (std::is_same_v<S, float>) return _mm256_min_ps(a, b);
				if constexpr (std::is_same_v<S, uint64_t>) return _mm256_min_epu64(a, b);
				if constexpr (std::is_same_v<S, uint32_t>) return _mm256_min_epu32(a, b);
				if constexpr (std::is_same_v<S, uint16_t>) return _mm256_min_epu16(a, b);
				if constexpr (std::is_same_v<S, uint8_t>) return _mm256_min_epu8(a, b);
				if constexpr (std::is_same_v<S, int64_t>) return _mm256_min_epi64(a, b);
				if constexpr (std::is_same_v<S, int32_t>) return _mm256_min_epi32(a, b);
				if constexpr (std::is_same_v<S, int16_t>) return _mm256_min_epi16(a, b);
				if constexpr (std::is_same_v<S, int8_t>) return _mm256_min_epi8(a, b);
			}
			else if constexpr (inRange(sizeof(T), 0, 16))
			{
				if constexpr (std::is_same_v<S, double>) return _mm_min_pd(a, b);
				if constexpr (std::is_same_v<S, float>) return _mm_min_ps(a, b);
				if constexpr (std::is_same_v<S, uint64_t>) return _mm_min_epu64(a, b);
				if constexpr (std::is_same_v<S, uint32_t>) return _mm_min_epu32(a, b);
				if constexpr (std::is_same_v<S, uint16_t>) return _mm_min_epu16(a, b);
				if constexpr (std::is_same_v<S, uint8_t>) return _mm_min_epu8(a, b);
				if constexpr (std::is_same_v<S, int64_t>) return _mm_min_epi64(a, b);
				if constexpr (std::is_same_v<S, int32_t>) return _mm_min_epi32(a, b);
				if constexpr (std::is_same_v<S, int16_t>) return _mm_min_epi16(a, b);
				if constexpr (std::is_same_v<S, int8_t>) return _mm_min_epi8(a, b);
			}
		}
		else
		{
			T ret;
			for (size_t i = 0; i < N; ++i) ret[i] = std::min(a[i], b[i]);
			return ret;
		}
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> max(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using T = SIMD_Vector<S, N>;
		if constexpr (capabilities::current == capabilities::zen4)
		{
			if constexpr (sizeof(T) > 64) return concat(max(a.lo, b.lo), max(a.hi, b.hi));
			else if constexpr (inRange(sizeof(T), 33, 64))
			{
				if constexpr (std::is_same_v<S, double>) return _mm512_max_pd(a, b);
				if constexpr (std::is_same_v<S, float>) return _mm512_max_ps(a, b);
				if constexpr (std::is_same_v<S, uint64_t>) return _mm512_max_epu64(a, b);
				if constexpr (std::is_same_v<S, uint32_t>) return _mm512_max_epu32(a, b);
				if constexpr (std::is_same_v<S, uint16_t>) return _mm512_max_epu16(a, b);
				if constexpr (std::is_same_v<S, uint8_t>) return _mm512_max_epu8(a, b);
				if constexpr (std::is_same_v<S, int64_t>) return _mm512_max_epi64(a, b);
				if constexpr (std::is_same_v<S, int32_t>) return _mm512_max_epi32(a, b);
				if constexpr (std::is_same_v<S, int16_t>) return _mm512_max_epi16(a, b);
				if constexpr (std::is_same_v<S, int8_t>) return _mm512_max_epi8(a, b);
			}
			else if constexpr (inRange(sizeof(T), 17, 32))
			{
				if constexpr (std::is_same_v<S, double>) return _mm256_max_pd(a, b);
				if constexpr (std::is_same_v<S, float>) return _mm256_max_ps(a, b);
				if constexpr (std::is_same_v<S, uint64_t>) return _mm256_max_epu64(a, b);
				if constexpr (std::is_same_v<S, uint32_t>) return _mm256_max_epu32(a, b);
				if constexpr (std::is_same_v<S, uint16_t>) return _mm256_max_epu16(a, b);
				if constexpr (std::is_same_v<S, uint8_t>) return _mm256_max_epu8(a, b);
				if constexpr (std::is_same_v<S, int64_t>) return _mm256_max_epi64(a, b);
				if constexpr (std::is_same_v<S, int32_t>) return _mm256_max_epi32(a, b);
				if constexpr (std::is_same_v<S, int16_t>) return _mm256_max_epi16(a, b);
				if constexpr (std::is_same_v<S, int8_t>) return _mm256_max_epi8(a, b);
			}
			else if constexpr (inRange(sizeof(T), 0, 16))
			{
				if constexpr (std::is_same_v<S, double>) return _mm_max_pd(a, b);
				if constexpr (std::is_same_v<S, float>) return _mm_max_ps(a, b);
				if constexpr (std::is_same_v<S, uint64_t>) return _mm_max_epu64(a, b);
				if constexpr (std::is_same_v<S, uint32_t>) return _mm_max_epu32(a, b);
				if constexpr (std::is_same_v<S, uint16_t>) return _mm_max_epu16(a, b);
				if constexpr (std::is_same_v<S, uint8_t>) return _mm_max_epu8(a, b);
				if constexpr (std::is_same_v<S, int64_t>) return _mm_max_epi64(a, b);
				if constexpr (std::is_same_v<S, int32_t>) return _mm_max_epi32(a, b);
				if constexpr (std::is_same_v<S, int16_t>) return _mm_max_epi16(a, b);
				if constexpr (std::is_same_v<S, int8_t>) return _mm_max_epi8(a, b);
			}
		}
		else
		{
			T ret;
			for (size_t i = 0; i < N; ++i) ret[i] = std::max(a[i], b[i]);
			return ret;
		}
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> clamp(const SIMD_Vector<S, N>& val, const SIMD_Vector<S, N>& min, const SIMD_Vector<S, N>& max)
	{
		auto clampedLow = AVXXY_NAMESPACE::max(val, min);
		return AVXXY_NAMESPACE::min(clampedLow, max);
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> unpacklo(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using T = SIMD_Vector<S, N>;
		if constexpr (sizeof(T) > 64) return concat(unpacklo(a.lo, b.lo), unpacklo(a.hi, b.hi)); //TODO: verify
		else if constexpr (inRange(sizeof(T), 33, 64))
		{
			if constexpr (std::is_same_v<S, double>) return _mm512_unpacklo_pd(a, b);
			if constexpr (std::is_same_v<S, float>) return _mm512_unpacklo_ps(a, b);
			if constexpr (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>) return _mm512_unpacklo_epi64(a, b);
			if constexpr (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>) return _mm512_unpacklo_epi32(a, b);
			if constexpr (std::is_same_v<S, int16_t> || std::is_same_v<S, uint16_t>) return _mm512_unpacklo_epi16(a, b);
			if constexpr (std::is_same_v<S, int8_t> || std::is_same_v<S, uint8_t>) return _mm512_unpacklo_epi8(a, b);
		}
		else if constexpr (inRange(sizeof(T), 17, 32))
		{
			if constexpr (std::is_same_v<S, double>) return _mm256_unpacklo_pd(a, b);
			if constexpr (std::is_same_v<S, float>) return _mm256_unpacklo_ps(a, b);
			if constexpr (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>) return _mm256_unpacklo_epi64(a, b);
			if constexpr (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>) return _mm256_unpacklo_epi32(a, b);
			if constexpr (std::is_same_v<S, int16_t> || std::is_same_v<S, uint16_t>) return _mm256_unpacklo_epi16(a, b);
			if constexpr (std::is_same_v<S, int8_t> || std::is_same_v<S, uint8_t>) return _mm256_unpacklo_epi8(a, b);
		}
		else if constexpr (inRange(sizeof(T), 0, 16))
		{
			if constexpr (std::is_same_v<S, double>) return _mm_unpacklo_pd(a, b);
			if constexpr (std::is_same_v<S, float>) return _mm_unpacklo_ps(a, b);
			if constexpr (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>) return _mm_unpacklo_epi64(a, b);
			if constexpr (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>) return _mm_unpacklo_epi32(a, b);
			if constexpr (std::is_same_v<S, int16_t> || std::is_same_v<S, uint16_t>) return _mm_unpacklo_epi16(a, b);
			if constexpr (std::is_same_v<S, int8_t> || std::is_same_v<S, uint8_t>) return _mm_unpacklo_epi8(a, b);
		}
		else static_assert(always_false_v<S>, "unpacklo");
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> unpackhi(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using T = SIMD_Vector<S, N>;
		if constexpr (sizeof(T) > 64) return concat(unpackhi(a.lo, b.lo), unpackhi(a.hi, b.hi)); //TODO: verify
		else if constexpr (inRange(sizeof(T), 33, 64))
		{
			if constexpr (std::is_same_v<S, double>) return _mm512_unpackhi_pd(a, b);
			if constexpr (std::is_same_v<S, float>) return _mm512_unpackhi_ps(a, b);
			if constexpr (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>) return _mm512_unpackhi_epi64(a, b);
			if constexpr (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>) return _mm512_unpackhi_epi32(a, b);
			if constexpr (std::is_same_v<S, int16_t> || std::is_same_v<S, uint16_t>) return _mm512_unpackhi_epi16(a, b);
			if constexpr (std::is_same_v<S, int8_t> || std::is_same_v<S, uint8_t>) return _mm512_unpackhi_epi8(a, b);
		}
		else if constexpr (inRange(sizeof(T), 17, 32))
		{
			if constexpr (std::is_same_v<S, double>) return _mm256_unpackhi_pd(a, b);
			if constexpr (std::is_same_v<S, float>) return _mm256_unpackhi_ps(a, b);
			if constexpr (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>) return _mm256_unpackhi_epi64(a, b);
			if constexpr (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>) return _mm256_unpackhi_epi32(a, b);
			if constexpr (std::is_same_v<S, int16_t> || std::is_same_v<S, uint16_t>) return _mm256_unpackhi_epi16(a, b);
			if constexpr (std::is_same_v<S, int8_t> || std::is_same_v<S, uint8_t>) return _mm256_unpackhi_epi8(a, b);
		}
		else if constexpr (inRange(sizeof(T), 0, 16))
		{
			if constexpr (std::is_same_v<S, double>) return _mm_unpackhi_pd(a, b);
			if constexpr (std::is_same_v<S, float>) return _mm_unpackhi_ps(a, b);
			if constexpr (std::is_same_v<S, int64_t> || std::is_same_v<S, uint64_t>) return _mm_unpackhi_epi64(a, b);
			if constexpr (std::is_same_v<S, int32_t> || std::is_same_v<S, uint32_t>) return _mm_unpackhi_epi32(a, b);
			if constexpr (std::is_same_v<S, int16_t> || std::is_same_v<S, uint16_t>) return _mm_unpackhi_epi16(a, b);
			if constexpr (std::is_same_v<S, int8_t> || std::is_same_v<S, uint8_t>) return _mm_unpackhi_epi8(a, b);
		}
		else static_assert(always_false_v<S>, "unpackhi");
	}

	template<typename S, size_t N>
		requires (N <= 64)
	__forceinline typename SIMD_Vector<S, N>::MaskType cmp_less(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using T = SIMD_Vector<S, N>;
		if constexpr (sizeof(T) > 64) return concat_masks(cmp_less(a.lo, b.lo), cmp_less(a.hi, b.hi));
		else if constexpr (inRange(sizeof(T), 33, 64))
		{
			if constexpr (std::is_same_v<S, double>) return _mm512_cmp_pd_mask(a, b, _CMP_LT_OQ);
			else if constexpr (std::is_same_v<S, float>) return _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ);
			else if constexpr (std::is_same_v<S, int64_t>) return _mm512_cmplt_epi64_mask(a, b);
			else if constexpr (std::is_same_v<S, uint64_t>) return _mm512_cmplt_epu64_mask(a, b);
			else if constexpr (std::is_same_v<S, int32_t>) return _mm512_cmplt_epi32_mask(a, b);
			else if constexpr (std::is_same_v<S, uint32_t>) return _mm512_cmplt_epu32_mask(a, b);
			else if constexpr (std::is_same_v<S, int16_t>) return _mm512_cmplt_epi16_mask(a, b);
			else if constexpr (std::is_same_v<S, uint16_t>) return _mm512_cmplt_epu16_mask(a, b);
			else if constexpr (std::is_same_v<S, int8_t>) return _mm512_cmplt_epi8_mask(a, b);
			else if constexpr (std::is_same_v<S, uint8_t>) return _mm512_cmplt_epu8_mask(a, b);
		}
		else if constexpr (inRange(sizeof(T), 17, 32))
		{
			if constexpr (std::is_same_v<S, double>) return _mm256_cmp_pd_mask(a, b, _CMP_LT_OQ);
			else if constexpr (std::is_same_v<S, float>) return _mm256_cmp_ps_mask(a, b, _CMP_LT_OQ);
			else if constexpr (std::is_same_v<S, int64_t>) return _mm256_cmplt_epi64_mask(a, b);
			else if constexpr (std::is_same_v<S, uint64_t>) return _mm256_cmplt_epu64_mask(a, b);
			else if constexpr (std::is_same_v<S, int32_t>) return _mm256_cmplt_epi32_mask(a, b);
			else if constexpr (std::is_same_v<S, uint32_t>) return _mm256_cmplt_epu32_mask(a, b);
			else if constexpr (std::is_same_v<S, int16_t>) return _mm256_cmplt_epi16_mask(a, b);
			else if constexpr (std::is_same_v<S, uint16_t>) return _mm256_cmplt_epu16_mask(a, b);
			else if constexpr (std::is_same_v<S, int8_t>) return _mm256_cmplt_epi8_mask(a, b);
			else if constexpr (std::is_same_v<S, uint8_t>) return _mm256_cmplt_epu8_mask(a, b);
		}
		else if constexpr (inRange(sizeof(T), 0, 16))
		{
			if constexpr (std::is_same_v<S, double>) return _mm_cmp_pd_mask(a, b, _CMP_LT_OQ);
			else if constexpr (std::is_same_v<S, float>) return _mm_cmp_ps_mask(a, b, _CMP_LT_OQ);
			else if constexpr (std::is_same_v<S, int64_t>) return _mm_cmplt_epi64_mask(a, b);
			else if constexpr (std::is_same_v<S, uint64_t>) return _mm_cmplt_epu64_mask(a, b);
			else if constexpr (std::is_same_v<S, int32_t>) return _mm_cmplt_epi32_mask(a, b);
			else if constexpr (std::is_same_v<S, uint32_t>) return _mm_cmplt_epu32_mask(a, b);
			else if constexpr (std::is_same_v<S, int16_t>) return _mm_cmplt_epi16_mask(a, b);
			else if constexpr (std::is_same_v<S, uint16_t>) return _mm_cmplt_epu16_mask(a, b);
			else if constexpr (std::is_same_v<S, int8_t>) return _mm_cmplt_epi8_mask(a, b);
			else if constexpr (std::is_same_v<S, uint8_t>) return _mm_cmplt_epu8_mask(a, b);
		}
		else static_assert(always_false_v<S>, "cmplt");
	}

	template<typename S, size_t N>
		requires (N <= 64)
	__forceinline typename SIMD_Vector<S, N>::MaskType cmp_less_or_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using T = SIMD_Vector<S, N>;
		if constexpr (sizeof(T) > 64) return concat_masks(cmp_less_or_equal(a.lo, b.lo), cmp_less_or_equal(a.hi, b.hi));
		else if constexpr (inRange(sizeof(T), 33, 64))
		{
			if constexpr (std::is_same_v<S, double>) return _mm512_cmp_pd_mask(a, b, _CMP_LE_OQ);
			else if constexpr (std::is_same_v<S, float>) return _mm512_cmp_ps_mask(a, b, _CMP_LE_OQ);
			else if constexpr (std::is_same_v<S, int64_t>) return _mm512_cmple_epi64_mask(a, b);
			else if constexpr (std::is_same_v<S, uint64_t>) return _mm512_cmple_epu64_mask(a, b);
			else if constexpr (std::is_same_v<S, int32_t>) return _mm512_cmple_epi32_mask(a, b);
			else if constexpr (std::is_same_v<S, uint32_t>) return _mm512_cmple_epu32_mask(a, b);
			else if constexpr (std::is_same_v<S, int16_t>) return _mm512_cmple_epi16_mask(a, b);
			else if constexpr (std::is_same_v<S, uint16_t>) return _mm512_cmple_epu16_mask(a, b);
			else if constexpr (std::is_same_v<S, int8_t>) return _mm512_cmple_epi8_mask(a, b);
			else if constexpr (std::is_same_v<S, uint8_t>) return _mm512_cmple_epu8_mask(a, b);
		}
		else if constexpr (inRange(sizeof(T), 17, 32))
		{
			if constexpr (std::is_same_v<S, double>) return _mm256_cmp_pd_mask(a, b, _CMP_LE_OQ);
			else if constexpr (std::is_same_v<S, float>) return _mm256_cmp_ps_mask(a, b, _CMP_LE_OQ);
			else if constexpr (std::is_same_v<S, int64_t>) return _mm256_cmple_epi64_mask(a, b);
			else if constexpr (std::is_same_v<S, uint64_t>) return _mm256_cmple_epu64_mask(a, b);
			else if constexpr (std::is_same_v<S, int32_t>) return _mm256_cmple_epi32_mask(a, b);
			else if constexpr (std::is_same_v<S, uint32_t>) return _mm256_cmple_epu32_mask(a, b);
			else if constexpr (std::is_same_v<S, int16_t>) return _mm256_cmple_epi16_mask(a, b);
			else if constexpr (std::is_same_v<S, uint16_t>) return _mm256_cmple_epu16_mask(a, b);
			else if constexpr (std::is_same_v<S, int8_t>) return _mm256_cmple_epi8_mask(a, b);
			else if constexpr (std::is_same_v<S, uint8_t>) return _mm256_cmple_epu8_mask(a, b);
		}
		else if constexpr (inRange(sizeof(T), 0, 16))
		{
			if constexpr (std::is_same_v<S, double>) return _mm_cmp_pd_mask(a, b, _CMP_LE_OQ);
			else if constexpr (std::is_same_v<S, float>) return _mm_cmp_ps_mask(a, b, _CMP_LE_OQ);
			else if constexpr (std::is_same_v<S, int64_t>) return _mm_cmple_epi64_mask(a, b);
			else if constexpr (std::is_same_v<S, uint64_t>) return _mm_cmple_epu64_mask(a, b);
			else if constexpr (std::is_same_v<S, int32_t>) return _mm_cmple_epi32_mask(a, b);
			else if constexpr (std::is_same_v<S, uint32_t>) return _mm_cmple_epu32_mask(a, b);
			else if constexpr (std::is_same_v<S, int16_t>) return _mm_cmple_epi16_mask(a, b);
			else if constexpr (std::is_same_v<S, uint16_t>) return _mm_cmple_epu16_mask(a, b);
			else if constexpr (std::is_same_v<S, int8_t>) return _mm_cmple_epi8_mask(a, b);
			else if constexpr (std::is_same_v<S, uint8_t>) return _mm_cmple_epu8_mask(a, b);
		}
		else static_assert(always_false_v<S>, "cmple");
	}

	template<typename S, size_t N>
	std::ostream& operator<<(std::ostream& os, const SIMD_Vector<S, N>& v)
	{
		for (int i = 0; i < N; ++i)
		{
			os << ((std::is_integral_v<S> && sizeof(S) == 1) ? int(v[i]) : v[i]); //ITS INT8, not CHAR, DAMN IT!
			if (i < N - 1) os << " ";
		};
		return os;
	}
}