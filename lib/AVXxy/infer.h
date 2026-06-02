#pragma once
#include "namespace.h"
#include <stdint.h>
#include <type_traits>
#include <immintrin.h>
namespace AVXXY_NAMESPACE
{
	//Maps bit count to smallest signed integer type that has greater or equal number of bits
	template <size_t Size>
	struct bits_to_int_t
	{
		static_assert(Size <= 64, "Unsupported size for bits_to_int_t");
		using type =
			std::conditional_t<Size >= 33, int64_t,
			std::conditional_t<Size >= 17, int32_t,
			std::conditional_t<Size >= 9, int16_t, int8_t>>>;
	};

	//Maps bit count to smallest unsigned integer type that has greater or equal number of bits
	template <size_t Size>
	struct bits_to_uint_t
	{
		static_assert(Size <= 64, "Unsupported size for bits_to_uint_t");
		using type =
			std::conditional_t<Size >= 33, uint64_t,
			std::conditional_t<Size >= 17, uint32_t,
			std::conditional_t<Size >= 9, uint16_t, uint8_t>>>;
	};

	//Maps bit count to smallest float type that has greater or equal number of bits
	template <size_t Size>
	struct bits_to_float_t
	{
		static_assert(Size <= 64, "Unsupported size for bits_to_float_t");
		using type =
			std::conditional_t<Size >= 33, double, float>;
	};

	template<typename T>
	struct reg128
	{
		using type = std::conditional_t<std::is_integral_v<T>, __m128i,
			std::conditional_t<std::is_same_v<T, float>, __m128,
			std::conditional_t<std::is_same_v<T, double>, __m128d, void>>>;
	};
	template<typename T>
	struct reg256
	{
		using type = std::conditional_t<std::is_integral_v<T>, __m256i,
			std::conditional_t<std::is_same_v<T, float>, __m256,
			std::conditional_t<std::is_same_v<T, double>, __m256d, void>>>;
	};
	template<typename T>
	struct reg512
	{
		using type = std::conditional_t<std::is_integral_v<T>, __m512i,
			std::conditional_t<std::is_same_v<T, float>, __m512,
			std::conditional_t<std::is_same_v<T, double>, __m512d, void>>>;
	};

	//Smallest intrinsic vector type that can hold LaneCount values of ScalarType
	template<typename ScalarType, size_t LaneCount>
	struct native_reg_t
	{
		static inline constexpr size_t Bits = sizeof(ScalarType) * 8 * LaneCount;
		using type = std::conditional_t<Bits <= 128, typename reg128<ScalarType>::type, 
			std::conditional_t<Bits <= 256, typename reg256<ScalarType>::type, typename reg512<ScalarType>::type>>;
	};

	constexpr bool isPowerOf2(size_t N)
	{
		if (N <= 2) return true;
		if (N % 2 != 0) return false;
		return isPowerOf2(N / 2);
	}

	constexpr size_t aligned_size(size_t size, size_t alignment)
	{
		if (size % alignment == 0) return size;
		return ((size / alignment) + 1) * alignment;
	}

	constexpr size_t ceil_div(size_t numerator, size_t denominator)
	{
		if (numerator % denominator == 0) return numerator / denominator;
		return numerator / denominator + 1;
	}

	template<typename T, size_t count, size_t alignmentRequirement>
	constexpr size_t padded_element_count()
	{
		constexpr size_t sz = sizeof(T) * count;
		if (sz % alignmentRequirement == 0) return count;
		return (((sz / alignmentRequirement) + 1) * sz) / sizeof(alignmentRequirement);
	}

	constexpr bool inRange(size_t val, size_t min, size_t max)
	{
		return val >= min && val <= max;
	}

	template<class...>
	inline constexpr bool always_false_v = false;
}