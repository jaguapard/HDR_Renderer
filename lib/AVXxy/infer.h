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

	template<class...>
	inline constexpr bool always_false_v = false;
}