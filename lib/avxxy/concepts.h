#pragma once
#include "namespace.h"
#include <type_traits>
#include "utils.h"
#include <immintrin.h>
#include <cstdint>
#include <cstddef>

namespace AVXXY_NAMESPACE
{
	namespace concepts
	{
		template <typename T, typename... Ts> inline constexpr bool is_any_of_v = (std::is_same_v<T, Ts> || ...);
		template<typename T> concept IsScalarType = is_any_of_v<T, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, float, double>;
		template<typename... Ts> concept AllAreScalarTypes = (IsScalarType<Ts> && ...);
		template<typename T> concept IsIntrinsicVector = is_any_of_v<T, __m128i, __m128, __m128d, __m256i, __m256, __m256d, __m512i, __m512, __m512d>;

		//indicates wheter the type is SIMD vector that fits only into zmm registers (33-64 bytes)
		template <typename T> inline constexpr bool zmm_sized = T::IsSimdVector && utils::inRange(sizeof(T), 33, 64);
		//indicates wheter the type is SIMD vector that fits only into ymm registers (17-32 bytes)
		template <typename T> inline constexpr bool ymm_sized = T::IsSimdVector && utils::inRange(sizeof(T), 17, 32);
		//indicates wheter the type is SIMD vector that fits only into xmm registers (less than or equal to 16 bytes)
		template <typename T> inline constexpr bool xmm_sized = T::IsSimdVector && utils::inRange(sizeof(T), 0, 16);

		template <typename T> requires (IsScalarType<T>) inline constexpr bool is_f32 = std::is_same_v<T, float>;
		template <typename T> requires (IsScalarType<T>) inline constexpr bool is_f64 = std::is_same_v<T, double>;
		template <typename T> requires (IsScalarType<T>) inline constexpr bool is_i64 = std::is_same_v<T, int64_t>;
		template <typename T> requires (IsScalarType<T>) inline constexpr bool is_i32 = std::is_same_v<T, int32_t>;
		template <typename T> requires (IsScalarType<T>) inline constexpr bool is_i16 = std::is_same_v<T, int16_t>;
		template <typename T> requires (IsScalarType<T>) inline constexpr bool is_i8 = std::is_same_v<T, int8_t>;
		template <typename T> requires (IsScalarType<T>) inline constexpr bool is_u64 = std::is_same_v<T, uint64_t>;
		template <typename T> requires (IsScalarType<T>) inline constexpr bool is_u32 = std::is_same_v<T, uint32_t>;
		template <typename T> requires (IsScalarType<T>) inline constexpr bool is_u16 = std::is_same_v<T, uint16_t>;
		template <typename T> requires (IsScalarType<T>) inline constexpr bool is_u8 = std::is_same_v<T, uint8_t>;
		//indicates wheter this type is 8 bit integer, signed or unsigned
		template <typename T> inline constexpr bool any_i8 = (is_u8<T> || is_i8<T>);
		//indicates wheter this type is 16 bit integer, signed or unsigned
		template <typename T> inline constexpr bool any_i16 = (is_u16<T> || is_i16<T>);
		//indicates wheter this type is 32 bit integer, signed or unsigned
		template <typename T> inline constexpr bool any_i32 = (is_u32<T> || is_i32<T>);
		//indicates wheter this type is 64 bit integer, signed or unsigned
		template <typename T> inline constexpr bool any_i64 = (is_u64<T> || is_i64<T>);
		//indicates wheter this type is integral
		template <typename T> requires (IsScalarType<T>) inline constexpr bool any_int = std::is_integral_v<T>;
		//indicates wheter this type is not integral (TODO: limit it only to doubles and floats, and maybe FP16/BF16?)
		template <typename T> requires (IsScalarType<T>) inline constexpr bool not_int = !std::is_integral_v<T>;

		template<typename T>
		struct same_size_uint_t
		{
			static_assert(sizeof(T) > 0 && sizeof(T) <= 8 && utils::isPowerOf2(sizeof(T)), "Unsupported size for same_size_uint_t");
			using type = std::conditional_t<sizeof(T) == 8, uint64_t,
				std::conditional_t<sizeof(T) == 4, uint32_t,
				std::conditional_t<sizeof(T) == 2, uint16_t, uint8_t>>>;
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

		template<class...>
		inline constexpr bool always_false_v = false;
	}
}