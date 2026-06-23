#pragma once
#include "../namespace.h"
#include <stdint.h>
#include <immintrin.h>
#include "../small_fp.h"
#include <bit>
#include "enums.h"

namespace AVXXY_NAMESPACE
{
	namespace meta
	{
		//template <typename T>
		//concept SupportsSizeClass = 

		template<class...> inline constexpr bool always_false_v = false;
		template <typename T, typename... Ts> inline constexpr bool is_any_of_v = (std::is_same_v<T, Ts> || ...);
		//Is this a supported scalar type? Any of these: signed/unsigned 8, 16, 32 and 64 bit ints, float, double, custom FP16 or BF16 type
		template<typename T> concept IsScalarType = is_any_of_v<T, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, float, double, fp16_t, bf16_t>;
		template<typename... Ts> concept AllAreScalarTypes = (IsScalarType<Ts> && ...);
		//Is this a valid intrinsic vector type? Does not check for actual availabilty (i.e. __m512 will pass this test even if AVX512 is not available)
		template<typename T> concept IsIntrinsicVector = is_any_of_v<T, __m128i, __m128, __m128d, __m128h, __m128bh, __m256i, __m256, __m256d, __m256h, __m256bh, __m512i, __m512, __m512d, __m512h, __m512bh>;


		//template <typename T>
		//struct ScalarTraits;

		template<meta::ScalarSizeClassEnum LS>
		struct ScalarSizeTraits
		{
			using IntT = std::conditional_t <LS == ScalarSizeClassEnum::byte, int8_t,
				std::conditional_t<LS == ScalarSizeClassEnum::word, int16_t,
				std::conditional_t<LS == ScalarSizeClassEnum::dword, int32_t, int64_t>>>;
			using UintT = std::conditional_t<LS == ScalarSizeClassEnum::byte, uint8_t,
				std::conditional_t<LS == ScalarSizeClassEnum::word, uint16_t,
				std::conditional_t<LS == ScalarSizeClassEnum::dword, uint32_t, uint64_t>>>;
			static constexpr UintT AllOnesUint = ~UintT(0);
			static constexpr UintT SignMask = UintT(1) << (sizeof(UintT) * 8 - 1);
			static constexpr ScalarSizeClassEnum size_class = LS;
			static constexpr size_t ByteSize = sizeof(UintT);
		};
		
		template<typename T> requires (IsScalarType<T>)
			inline constexpr ScalarSizeClassEnum scalar_size_class_v = []() {
			if constexpr (sizeof(T) == 1) return ScalarSizeClassEnum::byte;
			else if constexpr (sizeof(T) == 2) return ScalarSizeClassEnum::word;
			else if constexpr (sizeof(T) == 4) return ScalarSizeClassEnum::dword;
			else return ScalarSizeClassEnum::qword;
			}();

		template<typename S>
			requires IsScalarType<S>
		struct ScalarTraits : ScalarSizeTraits<scalar_size_class_v<S>>
		{
			static inline constexpr bool is_fp16 = std::is_same_v<S, fp16_t>;
			static inline constexpr bool is_bf16 = std::is_same_v<S, bf16_t>;
			static inline constexpr bool is_f32 = std::is_same_v<S, float>;
			static inline constexpr bool is_f64 = std::is_same_v<S, double>;
			static inline constexpr bool is_i64 = std::is_same_v<S, int64_t>;
			static inline constexpr bool is_i32 = std::is_same_v<S, int32_t>;
			static inline constexpr bool is_i16 = std::is_same_v<S, int16_t>;
			static inline constexpr bool is_i8 = std::is_same_v<S, int8_t>;
			static inline constexpr bool is_u64 = std::is_same_v<S, uint64_t>;
			static inline constexpr bool is_u32 = std::is_same_v<S, uint32_t>;
			static inline constexpr bool is_u16 = std::is_same_v<S, uint16_t>;
			static inline constexpr bool is_u8 = std::is_same_v<S, uint8_t>;

			//indicates wheteher this type is a signed 8 or 16 bit integer
			static inline constexpr bool is_small_sint = is_i16 || is_i8;
			//indicates wheteher this type is a unsigned 8 or 16 bit integer
			static inline constexpr bool is_small_uint = is_u16 || is_u8;
			//indicates wheteher this type is any 8 or 16 bit integer, signed or unsigned
			static inline constexpr bool any_small_int = is_small_sint || is_small_uint;

			//indicates wheteher this type is a floating point type (double, single, half precision or BF16)
			//Note that std::is_floating_point_v is not exactly equal to this, since FP16 and BF16 have limited support and are using custom types
			static inline constexpr bool any_float = is_any_of_v<S, float, double, fp16_t, bf16_t>;
			//indicates whether this type is 8 bit integer, signed or unsigned
			static inline constexpr bool any_i8 = (is_u8 || is_i8);
			//indicates whether this type is 16 bit integer, signed or unsigned
			static inline constexpr bool any_i16 = (is_u16 || is_i16);
			//indicates whether this type is 32 bit integer, signed or unsigned
			static inline constexpr bool any_i32 = (is_u32 || is_i32);
			//indicates whether this type is 64 bit integer, signed or unsigned
			static inline constexpr bool any_i64 = (is_u64 || is_i64);
			//indicates whether this type is integral
			static inline constexpr bool any_int = std::is_integral_v<S>;
			//indicates whether this type is not integral
			static inline constexpr bool not_int = !std::is_integral_v<S>;
		};

		//Returns true if this value is a power of 2.
		//0 and 1 are NOT considered powers of 2
		static constexpr bool isPowerOf2(size_t N)
		{
			return std::popcount(N) == 1 && N >= 2;
		}

		//Maps sizeInBytes to VectorSizeClassEnum member
		static constexpr VectorSizeClassEnum vector_size_class(size_t sizeInBytes)
		{
			if (sizeInBytes <= 16) return VectorSizeClassEnum::XMM;
			else if (sizeInBytes <= 32) return VectorSizeClassEnum::YMM;
			else if (sizeInBytes <= 64) return VectorSizeClassEnum::ZMM;
			else return VectorSizeClassEnum::XL;
		}

		template<typename T> //requires (SupportsSizeClass<T>) 
		inline constexpr VectorSizeClassEnum vector_size_class_v = vector_size_class(sizeof(T));

		//@note for now, bigger than 64 lanes vectors are not supported (mainly due to mask type not being ready for it)
		//@tparam S scalar type of the would-be vector
		//@tparam N lane count of the would-be vector
		template<typename S, size_t N> concept IsValid_SIMD_Vector = N >= 2 && N <= 64 && isPowerOf2(N) && IsScalarType<S>;

		template<typename T> inline constexpr bool xmm_sized = vector_size_class_v<T> == VectorSizeClassEnum::XMM;
		template<typename T> inline constexpr bool ymm_sized = vector_size_class_v<T> == VectorSizeClassEnum::YMM;
		template<typename T> inline constexpr bool zmm_sized = vector_size_class_v<T> == VectorSizeClassEnum::ZMM;

		constexpr bool is_xmm_size(size_t N) { return N <= 16; }
		constexpr bool is_ymm_size(size_t N) { return N > 16 && N <= 32; }
		constexpr bool is_zmm_size(size_t N) { return N > 32 && N <= 64; }

		template <typename T> requires (IsScalarType<T>) inline constexpr bool is_fp16 = std::is_same_v<T, fp16_t>;
		template <typename T> requires (IsScalarType<T>) inline constexpr bool is_bf16 = std::is_same_v<T, bf16_t>;
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

		//indicates wheteher this type is a signed 8 or 16 bit integer
		template <typename T> requires (IsScalarType<T>) inline constexpr bool is_small_sint = is_i16<T> || is_i8<T>;
		//indicates wheteher this type is a unsigned 8 or 16 bit integer
		template <typename T> requires (IsScalarType<T>) inline constexpr bool is_small_uint = is_u16<T> || is_u8<T>;
		//indicates wheteher this type is any 8 or 16 bit integer, signed or unsigned
		template <typename T> requires (IsScalarType<T>) inline constexpr bool any_small_int = is_small_sint<T> || is_small_uint<T>;

		//indicates wheteher this type is a floating point type (double, single, half precision or BF16)
		//Note that std::is_floating_point_v is not exactly equal to this, since FP16 and BF16 have limited support and are using custom types
		template <typename T> requires (IsScalarType<T>) inline constexpr bool any_float = is_any_of_v<T, float, double, fp16_t, bf16_t>;
		//indicates whether this type is 8 bit integer, signed or unsigned
		template <typename T> inline constexpr bool any_i8 = (is_u8<T> || is_i8<T>);
		//indicates whether this type is 16 bit integer, signed or unsigned
		template <typename T> inline constexpr bool any_i16 = (is_u16<T> || is_i16<T>);
		//indicates whether this type is 32 bit integer, signed or unsigned
		template <typename T> inline constexpr bool any_i32 = (is_u32<T> || is_i32<T>);
		//indicates whether this type is 64 bit integer, signed or unsigned
		template <typename T> inline constexpr bool any_i64 = (is_u64<T> || is_i64<T>);
		//indicates whether this type is integral
		template <typename T> requires (IsScalarType<T>) inline constexpr bool any_int = std::is_integral_v<T>;
		//indicates whether this type is not integralmore
		template <typename T> requires (IsScalarType<T>) inline constexpr bool not_int = !std::is_integral_v<T>;

		template <typename T> concept IsCvtOp = requires {typename T::cvt_to_t; };
		template <typename T> concept IsLoadOp = requires {T::_avxxy_is_load_tag; };
		template <typename T> concept IsGatherOp = requires {T::_avxxy_is_gather_tag; };
		template <typename T> concept IsScatterOp = requires {T::_avxxy_is_scatter_tag; };
		template <typename T> concept IsMovmOp = requires { T::_avxxy_is_movm_tag; };
	}
}