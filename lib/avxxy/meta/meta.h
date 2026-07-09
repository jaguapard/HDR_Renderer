#pragma once
#include "../namespace.h"
#include <stdint.h>
#include <immintrin.h>
#include "../small_fp.h"
#include <bit>
#include "enums.h"
#include <array>
#include "../FeatureSet.h"
#include "../settings.h"

namespace AVXXY_NAMESPACE
{
	namespace meta
	{
		//template <typename T>
		//concept SupportsSizeClass = 

		template<class...> inline constexpr bool always_false_v = false;
		template <typename T, typename... Ts> inline constexpr bool is_any_of_v = (std::same_as<T, Ts> || ...);
		//Is this a supported scalar type? Any of these: signed/unsigned 8, 16, 32 and 64 bit ints, float, double, custom FP16 or BF16 type
		template<typename T> concept IsScalarType = is_any_of_v<T, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, float, double, fp16_t, bf16_t>;
		template<typename... Ts> concept AllAreScalarTypes = (IsScalarType<Ts> && ...);
		//Is this a valid intrinsic vector type? Does not check for actual availabilty (i.e. __m512 will pass this test even if AVX512 is not available)
		template<typename T> concept IsIntrinsicVector = is_any_of_v<T, __m128i, __m128, __m128d, __m128h, __m128bh, __m256i, __m256, __m256d, __m256h, __m256bh, __m512i, __m512, __m512d, __m512h, __m512bh>;

		//Constexpr variable of type T with all bits set to one
		template<typename T> requires std::is_trivially_copyable_v<T> constexpr inline T AllOnes = []() {
			std::array<uint8_t, sizeof(T)> arr;
			for (auto& it : arr) it = 0xFF;
			return std::bit_cast<T>(arr);
			}();
		//Constexpr variable of type T with all bits set to zero
		template<typename T> requires std::is_trivially_copyable_v<T> constexpr inline T AllZeros = []() {
			std::array<uint8_t, sizeof(T)> arr;
			for (auto& it : arr) it = 0;
			return std::bit_cast<T>(arr);
			}();

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
			static constexpr UintT AllZeroesUint = UintT(0);
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
			
		};

		template<typename T> requires IsScalarType<T> inline constexpr T UppermostBitMask = std::bit_cast<T>(ScalarTraits<T>::SignMask);

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

		template<typename S, size_t N>
		struct VectorTraits
		{
			static constexpr ScalarTraits<S> scalarTraits;
			static inline constexpr size_t ActiveSize = sizeof(S) * N;
			static inline constexpr std::array<S, N> AllZeroesArray = []() {
				std::array<S, N> ret; for (auto& it : ret) it = std::bit_cast<S>(scalarTraits.AllZeroesUint);
				return ret;
				}();
			static inline constexpr std::array<S, N> AllOnesArray = []() {
				std::array<S, N> ret; for (auto& it : ret) it = std::bit_cast<S>(scalarTraits.AllOnesUint);
				return ret;
				}();
			static inline constexpr VectorSizeClassEnum sizeClass = vector_size_class_v<std::array<S, N>>;
		};

		//@note for now, bigger than 64 lanes vectors are not supported (mainly due to mask type not being ready for it)
		//@tparam S scalar type of the would-be vector
		//@tparam N lane count of the would-be vector
		template<typename S, size_t N> concept IsValid_SIMD_Vector = IsScalarType<S> && ((N == 1) || isPowerOf2(N));

		template<typename T> concept xmm_sized = vector_size_class_v<T> == VectorSizeClassEnum::XMM;
		template<typename T> concept ymm_sized = vector_size_class_v<T> == VectorSizeClassEnum::YMM;
		template<typename T> concept zmm_sized = vector_size_class_v<T> == VectorSizeClassEnum::ZMM;

		constexpr bool is_xmm_size(size_t N) { return N <= 16; }
		constexpr bool is_ymm_size(size_t N) { return N > 16 && N <= 32; }
		constexpr bool is_zmm_size(size_t N) { return N > 32 && N <= 64; }

		template <typename T> concept is_fp16 = std::same_as<T, fp16_t>;
		template <typename T> concept is_bf16 = std::same_as<T, bf16_t>;
		template <typename T> concept is_f32 = std::same_as<T, float>;
		template <typename T> concept is_f64 = std::same_as<T, double>;
		template <typename T> concept is_i64 = std::same_as<T, int64_t>;
		template <typename T> concept is_i32 = std::same_as<T, int32_t>;
		template <typename T> concept is_i16 = std::same_as<T, int16_t>;
		template <typename T> concept is_i8 = std::same_as<T, int8_t>;
		template <typename T> concept is_u64 = std::same_as<T, uint64_t>;
		template <typename T> concept is_u32 = std::same_as<T, uint32_t>;
		template <typename T> concept is_u16 = std::same_as<T, uint16_t>;
		template <typename T> concept is_u8 = std::same_as<T, uint8_t>;

		//indicates wheteher this type is a signed 8 or 16 bit integer
		template <typename T> requires (IsScalarType<T>) inline constexpr bool is_small_sint = is_i16<T> || is_i8<T>;
		//indicates wheteher this type is a unsigned 8 or 16 bit integer
		template <typename T> requires (IsScalarType<T>) inline constexpr bool is_small_uint = is_u16<T> || is_u8<T>;
		//indicates wheteher this type is any 8 or 16 bit integer, signed or unsigned
		template <typename T> requires (IsScalarType<T>) inline constexpr bool any_small_int = is_small_sint<T> || is_small_uint<T>;

		//indicates wheteher this type is a floating point type (double, single, half precision or BF16)
		//Note that std::is_floating_point_v is not exactly equal to this, since FP16 and BF16 have limited support and are using custom types
		template <typename T> concept any_float = is_any_of_v<T, float, double, fp16_t, bf16_t>;
		//indicates whether this type is 8 bit integer, signed or unsigned
		template <typename T> concept any_i8 = (is_u8<T> || is_i8<T>);
		//indicates whether this type is 16 bit integer, signed or unsigned
		template <typename T> concept any_i16 = (is_u16<T> || is_i16<T>);
		//indicates whether this type is 32 bit integer, signed or unsigned
		template <typename T> concept any_i32 = (is_u32<T> || is_i32<T>);
		//indicates whether this type is 64 bit integer, signed or unsigned
		template <typename T> concept any_i64 = (is_u64<T> || is_i64<T>);
		//indicates whether this type is integral scalar type
		template <typename T> concept any_int = std::is_integral_v<T> && IsScalarType<T>;
		template <typename T> concept any_uint = any_int<T> && !std::is_signed_v<T>;
		//indicates whether this type is not integral
		template <typename T> concept not_int = IsScalarType<T> && !std::is_integral_v<T>;

		template <typename T> concept IsCvtOp = requires {typename T::cvt_to_t; };
		template <typename T> concept IsLoadOp = requires {T::_avxxy_is_load_tag; };
		template <typename T> concept IsGatherOp = requires {T::_avxxy_is_gather_tag; };
		template <typename T> concept IsScatterOp = requires {T::_avxxy_is_scatter_tag; };
		template <typename T> concept IsMovmOp = requires { T::_avxxy_is_movm_tag; };

		template<size_t N1, size_t N2>
		concept SameSizeClasses = ((is_xmm_size(N1) && is_xmm_size(N2)) || (is_ymm_size(N1) && is_ymm_size(N2)) || (is_zmm_size(N1) && is_zmm_size(N2)));

		//Returns the number of elements of type S that the largest architectual registers of current feature set can hold.
		//Note that this in no way related to whether or not the operations on these vectors will be native or not.
		//It is purely a numerical size quantity, equal to largest native vector width divided by sizeof(S).
		//Largest native vector widths are:
		//128 bits for SSE and above
		//256 bits for AVX and above
		//512 bits for AVX512-F and above
		template<typename S> requires IsScalarType<S> inline constexpr size_t REG_LANE_COUNT_FOR = []() {
			using namespace internals;
			/*
			* 
			if constexpr (FS.has(AVX512_F) && (any_i64<S> || any_i32<S> || is_f32<S> || is_f64<S>)) return 64 / sizeof(S);
			else if constexpr ((FS.has(AVX2) && any_int<S>) || (FS.has(AVX) && (is_f32<S> || is_f64<S>))) return 32 / sizeof(S);
			else if constexpr ((FS.has(SSE2) && (any_int<S> || is_f64<S>)) || (FS.has(SSE) && (is_f32<S>))) return 16 / sizeof(S);*/
			//actually, maybe it's better to not allow mismatched vectors and masks. Thus, for now match based on which instruction set first introduced these vectors
			if constexpr (FS.has(AVX512_F)) return 64 / sizeof(S);
			else if constexpr (FS.has(AVX)) return 32 / sizeof(S);
			else if constexpr (FS.has(SSE)) return 16 / sizeof(S);
			else return 1;
			}();

		template<typename T>
		concept IsSimdVector = requires { T::IsSimdVector; };

		template<typename S>
		concept vpopcnt_allowed = (meta::any_int<S> || settings::ALLOW_VPOPCNT_FOR_NON_INTS);

		//TODO: relax this requirement some time. It needs at least 1 S element starting at 64 bits of xmm
		template<typename S, size_t N>
		concept unpackhi_legal = IsScalarType<S> && (sizeof(S) * N % 16 == 0);
	}
}