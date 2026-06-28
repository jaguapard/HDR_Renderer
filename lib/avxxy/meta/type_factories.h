#pragma once
#include "../namespace.h"
#include "meta.h"
#include <type_traits>
#include <array>

namespace AVXXY_NAMESPACE
{
	namespace meta
	{
		template<typename S>
		requires (IsScalarType<S>)
		using xmm_t =
			std::conditional_t<std::is_integral_v<S>, __m128i,
			std::conditional_t<std::is_same_v<S, float>, __m128,
			std::conditional_t<std::is_same_v<S, double>, __m128d,
			std::conditional_t<std::is_same_v<S, fp16_t>, __m128h,
			std::conditional_t<std::is_same_v<S, bf16_t>, __m128bh,
			void>>>>>;

		template<typename S>
			requires (IsScalarType<S>)
		using ymm_t =
			std::conditional_t<std::is_integral_v<S>, __m256i,
			std::conditional_t<std::is_same_v<S, float>, __m256,
			std::conditional_t<std::is_same_v<S, double>, __m256d,
			std::conditional_t<std::is_same_v<S, fp16_t>, __m256h,
			std::conditional_t<std::is_same_v<S, bf16_t>, __m256bh,
			void>>>>>;

		template<typename S>
			requires (IsScalarType<S>)
		using zmm_t =
			std::conditional_t<std::is_integral_v<S>, __m512i,
			std::conditional_t<std::is_same_v<S, float>, __m512,
			std::conditional_t<std::is_same_v<S, double>, __m512d,
			std::conditional_t<std::is_same_v<S, fp16_t>, __m512h,
			std::conditional_t<std::is_same_v<S, bf16_t>, __m512bh,
			void>>>>>;

		template<typename S>
			requires (IsScalarType<S>)
		using same_sized_int_t =
			std::conditional_t<sizeof(S) == 1, int8_t,
			std::conditional_t<sizeof(S) == 2, int16_t,
			std::conditional_t<sizeof(S) == 4, int32_t, int64_t>>>;

		template<typename S>
			requires (IsScalarType<S>)
		using same_sized_uint_t =
			std::conditional_t<sizeof(S) == 1, uint8_t,
			std::conditional_t<sizeof(S) == 2, uint16_t,
			std::conditional_t<sizeof(S) == 4, uint32_t, uint64_t>>>;

		//Returns an intrinsic type that can hold N elements of type S
		//For 0..16 bytes: 128 bit intrinsic types (__m128, __m128i, __m128d, __m128h, __m128bh)
		//For 17..32: bytes: 256 bit intrinsic types (__m256, __m256i, __m256d, __m256h, __m256bh)
		//For 33..64: bytes: 512 bit intrinsic types (__m512, __m512i, __m512d, __m512h, __m512bh)
		//For larger than 64 bytes: std::array of 512 bit types
		template<typename S, size_t N>
		requires (IsScalarType<S>)
		using typed_intrinsic_storage_t =
			std::conditional_t<sizeof(S)* N <= 16, xmm_t<S>,
			std::conditional_t<sizeof(S)* N <= 32, ymm_t<S>,
			std::conditional_t<sizeof(S)* N <= 64, zmm_t<S>,
			std::array<zmm_t<S>, sizeof(S) * N / 64>>>>;

		template<size_t N>
		requires (N<=64)
		using bits_to_uint_t =
			std::conditional_t<N <= 8, uint8_t,
			std::conditional_t<N <= 16, uint16_t,
			std::conditional_t<N <= 32, uint32_t, uint64_t>>>;
		/*
		template<typename S, size_t N>
			requires (IsScalarType<S>)
		using mask_t = SIMD_Mask<scalar_size_class_v<S>, N>; 
		*/
		//using typed_intrinsic_storage_t = 
		//	std::conditional_t<
		//template<typename S, size_t N>
		//using typed_intrinsic_storage_t =  
	}
}