#pragma once
#include "shared.h"
namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		using namespace meta;
		struct ISA_AVX512_VBMI2
		{
			static inline constexpr FeatureSet FS = internals::FS_current;
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op,op_compress> && any_small_int<S> && sizeof(SIMD_Vector<S, N>) >= (FS.has(AVX512_VL) ? 0 : 33))
			static auto eval(const mask_t<S, N>& mask, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& src = 0)
			{
				//TODO: more than 64 bytes!
				//if constexpr (sizeof(SIMD_Vector<S, N>) > 64) {};
				//else
				using T = SIMD_Vector<S, N>;
				if constexpr (zmm_sized<T> && any_i16<S>) return _mm512_mask_compress_epi16(src, mask, a);
				else if constexpr (zmm_sized<T> && any_i8<S>) return _mm512_mask_compress_epi8(src, mask, a);
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && any_i16<S>) return _mm256_mask_compress_epi16(src, mask, a);
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && any_i8<S>) return _mm256_mask_compress_epi8(src, mask, a);
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && any_i16<S>) return _mm_mask_compress_epi16(src, mask, a);
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && any_i8<S>) return _mm_mask_compress_epi8(src, mask, a);
				else return fail_ack_t{};
			}
		};
	}
}