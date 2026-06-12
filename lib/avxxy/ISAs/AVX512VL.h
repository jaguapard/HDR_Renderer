#pragma once
#include "../namespace.h"
#include "../tags.h"
#include "../SIMD_BitMask.h"
#include "../SIMD_Vector.h"
#include "../FeatureSet.h"
namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		namespace ISA
		{
			template<internals::FeatureSet FS>
			struct AVX512VL
			{
				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_mask_mov, const SIMD_Vector<S, N>& ifBitClear, const SIMD_BitMask<N>& mask, const SIMD_Vector<S, N>& ifBitSet)
					requires (sizeof(S) < 4 && FS.AVX512.BW && (concepts::xmm_sized<SIMD_Vector<S,N>> || concepts::ymm_sized<SIMD_Vector<S,N>>))
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (ymm_sized<T> && any_i16<S>) return _mm256_mask_mov_epi16(ifBitClear, mask, ifBitSet);
					else if constexpr (ymm_sized<T> && any_i8<S>) return _mm256_mask_mov_epi8(ifBitClear, mask, ifBitSet);
					else if constexpr (xmm_sized<T> && any_i16<S>) return _mm_mask_mov_epi16(ifBitClear, mask, ifBitSet);
					else if constexpr (xmm_sized<T> && any_i8<S>) return _mm_mask_mov_epi8(ifBitClear, mask, ifBitSet);
					else static_assert(always_false_v<S>);
				}

				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_mask_mov, const SIMD_Vector<S, N>& ifBitClear, const SIMD_BitMask<N>& mask, const SIMD_Vector<S, N>& ifBitSet)
					requires (sizeof(S) >= 4 && FS.AVX512.F && (concepts::xmm_sized<SIMD_Vector<S, N>> || concepts::ymm_sized<SIMD_Vector<S, N>>))
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (ymm_sized<T> && is_f64<S>) return _mm256_mask_mov_pd(ifBitClear, mask, ifBitSet);
					else if constexpr (ymm_sized<T> && is_f32<S>) return _mm256_mask_mov_ps(ifBitClear, mask, ifBitSet);
					else if constexpr (ymm_sized<T> && any_i64<S>) return _mm256_mask_mov_epi64(ifBitClear, mask, ifBitSet);
					else if constexpr (ymm_sized<T> && any_i32<S>) return _mm256_mask_mov_epi32(ifBitClear, mask, ifBitSet);

					else if constexpr (xmm_sized<T> && is_f64<S>) return _mm_mask_mov_pd(ifBitClear, mask, ifBitSet);
					else if constexpr (xmm_sized<T> && is_f32<S>) return _mm_mask_mov_ps(ifBitClear, mask, ifBitSet);
					else if constexpr (xmm_sized<T> && any_i64<S>) return _mm_mask_mov_epi64(ifBitClear, mask, ifBitSet);
					else if constexpr (xmm_sized<T> && any_i32<S>) return _mm_mask_mov_epi32(ifBitClear, mask, ifBitSet);
					else static_assert(always_false_v<S>);
				}
			};
		}
	}
}