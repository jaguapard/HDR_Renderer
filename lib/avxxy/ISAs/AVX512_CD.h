#pragma once
#include "../namespace.h"
#include "../tags.h"
#include "../SIMD_BitMask.h"
#include "../SIMD_Vector.h"
#include "../FeatureSet.h"
#include "../funcs.h"

namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		namespace ISA
		{
			using namespace concepts;
			using namespace utils;
			struct AVX512CD
			{
				static inline constexpr FeatureSet FS = internals::FS_current;
				//TODO: can make conflict detection for smaller scalar types too
				template<typename S, size_t N>
					requires (sizeof(S) * 8 >= N && sizeof(S) >= 4 && sizeof(SIMD_Vector<S,N>) >= (FS.has(AVX512_VL) ? 0 : 33))
				static SIMD_Vector<typename same_size_uint_t<S>::type, N> eval(op_conflict, const SIMD_Vector<S, N>& a)
				{
					using T = SIMD_Vector<S, N>;
					static_assert(sizeof(T) <= 64); //TODO: implement chained conflict detection
					if constexpr (zmm_sized<T> && sizeof(S) == 4) return _mm512_conflict_epi32(vcast<SIMD_Vector<int32_t, N>>(a));
					else if constexpr (zmm_sized<T> && sizeof(S) == 8) return _mm512_conflict_epi64(vcast<SIMD_Vector<int64_t, N>>(a));
					else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && sizeof(S) == 4) return _mm256_conflict_epi32(vcast<SIMD_Vector<int32_t, N>>(a));
					else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && sizeof(S) == 8) return _mm256_conflict_epi64(vcast<SIMD_Vector<int64_t, N>>(a));
					else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && sizeof(S) == 4) return _mm_conflict_epi32(vcast<SIMD_Vector<int32_t, N>>(a));
					else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && sizeof(S) == 8) return _mm_conflict_epi64(vcast<SIMD_Vector<int64_t, N>>(a));
					else static_assert(always_false_v<T>);
				}

				//TODO: add lzcnt
			};
		}
	}
}