#pragma once
#include "shared.h"
namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		using namespace meta;
#if 1
		struct ISA_AVX512_CD {};
		#else
		struct ISA_AVX512_CD
		{
			//TODO: can make conflict detection for smaller scalar types too
			//TODO: update this for new architecture and change return type to bits_to_uint_t<N>
			template<typename Op, typename S, size_t N>
				requires (sizeof(S) * 8 >= N && sizeof(S) >= 4 && sizeof(SIMD_Vector<S, N>) >= (FS.has(AVX512_VL) ? 0 : 33) && std::same_as<Op,op_conflict>)
			static SIMD_Vector<typename ScalarTraits<S>::UintT, N> eval(const SIMD_Vector<S, N>& a)
			{
				using T = SIMD_Vector<S, N>;
				static_assert(sizeof(T) <= 64); //TODO: implement chained conflict detection
				if constexpr (zmm_sized<T> && sizeof(S) == 4) return _mm512_conflict_epi32(vcast<SIMD_Vector<int32_t, N>>(a));
				else if constexpr (zmm_sized<T> && sizeof(S) == 8) return _mm512_conflict_epi64(vcast<SIMD_Vector<int64_t, N>>(a));
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && sizeof(S) == 4) return _mm256_conflict_epi32(vcast<SIMD_Vector<int32_t, N>>(a));
				else if constexpr (FS.has(AVX512_VL) && ymm_sized<T> && sizeof(S) == 8) return _mm256_conflict_epi64(vcast<SIMD_Vector<int64_t, N>>(a));
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && sizeof(S) == 4) return _mm_conflict_epi32(vcast<SIMD_Vector<int32_t, N>>(a));
				else if constexpr (FS.has(AVX512_VL) && xmm_sized<T> && sizeof(S) == 8) return _mm_conflict_epi64(vcast<SIMD_Vector<int64_t, N>>(a));
				else return fail_ack_t{};
			}

			//TODO: add lzcnt
		};
#endif
	}
}