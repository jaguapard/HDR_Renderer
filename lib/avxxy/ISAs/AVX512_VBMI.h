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
			struct AVX512VBMI
			{
				template<typename S, size_t N, typename I>
					requires (concepts::any_i8<S>&& concepts::any_int<I>&& concepts::zmm_sized<SIMD_Vector<S, N>>)//sizeof(SIMD_Vector<S,N>& > 32))
				static SIMD_Vector<S, N> eval(op_permx, const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& ind)
				{
					using namespace concepts;
					using canon_t = typename same_size_uint_t<S>::type;
					if constexpr (sizeof(I) != sizeof(S)) return permx(a, vcvt<canon_t>(ind));
					//TODO: add > 64 byte permutex!
					else if constexpr (any_i8<S>) return _mm512_permutexvar_epi8(ind, a);
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N, typename I>
					requires (concepts::any_i8<S>&& concepts::any_int<I>&& concepts::zmm_sized<SIMD_Vector<S, N>>)//sizeof(SIMD_Vector<S,N>& > 32))
				static SIMD_Vector<S, N> eval(op_permx2, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b, const SIMD_Vector<I, N>& ind)
				{
					using namespace concepts;
					using canon_t = typename same_size_uint_t<S>::type;
					if constexpr (sizeof(I) != sizeof(S)) return permx(a, vcvt<canon_t>(ind));
					//TODO: add > 64 byte permutex2!
					else if constexpr (any_i8<S>) return _mm512_permutex2var_epi8(a, ind, b);
					else static_assert(always_false_v<S>);
				}
			};
		}
	}
}