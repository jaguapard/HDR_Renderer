#pragma once
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
			struct AVX512F
			{
				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_add, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
					requires (sizeof(S) >= 4 && sizeof(SIMD_Vector<S, N>) > 32)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 64) return { add(a.lo(), b.lo()), add(a.hi(), b.hi()) };
					else if constexpr (is_f64<S>) return _mm512_add_pd(a, b);
					else if constexpr (is_f32<S>) return _mm512_add_ps(a, b);
					else if constexpr (any_i64<S>) return _mm512_add_epi64(a, b);
					else if constexpr (any_i32<S>) return _mm512_add_epi32(a, b);
					else static_assert(always_false_v<S>);
				}

				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_sub, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
					requires (sizeof(S) >= 4 && sizeof(SIMD_Vector<S, N>) > 32)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 64) return { sub(a.lo(), b.lo()), sub(a.hi(), b.hi()) };
					else if constexpr (is_f64<S>) return _mm512_sub_pd(a, b);
					else if constexpr (is_f32<S>) return _mm512_sub_ps(a, b);
					else if constexpr (any_i64<S>) return _mm512_sub_epi64(a, b);
					else if constexpr (any_i32<S>) return _mm512_sub_epi32(a, b);
					else static_assert(always_false_v<S>);
				}

				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_mul, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
					requires (sizeof(S) >= 4 && sizeof(SIMD_Vector<S, N>) > 32)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 64) return { mul(a.lo(), b.lo()), mul(a.hi(), b.hi()) };
					else if constexpr (is_f64<S>) return _mm512_mul_pd(a, b);
					else if constexpr (is_f32<S>) return _mm512_mul_ps(a, b);
					else if constexpr (any_i64<S>) return _mm512_mullox_epi64(a, b);
					else if constexpr (any_i32<S>) return _mm512_mullo_epi32(a, b);
					else static_assert(always_false_v<S>);
				}

				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_mask_mov, const SIMD_Vector<S, N>& ifBitClear, const SIMD_BitMask<N>& mask, const SIMD_Vector<S, N>& ifBitSet)
					requires (sizeof(S) >= 4 && sizeof(SIMD_Vector<S, N>) > 32)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 64) return { mask_mov(ifBitClear.lo(), mask.lo(), ifBitSet.lo()), mask_mov(ifBitClear.hi(), mask.hi(), ifBitSet.hi()) };
					else if constexpr (is_f64<S>) return _mm512_mask_mov_pd(ifBitClear, mask, ifBitSet);
					else if constexpr (is_f32<S>) return _mm512_mask_mov_ps(ifBitClear, mask, ifBitSet);
					else if constexpr (any_i64<S>) return _mm512_mask_mov_epi64(ifBitClear, mask, ifBitSet);
					else if constexpr (any_i32<S>) return _mm512_mask_mov_epi32(ifBitClear, mask, ifBitSet);
					else static_assert(always_false_v<S>);
				}
			};
		}
	}
};