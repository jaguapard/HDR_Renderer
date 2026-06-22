#pragma once
#include "shared.h"

namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		struct op_add : OperationBase
		{
			template<typename S, size_t N>
			static SIMD_Vector<S, N> run(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				if constexpr (MaxVectorSize != 0 && sizeof(SIMD_Vector<S, N>) > MaxVectorSize) return { run(a.lo(), b.lo()), run(a.hi(),b.hi()) };
				else return backend(a, b);
			}

		private:
			template<typename S, size_t N>
			static SIMD_Vector<S, N> backend(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				using T = SIMD_Vector<S, N>;
				if constexpr (FS.has(AVX512_BW))
				{
					if constexpr (zmm_sized<T> && any_i16<S>) return _mm512_add_epi16(a, b);
					else if constexpr (zmm_sized<T> && any_i8<S>) return _mm512_add_epi8(a, b);
				}
				if constexpr (FS.has(internals::AVX512_F))
				{
					if constexpr (zmm_sized<T> && is_f64<S>) return _mm512_add_pd(a, b);
					else if constexpr (zmm_sized<T> && is_f32<S>) return _mm512_add_ps(a, b);
					else if constexpr (zmm_sized<T> && any_i64<S>) return _mm512_add_epi64(a, b);
					else if constexpr (zmm_sized<T> && any_i32<S>) return _mm512_add_epi32(a, b);
				}
				if constexpr (FS.has(AVX2))
				{
					if constexpr (ymm_sized<T> && any_i64<S>) return _mm256_add_epi64(a, b);
					if constexpr (ymm_sized<T> && any_i32<S>) return _mm256_add_epi32(a, b);
					if constexpr (ymm_sized<T> && any_i16<S>) return _mm256_add_epi16(a, b);
					if constexpr (ymm_sized<T> && any_i8<S>) return _mm256_add_epi8(a, b);
				}
				if constexpr (FS.has(AVX))
				{
					if constexpr (ymm_sized<T> && is_f64<S>) return _mm256_add_pd(a, b);
					if constexpr (ymm_sized<T> && is_f32<S>) return _mm256_add_ps(a, b);
				}
				if constexpr (FS.has(SSE2))
				{
					if constexpr (xmm_sized<T> && is_f64<S>) return _mm_add_pd(a, b);
					else if constexpr (xmm_sized<T> && any_i64<S>) return _mm_add_epi64(a, b);
					else if constexpr (xmm_sized<T> && any_i32<S>) return _mm_add_epi32(a, b);
					else if constexpr (xmm_sized<T> && any_i16<S>) return _mm_add_epi16(a, b);
					else if constexpr (xmm_sized<T> && any_i8<S>) return _mm_add_epi8(a, b);
				}
				if constexpr (FS.has(SSE) && xmm_sized<T> && is_f32<S>) return _mm_add_ps(a, b);

				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = a[i] + b[i];
				return ret;
			}
		};
	}
}