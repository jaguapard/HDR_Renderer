#pragma once
#include "../namespace.h"
#include "../tags.h"
#include "../SIMD_BitMask.h"
#include "../SIMD_Vector.h"
#include "../FeatureSet.h"
#include "../funcs.h"
#include "../tables.h"

namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		namespace ISA
		{
			using namespace concepts;
			using namespace utils;
			template<internals::FeatureSet FS>
			struct AVX
			{
				template<typename S, size_t N>
					requires ((is_f32<S> || is_f64<S>) && sizeof(SIMD_Vector<S, N>) > 16)
				static SIMD_Vector<S, N> eval(op_abs, const SIMD_Vector<S, N>& a)
				{
					using U = same_size_uint_t<S>::type;
					constexpr U sb = U(1) << (sizeof(S) * 8 - 1);
					return logic_xor(a, SIMD_Vector<S,N>(std::bit_cast<S>(~sb))); //force sign bit to 0
				}

				template<typename S, size_t N>
					requires ((is_f32<S> || is_f64<S>) && sizeof(SIMD_Vector<S, N>) > 16)
				static SIMD_Vector<S, N> eval(op_add, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32) return { add(a.lo(), b.lo()), add(a.hi(),b.hi()) };
					else if constexpr (ymm_sized<T> && is_f64<S>) return _mm256_add_pd(a, b);
					else if constexpr (ymm_sized<T> && is_f32<S>) return _mm256_add_ps(a, b);
					else static_assert(always_false_v<T>);
				}
				template<typename S, size_t N>
					requires ((is_f32<S> || is_f64<S>) && sizeof(SIMD_Vector<S, N>) > 16)
				static SIMD_Vector<S, N> eval(op_sub, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32) return { sub(a.lo(), b.lo()), sub(a.hi(),b.hi()) };
					else if constexpr (ymm_sized<T> && is_f64<S>) return _mm256_sub_pd(a, b);
					else if constexpr (ymm_sized<T> && is_f32<S>) return _mm256_sub_ps(a, b);
					else static_assert(always_false_v<T>);
				}

				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 16)
				static SIMD_Vector<S, N> eval(op_and, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32) return { logic_and(a.lo(), b.lo()), logic_and(a.hi(), b.hi()) };
					else if constexpr (ymm_sized<T>) return _mm256_and_ps(vreinterpret<__m256>(a), vreinterpret<__m256>(b));
					else static_assert(always_false_v<T>);
				}
				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 16)
				static SIMD_Vector<S, N> eval(op_or, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32) return { logic_or(a.lo(), b.lo()), logic_or(a.hi(), b.hi()) };
					else if constexpr (ymm_sized<T>) return _mm256_or_ps(vreinterpret<__m256>(a), vreinterpret<__m256>(b));
					else static_assert(always_false_v<T>);
				}
				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 16)
				static SIMD_Vector<S, N> eval(op_xor, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32) return { logic_xor(a.lo(), b.lo()), logic_xor(a.hi(), b.hi()) };
					else if constexpr (ymm_sized<T>) return _mm256_xor_ps(vreinterpret<__m256>(a), vreinterpret<__m256>(b));
					else static_assert(always_false_v<T>);
				}
				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 16)
				static SIMD_Vector<S, N> eval(op_not, const SIMD_Vector<S, N>& a)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32) return { logic_not(a.lo()), logic_not(a.hi()) };
					//can't use compare random vectors trick, since FP comparisons are wonky with NaNs and signed 0, thus, have to explicitly set all ones
					else if (ymm_sized<T>) return _mm256_xor_ps(vreinterpret<__m256>(a), _mm256_set1_ps(std::bit_cast<float>(0xFFFFFFFF)));
					else static_assert(always_false_v<T>);
				}
			};
		}
	}
}