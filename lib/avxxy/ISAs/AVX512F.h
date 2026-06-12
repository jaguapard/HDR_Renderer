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
			using namespace concepts;
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

				template<typename S, size_t N>
					requires (sizeof(S) >= 4 && sizeof(SIMD_Vector<S, N>) > 32)
				static SIMD_Vector<S, N> eval(op_load<S, N>, const void* p, const SIMD_BitMask<N>& mask = SIMD_BitMask<N>::AllOnes, const SIMD_Vector<S, N>& src = 0)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					SIMD_Vector<S, N> ret;
					const S* sp = (const S*)p;
					if constexpr (sizeof(T) > 64) return { load<S,N / 2>(sp,mask.lo(), src.lo()), load<S,N / 2>(sp + N / 2, mask.hi(), src.hi()) };
					else if constexpr (is_f64<S>) return _mm512_mask_loadu_pd(src, mask, p);
					else if constexpr (is_f32<S>) return _mm512_mask_loadu_ps(src, mask, p);
					else if constexpr (any_i64<S>) return _mm512_mask_loadu_epi64(src, mask, p);
					else if constexpr (any_i32<S>) return _mm512_mask_loadu_epi32(src, mask, p);
					else static_assert(always_false_v<S>);
				}

				template<typename To, size_t N, typename From>
				requires (std::max(sizeof(SIMD_Vector<To, N>), sizeof(SIMD_Vector<From, N>)) > 32 && (
					// from double
					(is_f64<From> && is_i32<To>) || (is_f64<From> && is_u32<To>) || (is_f64<From> && is_f32<To>)
					|| (is_f32<From> && is_i32<To>) || (is_f32<From> && is_u32<To>)|| (is_f32<From> && is_f64<To>)

					(any_i64<From> && any_i32<To>) || (any_i64<From> && any_i16<To>) || (any_i64<From> && any_i8<To>)

					// from i32
					|| (is_i32<From> && is_f64<To>) || (is_i32<From> && is_f32<To>) || (is_i32<From> && any_i64<To>)
					|| (is_i32<From> && any_i16<To>) || (is_i32<From> && any_i8<To>)

					// from u32
					|| (is_u32<From> && is_f64<To>) || (is_u32<From> && is_f32<To>) || (is_u32<From> && any_i64<To>)

					// from 16-bit ints
					|| (is_i16<From> && any_i64<To>) || (is_i16<From> && any_i32<To>)
					|| (is_u16<From> && any_i64<To>) || (is_u16<From> && any_i32<To>)

					// from 8-bit ints
					|| (is_i8<From> && any_i64<To>) || (is_i8<From> && any_i32<To>)
					|| (is_u8<From> && any_i64<To>) || (is_u8<From> && any_i32<To>)))
				static SIMD_Vector<To, N> eval(op_cvt<To>, const SIMD_Vector<From, N>& a)
				{
					using namespace concepts;
					using TV = SIMD_Vector<To, N>;
					using FV = SIMD_Vector<From, N>;
					constexpr size_t MaxSize = std::max(sizeof(TV), sizeof(FV));

					if constexpr (MaxSize > 64) return { vcvt<To>(a.lo()), vcvt<To>(a.hi()) };
					//from double
					else if constexpr (is_f64<From> && is_i32<To>) return _mm512_cvttpd_epi32(a);
					else if constexpr (is_f64<From> && is_u32<To>) return _mm512_cvttpd_epu32(a);
					else if constexpr (is_f64<From> && is_f32<To>) return _mm512_cvtpd_ps(a);

					//from float
					else if constexpr (is_f32<From> && is_i32<To>) return _mm512_cvttps_epi32(a);
					else if constexpr (is_f32<From> && is_u32<To>) return _mm512_cvttps_epu32(a);
					else if constexpr (is_f32<From> && is_f64<To>) return _mm512_cvtps_pd(a);

					//from i64
					else if constexpr (any_i64<From> && any_i32<To>) return _mm512_cvtepi64_epi32(a);
					else if constexpr (any_i64<From> && any_i16<To>) return _mm512_cvtepi64_epi16(a);
					else if constexpr (any_i64<From> && any_i8<To>) return _mm512_cvtepi64_epi8(a);

					//from i32
					else if constexpr (is_i32<From> && is_f64<To>) return _mm512_cvtepi32_pd(a);
					else if constexpr (is_i32<From> && is_f32<To>) return _mm512_cvtepi32_ps(a);
					else if constexpr (is_i32<From> && any_i64<To>) return _mm512_cvtepi32_epi64(a);
					else if constexpr (is_i32<From> && any_i16<To>) return _mm512_cvtepi32_epi16(a);
					else if constexpr (is_i32<From> && any_i8<To>) return _mm512_cvtepi32_epi8(a);

					//from u32
					else if constexpr (is_u32<From> && is_f64<To>) return _mm512_cvtepu32_pd(a);
					else if constexpr (is_u32<From> && is_f32<To>) return _mm512_cvtepu32_ps(a);
					else if constexpr (is_u32<From> && any_i64<To>) return _mm512_cvtepu32_epi64(a);

					//from 16 bit ints
					else if constexpr (is_i16<From> && any_i64<To>) return _mm512_cvtepi16_epi64(a);
					else if constexpr (is_i16<From> && any_i32<To>) return _mm512_cvtepi16_epi32(a);
					else if constexpr (is_u16<From> && any_i64<To>) return _mm512_cvtepu16_epi64(a);
					else if constexpr (is_u16<From> && any_i32<To>) return _mm512_cvtepu16_epi32(a);

					//from 8 bit ints
					else if constexpr (is_i8<From> && any_i64<To>) return _mm512_cvtepi8_epi64(a);
					else if constexpr (is_i8<From> && any_i32<To>) return _mm512_cvtepi8_epi32(a);
					else if constexpr (is_u8<From> && any_i64<To>) return _mm512_cvtepu8_epi64(a);
					else if constexpr (is_u8<From> && any_i32<To>) return _mm512_cvtepu8_epi32(a);
					else static_assert(always_false_v<To, From>);
				}


				template<typename S, size_t N>
					requires (sizeof(S) >= 4 && sizeof(SIMD_Vector<S, N>) > 32)
				static void eval(op_store, SIMD_Vector<S, N> vec, void* p, const SIMD_BitMask<N>& mask = SIMD_BitMask<N>::AllOnes)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					SIMD_Vector<S, N> ret;
					const S* sp = (const S*)p;
					if constexpr (sizeof(T) > 64) {
						store(vec.lo(), p, mask.lo());
						store(vec.hi(), sp + N / 2, mask.hi());
					}
					else if constexpr (is_f64<S>) return _mm512_mask_storeu_pd(p, mask, vec);
					else if constexpr (is_f32<S>) return _mm512_mask_storeu_ps(p, mask, vec);
					else if constexpr (any_i64<S>) return _mm512_mask_storeu_epi64(p, mask, vec);
					else if constexpr (any_i32<S>) return _mm512_mask_storeu_epi32(p, mask, vec);
					else static_assert(always_false_v<S>);
				}

				private:

			};
		}
	}
};