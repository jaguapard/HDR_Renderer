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
					requires (sizeof(S) >= 4 && sizeof(SIMD_Vector<S, N>) > 32)
				static SIMD_Vector<S, N> eval(op_abs, const SIMD_Vector<S, N>& a)
				{
					using namespace concepts;
					if constexpr (sizeof(SIMD_Vector<S, N>) > 64) return { abs(a.lo()), abs(a.hi()) };
					else if constexpr (is_f64<S>) return _mm512_abs_pd(a);
					else if constexpr (is_f32<S>) return _mm512_abs_ps(a);
					else if constexpr (is_i64<S>) return _mm512_abs_epi64(a);
					else if constexpr (is_i32<S>) return _mm512_abs_epi32(a);
					else static_assert(always_false_v<S>);
				}

				template<typename S, size_t N>
					requires (sizeof(S) >= 4 && sizeof(SIMD_Vector<S, N>) > 32)
				static SIMD_Vector<S, N> eval(op_min, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using namespace concepts;
					if constexpr (sizeof(SIMD_Vector<S, N>) > 64) return { min(a.lo(), b.lo()), min(a.hi(),b.hi()) };
					else if constexpr (is_f64<S>) return _mm512_min_pd(a, b);
					else if constexpr (is_f32<S>) return _mm512_min_ps(a, b);
					else if constexpr (is_i64<S>) return _mm512_min_epi64(a, b);
					else if constexpr (is_u64<S>) return _mm512_min_epu64(a, b);
					else if constexpr (is_i32<S>) return _mm512_min_epi32(a, b);
					else if constexpr (is_u32<S>) return _mm512_min_epu32(a, b);
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N>
					requires (sizeof(S) >= 4 && sizeof(SIMD_Vector<S, N>) > 32)
				static SIMD_Vector<S, N> eval(op_max, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using namespace concepts;
					if constexpr (sizeof(SIMD_Vector<S, N>) > 64) return { max(a.lo(), b.lo()), max(a.hi(),b.hi()) };
					else if constexpr (is_f64<S>) return _mm512_max_pd(a, b);
					else if constexpr (is_f32<S>) return _mm512_max_ps(a, b);
					else if constexpr (is_i32<S>) return _mm512_max_epi32(a, b);
					else if constexpr (is_u32<S>) return _mm512_max_epu32(a, b);
					else if constexpr (is_i64<S>) return _mm512_max_epi64(a, b);
					else if constexpr (is_u64<S>) return _mm512_max_epu64(a, b);
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
				requires (sizeof(SIMD_Vector<S,N>) > 32 && sizeof(S)>=4)
				static SIMD_BitMask<N> eval(op_cmpeq, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 64) return { cmp_equal(a.lo(),b.lo()), cmp_equal(a.hi(),b.hi()) };
					else if constexpr (is_f64<S>) return _mm512_cmp_pd_mask(a, b, _CMP_EQ_OQ);
					else if constexpr (is_f32<S>) return _mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ);
					else if constexpr (is_i64<S>) return _mm512_cmpeq_epi64_mask(a, b);
					else if constexpr (is_u64<S>) return _mm512_cmpeq_epu64_mask(a, b);
					else if constexpr (is_i32<S>) return _mm512_cmpeq_epi32_mask(a, b);
					else if constexpr (is_u32<S>) return _mm512_cmpeq_epu32_mask(a, b);
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N>
				requires (sizeof(SIMD_Vector<S,N>) > 32 && sizeof(S)>=4)
				static SIMD_BitMask<N> eval(op_cmpneq, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 64) return { cmp_not_equal(a.lo(),b.lo()), cmp_not_equal(a.hi(),b.hi()) };
					else if constexpr (is_f64<S>) return _mm512_cmp_pd_mask(a, b, _CMP_NEQ_OQ);
					else if constexpr (is_f32<S>) return _mm512_cmp_ps_mask(a, b, _CMP_NEQ_OQ);
					else if constexpr (is_i64<S>) return _mm512_cmpneq_epi64_mask(a, b);
					else if constexpr (is_u64<S>) return _mm512_cmpneq_epu64_mask(a, b);
					else if constexpr (is_i32<S>) return _mm512_cmpneq_epi32_mask(a, b);
					else if constexpr (is_u32<S>) return _mm512_cmpneq_epu32_mask(a, b);
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N>
				requires (sizeof(SIMD_Vector<S,N>) > 32 && sizeof(S)>=4)
				static SIMD_BitMask<N> eval(op_cmpgt, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 64) return { cmp_greater(a.lo(),b.lo()), cmp_greater(a.hi(),b.hi()) };
					else if constexpr (is_f64<S>) return _mm512_cmp_pd_mask(a, b, _CMP_GT_OQ);
					else if constexpr (is_f32<S>) return _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ);
					else if constexpr (is_i64<S>) return _mm512_cmpgt_epi64_mask(a, b);
					else if constexpr (is_u64<S>) return _mm512_cmpgt_epu64_mask(a, b);
					else if constexpr (is_i32<S>) return _mm512_cmpgt_epi32_mask(a, b);
					else if constexpr (is_u32<S>) return _mm512_cmpgt_epu32_mask(a, b);
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N>
				requires (sizeof(SIMD_Vector<S,N>) > 32 && sizeof(S)>=4)
				static SIMD_BitMask<N> eval(op_cmpge, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 64) return { cmp_greater_or_equal(a.lo(),b.lo()), cmp_greater_or_equal(a.hi(),b.hi()) };
					else if constexpr (is_f64<S>) return _mm512_cmp_pd_mask(a, b, _CMP_GE_OQ);
					else if constexpr (is_f32<S>) return _mm512_cmp_ps_mask(a, b, _CMP_GE_OQ);
					else if constexpr (is_i64<S>) return _mm512_cmpge_epi64_mask(a, b);
					else if constexpr (is_u64<S>) return _mm512_cmpge_epu64_mask(a, b);
					else if constexpr (is_i32<S>) return _mm512_cmpge_epi32_mask(a, b);
					else if constexpr (is_u32<S>) return _mm512_cmpge_epu32_mask(a, b);
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 32 && sizeof(S) >= 4)
				static SIMD_BitMask<N> eval(op_cmplt, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 64) return { cmp_less(a.lo(),b.lo()), cmp_less(a.hi(),b.hi()) };
					else if constexpr (is_f64<S>) return _mm512_cmp_pd_mask(a, b, _CMP_LT_OQ);
					else if constexpr (is_f32<S>) return _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ);
					else if constexpr (is_i64<S>) return _mm512_cmplt_epi64_mask(a, b);
					else if constexpr (is_u64<S>) return _mm512_cmplt_epu64_mask(a, b);
					else if constexpr (is_i32<S>) return _mm512_cmplt_epi32_mask(a, b);
					else if constexpr (is_u32<S>) return _mm512_cmplt_epu32_mask(a, b);
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 32 && sizeof(S) >= 4)
				static SIMD_BitMask<N> eval(op_cmple, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 64) return { cmp_less_or_equal(a.lo(),b.lo()), cmp_less_or_equal(a.hi(),b.hi()) };
					else if constexpr (is_f64<S>) return _mm512_cmp_pd_mask(a, b, _CMP_LE_OQ);
					else if constexpr (is_f32<S>) return _mm512_cmp_ps_mask(a, b, _CMP_LE_OQ);
					else if constexpr (is_i64<S>) return _mm512_cmple_epi64_mask(a, b);
					else if constexpr (is_u64<S>) return _mm512_cmple_epu64_mask(a, b);
					else if constexpr (is_i32<S>) return _mm512_cmple_epi32_mask(a, b);
					else if constexpr (is_u32<S>) return _mm512_cmple_epu32_mask(a, b);
					else static_assert(always_false_v<S>);
				}

				template<typename S, size_t N, typename I>
					requires (sizeof(S) >= 4 && concepts::any_int<I> && concepts::zmm_sized<SIMD_Vector<S, N>>)//sizeof(SIMD_Vector<S,N>& > 32))
				static SIMD_Vector<S, N> eval(op_permx, const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& ind)
				{
					using namespace concepts;
					using canon_t = typename same_size_uint_t<S>::type;
					if constexpr (sizeof(I) != sizeof(S)) return permx(a, vcvt<canon_t>(ind));
					//TODO: add > 64 byte permutex!
					else if constexpr (is_f64<S>) return _mm512_permutexvar_pd(ind, a);
					else if constexpr (is_f32<S>) return _mm512_permutexvar_ps(ind, a);
					else if constexpr (any_i64<S>) return _mm512_permutexvar_epi64(ind, a);
					else if constexpr (any_i32<S>) return _mm512_permutexvar_epi32(ind, a);
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N, typename I>
					requires (sizeof(S) >= 4 && concepts::any_int<I> && concepts::zmm_sized<SIMD_Vector<S, N>>)//sizeof(SIMD_Vector<S,N>& > 32))
				static SIMD_Vector<S, N> eval(op_permx2, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b, const SIMD_Vector<I, N>& ind)
				{
					using namespace concepts;
					using canon_t = typename same_size_uint_t<S>::type;
					if constexpr (sizeof(I) != sizeof(S)) return permx2(a, vcvt<canon_t>(ind));
					//TODO: add > 64 byte permutex2!
					else if constexpr (is_f64<S>) return _mm512_permutex2var_pd(a, ind, b);
					else if constexpr (is_f32<S>) return _mm512_permutex2var_ps(a, ind, b);
					else if constexpr (any_i64<S>) return _mm512_permutex2var_epi64(a, ind, b);
					else if constexpr (any_i32<S>) return _mm512_permutex2var_epi32(a, ind, b);
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

				template<typename S, size_t N>
					requires (std::is_floating_point_v<S> && sizeof(SIMD_Vector<S,N>) > 32)
				static SIMD_Vector<S, N> eval(op_floor, const SIMD_Vector<S, N>& a)
				{
					if constexpr (sizeof(SIMD_Vector<S, N>) > 64) return { floor(a.lo()), floor(a.hi()) };
					else if constexpr (is_f64<S>) return _mm512_floor_pd(a);
					else if constexpr (is_f32<S>) return _mm512_floor_ps(a);
					else static_assert(always_false_v<S>);
				}
				template<typename S, size_t N>
					requires (std::is_floating_point_v<S> && sizeof(SIMD_Vector<S,N>) > 32)
				static SIMD_Vector<S, N> eval(op_ceil, const SIMD_Vector<S, N>& a)
				{
					if constexpr (sizeof(SIMD_Vector<S, N>) > 64) return { ceil(a.lo()), ceil(a.hi()) };
					else if constexpr (is_f64<S>) return _mm512_ceil_pd(a);
					else if constexpr (is_f32<S>) return _mm512_ceil_ps(a);
					else static_assert(always_false_v<S>);
				}
				template<size_t N>
				requires (sizeof(SIMD_Vector<float, N>) > 32)
				static SIMD_Vector<float, N> eval(op_sqrtf, const SIMD_Vector<float, N>& a)
				{
					if constexpr (sizeof(SIMD_Vector<float, N>) > 64) return { sqrtf(a.lo()), sqrtf(a.hi()) };
					else return _mm512_sqrt_ps(a);
				}
				template<size_t N>
				requires (sizeof(SIMD_Vector<double, N>) > 32)
				static SIMD_Vector<double, N> eval(op_sqrtd, const SIMD_Vector<double, N>& a)
				{
					if constexpr (sizeof(SIMD_Vector<double, N>) > 64) return { sqrtd(a.lo()), sqrtd(a.hi()) };
					else return _mm512_sqrt_pd(a);
				}
				
				template<typename To, size_t N, typename From>
					requires (std::max(sizeof(SIMD_Vector<To, N>), sizeof(SIMD_Vector<From, N>)) > 32 && (
				// from double
				(is_f64<From> && is_i32<To>) || (is_f64<From> && is_u32<To>) || (is_f64<From> && is_f32<To>)
					|| (is_f32<From> && is_i32<To>) || (is_f32<From> && is_u32<To>) || (is_f32<From> && is_f64<To>)

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

				template<typename S, size_t N, size_t Scale, typename I>
					requires (concepts::any_int<I> && sizeof(S) >= 4 && std::max(sizeof(SIMD_Vector<S, N>), sizeof(SIMD_Vector<I, N>)) > 32)
				static SIMD_Vector<S, N> eval(op_gather<S, N, Scale>, const void* base, const SIMD_Vector<I, N>& ind, const SIMD_BitMask<N>& mask = SIMD_BitMask<N>::AllOnes, const SIMD_Vector<S, N>& src = 0)
				{
					//put everything up here to prevent else if chain breaks (since compilation gives useless errors by thinking unsanitized inputs surviving to native gathers
					using CanonicalIndex_t = std::conditional_t<(sizeof(I) <= 4), int32_t, int64_t>;
					using RetVec_t = SIMD_Vector<S, N>;
					using IndVec_t = SIMD_Vector<I, N>;
					constexpr size_t MaxSize = std::max(sizeof(RetVec_t), sizeof(IndVec_t));

					//if scale is not native, emulate it by gathering with scale 1 and manually calculated byte offsets. 
					//TODO: Can optimize a little by checking if Scale*maxint(I) fits into smaller sizes
					if constexpr (Scale != 1 && Scale != 2 && Scale != 4 && Scale != 8) return gather<S, N, 1>(base, vcvt<int64_t>(ind) * Scale, mask, src);
					
					//TODO: emulation of small int gathers (where elements gathered are small ints)
					else if constexpr (!std::is_same_v<I, CanonicalIndex_t>) return gather<S, N, Scale>(base, vcvt<CanonicalIndex_t>(ind), mask, src);

					//if we get here, means that indices are already in good format (4-byte or 8-byte)
					//break up large gather into halves
					else if constexpr (MaxSize > 64) return {
						gather<S, N / 2, Scale, I>(base, ind.lo(), mask.lo(), src.lo()),
						gather<S, N / 2, Scale, I>(base, ind.hi(), mask.hi(), src.hi()) };
					else if constexpr (utils::is_zmm_size(MaxSize))
					{
						//clang is a cry-baby with ind here for some reason, so force convert it. Pay attention to size!
						std::conditional_t<(concepts::zmm_sized<IndVec_t>), __m512i, __m256i> ni = ind;
						if constexpr (is_i64<I> && is_f64<S>) return _mm512_mask_i64gather_pd(src, mask, ni, base, Scale);
						else if constexpr (is_i64<I> && is_f32<S>) return _mm512_mask_i64gather_ps(src, mask, ni, base, Scale);
						else if constexpr (is_i64<I> && any_i64<S>) return _mm512_mask_i64gather_epi64(src, mask, ni, base, Scale);
						else if constexpr (is_i64<I> && any_i32<S>) return _mm512_mask_i64gather_epi32(src, mask, ni, base, Scale);

						else if constexpr (is_i32<I> && is_f64<S>) return _mm512_mask_i32gather_pd(src, mask, ni, base, Scale);
						else if constexpr (is_i32<I> && is_f32<S>) return _mm512_mask_i32gather_ps(src, mask, ni, base, Scale);
						else if constexpr (is_i32<I> && any_i64<S>) return _mm512_mask_i32gather_epi64(src, mask, ni, base, Scale);
						else if constexpr (is_i32<I> && any_i32<S>) return _mm512_mask_i32gather_epi32(src, mask, ni, base, Scale);
					}
					else static_assert(always_false_v<I, S>);
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

				template <size_t N>
				requires (sizeof(SIMD_Vector<float,N>) > 32)
				static SIMD_Vector<uint16_t, N> eval(op_fp32_to_fp16, const SIMD_Vector<float, N>& a)
				{
					if constexpr (sizeof(SIMD_Vector<float, N>) > 64) return { vcvt_fp32_fp16(a.lo()), vcvt_fp32_fp16(a.hi()) };
					else return _mm512_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT);
				}
				template <size_t N>
					requires (sizeof(SIMD_Vector<float, N>) > 32)
				static SIMD_Vector<float, N> eval(op_fp16_to_fp32, const SIMD_Vector<uint16_t, N>& a)
				{
					if constexpr (sizeof(SIMD_Vector<float, N>) > 64) return { vcvt_fp16_fp32(a.lo()), vcvt_fp16_fp32(a.hi()) };
					else return _mm512_cvtph_ps(a);
				}
				private:

			};
		}
	}
};