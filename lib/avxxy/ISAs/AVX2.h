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
			struct AVX2
			{
				static inline constexpr FeatureSet FS = internals::FS_current;
				template<typename S, size_t N>
					requires (std::is_signed_v<S> && std::is_integral_v<S> && sizeof(SIMD_Vector<S, N>) > 16)
				static SIMD_Vector<S, N> eval(op_abs, const SIMD_Vector<S, N>& a)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32) return { abs(a.lo()), abs(a.hi()) };
					else if constexpr (ymm_sized<T> && is_i64<S>)
					{
						__m256i cmp = _mm256_cmpgt_epi64(_mm256_setzero_si256(), a);
						return _mm256_blendv_epi8(a, -a, cmp);
					}
					else if constexpr (ymm_sized<T> && is_i32<S>) return _mm256_abs_epi32(a);
					else if constexpr (ymm_sized<T> && is_i16<S>) return _mm256_abs_epi16(a);
					else if constexpr (ymm_sized<T> && is_i16<S>) return _mm256_abs_epi8(a);
					else static_assert(always_false_v<T>);
				}

				template<typename S, size_t N>
				requires (any_int<S> && sizeof(SIMD_Vector<S, N>) > 16)
				static SIMD_Vector<S, N> eval(op_add, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32) return { add(a.lo(), b.lo()), add(a.hi(), b.hi()) };
					else if constexpr (ymm_sized<T> && any_i64<S>) return _mm256_add_epi64(a, b);
					else if constexpr (ymm_sized<T> && any_i32<S>) return _mm256_add_epi32(a, b);
					else if constexpr (ymm_sized<T> && any_i16<S>) return _mm256_add_epi16(a, b);
					else if constexpr (ymm_sized<T> && any_i8<S>) return _mm256_add_epi8(a, b);
					else static_assert(always_false_v<T>);
				}

				template<typename S, size_t N>
					requires (any_int<S> && sizeof(SIMD_Vector<S, N>) > 16)
				static SIMD_Vector<S, N> eval(op_sub, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32) return { sub(a.lo(), b.lo()), sub(a.hi(), b.hi()) };
					else if constexpr (ymm_sized<T> && any_i64<S>) return _mm256_sub_epi64(a, b);
					else if constexpr (ymm_sized<T> && any_i32<S>) return _mm256_sub_epi32(a, b);
					else if constexpr (ymm_sized<T> && any_i16<S>) return _mm256_sub_epi16(a, b);
					else if constexpr (ymm_sized<T> && any_i8<S>) return _mm256_sub_epi8(a, b);
					else static_assert(always_false_v<T>);
				}

				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_mul, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
					requires (sizeof(SIMD_Vector<S, N>) > 16 && any_int<S>)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32) return { mul(a.lo(), b.lo()), mul(a.hi(), b.hi()) };
					else if constexpr (ymm_sized<T> && any_i64<S>)
					{
						__m256i p1 = _mm256_mul_epu32(a, b); //alo*blo
						__m256i ahi = _mm256_srli_epi64(a, 32);
						__m256i bhi = _mm256_srli_epi64(b, 32);
						__m256i p2 = _mm256_slli_epi64(_mm256_mul_epu32(a, bhi), 32);
						__m256i p3 = _mm256_slli_epi64(_mm256_mul_epu32(b, ahi), 32);
						return _mm256_add_epi64(p3, _mm256_add_epi64(p1, p2));
					}
					else if constexpr (ymm_sized<T> && any_i32<S>) return _mm256_mullo_epi32(a, b);
					else if constexpr (ymm_sized<T> && any_i16<S>) return _mm256_mullo_epi16(a, b);
					else if constexpr (ymm_sized<T> && any_i8<S>)
					{
						using canon_t = std::conditional_t<std::is_unsigned_v<S>, uint16_t, int16_t>;
						return vcvt<S>(mul(vcvt<canon_t>(a), vcvt<canon_t>(b)));
					}
				}

				template<typename S, size_t N>
					requires (any_int<S> && sizeof(SIMD_Vector<S, N>) > 16)
				static SIMD_Vector<S, N> eval(op_or, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32) return { logic_or(a.lo(),b.lo()), logic_or(a.hi(),b.hi()) };
					else if constexpr (ymm_sized<T>) return _mm256_or_si256(a, b);
					else static_assert(always_false_v<T>);
				}
				template<typename S, size_t N>
					requires (any_int<S> && sizeof(SIMD_Vector<S, N>) > 16)
				static SIMD_Vector<S, N> eval(op_and, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32) return { logic_and(a.lo(),b.lo()), logic_and(a.hi(),b.hi()) };
					else if constexpr (ymm_sized<T>) return _mm256_and_si256(a, b);
					else static_assert(always_false_v<T>);
				}
				template<typename S, size_t N>
					requires (any_int<S> && sizeof(SIMD_Vector<S, N>) > 16)
				static SIMD_Vector<S, N> eval(op_xor, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32) return { logic_xor(a.lo(),b.lo()), logic_xor(a.hi(),b.hi()) };
					else if constexpr (ymm_sized<T>) return _mm256_xor_si256(a, b);
					else static_assert(always_false_v<T>);
				}
				template<typename S, size_t N>
					requires (any_int<S> && sizeof(SIMD_Vector<S, N>) > 16)
				static SIMD_Vector<S, N> eval(op_not, const SIMD_Vector<S, N>& a)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32) return { logic_not(a.lo()), logic_not(a.hi()) };
					else if constexpr (ymm_sized<T>) { __m256i u = _mm256_undefined_si256(); return _mm256_xor_si256(a, _mm256_cmpeq_epi32(u, u)); }
					else static_assert(always_false_v<T>);
				}

				template<typename S, size_t N, typename I>
					requires (concepts::any_int<S> && concepts::any_int<I>)
				static SIMD_Vector<S, N> eval(op_shl, const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& b)
				{
					using canon_t = same_size_uint_t<S>::type;
					using T = SIMD_Vector<S, N>;
					if constexpr (any_i16<S> || any_i8<S>) return vcvt<S>(shift_left(vcvt<uint32_t>(a), vcvt<uint32_t>(b)));
					else if constexpr (!std::is_same_v<I, canon_t>) return shift_left(a, vcvt<canon_t>(b));
					else if constexpr (sizeof(T) > 32) return { shift_left(a.lo(), b.lo()), shift_left(a.hi(),b.hi()) };
					else if constexpr (ymm_sized<T> && any_i64<S>) return _mm256_sllv_epi64(a, b);
					else if constexpr (ymm_sized<T> && any_i32<S>) return _mm256_sllv_epi32(a, b);
					else if constexpr (xmm_sized<T> && any_i64<S>) return _mm_sllv_epi64(a, b);
					else if constexpr (xmm_sized<T> && any_i32<S>) return _mm_sllv_epi32(a, b);
					else static_assert(always_false_v<T>);
				}
				template<typename S, size_t N, typename I>
					requires (concepts::any_int<S>&& concepts::any_int<I>)
				static SIMD_Vector<S, N> eval(op_shr, const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& b)
				{
					using canon_t = same_size_uint_t<S>::type;
					using T = SIMD_Vector<S, N>;
					if constexpr (any_i16<S> || any_i8<S>) return vcvt<S>(shift_right(vcvt<uint32_t>(a), vcvt<uint32_t>(b)));
					else if constexpr (!std::is_same_v<I, canon_t>) return shift_right(a, vcvt<canon_t>(b));
					else if constexpr (sizeof(T) > 32) return { shift_right(a.lo(), b.lo()), shift_right(a.hi(),b.hi()) };
					else if constexpr (ymm_sized<T> && any_i64<S>) return _mm256_srlv_epi64(a, b);
					else if constexpr (ymm_sized<T> && any_i32<S>) return _mm256_srlv_epi32(a, b);
					else if constexpr (xmm_sized<T> && any_i64<S>) return _mm_srlv_epi64(a, b);
					else if constexpr (xmm_sized<T> && any_i32<S>) return _mm_srlv_epi32(a, b);
					else static_assert(always_false_v<T>);
				}

				template<typename S, size_t N>
				requires (any_int<S> && sizeof(SIMD_Vector<S,N>) > 16)
				static SIMD_Vector<S, N> eval(op_mask_mov, const SIMD_Vector<S, N>& ifBitClear, const SIMD_Mask<S,N>& mask, const SIMD_Vector<S, N>& ifBitSet)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32) return { mask_mov(ifBitClear.lo(), mask.lo(), ifBitSet.lo()), mask_mov(ifBitClear.hi(), mask.hi(), ifBitSet.hi()) };
					else if constexpr (ymm_sized<T>)
					{
						auto vecm = vcast<SIMD_Vector<S, N>>(mask.as_vector());
						return _mm256_blendv_epi8(ifBitClear, ifBitSet, vecm);
					}
					else static_assert(always_false_v<T>);
				}

				template<typename S, size_t N>
					requires (any_int<S> && sizeof(SIMD_Vector<S, N>) > 16)
				static SIMD_Vector<S, N> eval(op_min, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32) return { min(a.lo(), b.lo()), min(a.hi(),b.hi()) };
					else if constexpr (ymm_sized<T> && is_i8<S>) return _mm256_min_epi8(a, b);
					else if constexpr (ymm_sized<T> && is_u8<S>) return _mm256_min_epu8(a, b);
					else if constexpr (ymm_sized<T> && is_i16<S>) return _mm256_min_epi16(a, b);
					else if constexpr (ymm_sized<T> && is_u16<S>) return _mm256_min_epu16(a, b);
					else if constexpr (ymm_sized<T> && is_i32<S>) return _mm256_min_epi32(a, b);
					else if constexpr (ymm_sized<T> && is_u32<S>) return _mm256_min_epu32(a, b);
					else if constexpr (ymm_sized<T> && any_i64<S>) return mask_mov(a, b < a, b);
					else static_assert(always_false_v<T>);
				}
				template<typename S, size_t N>
					requires (any_int<S> && sizeof(SIMD_Vector<S, N>) > 16)
				static SIMD_Vector<S, N> eval(op_max, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32) return { max(a.lo(), b.lo()), max(a.hi(),b.hi()) };
					else if constexpr (ymm_sized<T> && is_i8<S>) return _mm256_max_epi8(a, b);
					else if constexpr (ymm_sized<T> && is_u8<S>) return _mm256_max_epu8(a, b);
					else if constexpr (ymm_sized<T> && is_i16<S>) return _mm256_max_epi16(a, b);
					else if constexpr (ymm_sized<T> && is_u16<S>) return _mm256_max_epu16(a, b);
					else if constexpr (ymm_sized<T> && is_i32<S>) return _mm256_max_epi32(a, b);
					else if constexpr (ymm_sized<T> && is_u32<S>) return _mm256_max_epu32(a, b);
					else if constexpr (ymm_sized<T> && any_i64<S>) return mask_mov(a, b > a, b);
					else static_assert(always_false_v<T>);
				}
				template<typename S, size_t N>
					requires (any_int<S> && sizeof(SIMD_Vector<S, N>) > 16)
				static SIMD_Vector<S, N> eval(op_unpacklo, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32) return { unpacklo(a.lo(),b.lo()), unpacklo(a.hi(),b.hi()) };
					else if constexpr (ymm_sized<T> && any_i64<S>) return _mm256_unpacklo_epi64(a, b);
					else if constexpr (ymm_sized<T> && any_i32<S>) return _mm256_unpacklo_epi32(a, b);
					else if constexpr (ymm_sized<T> && any_i16<S>) return _mm256_unpacklo_epi16(a, b);
					else if constexpr (ymm_sized<T> && any_i8<S>) return _mm256_unpacklo_epi8(a, b);
					else static_assert(always_false_v<T>);
				}
				template<typename S, size_t N>
					requires (any_int<S> && sizeof(SIMD_Vector<S, N>) > 16)
				static SIMD_Vector<S, N> eval(op_unpackhi, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32) return { unpackhi(a.lo(),b.lo()), unpackhi(a.hi(),b.hi()) };
					else if constexpr (ymm_sized<T> && any_i64<S>) return _mm256_unpackhi_epi64(a, b);
					else if constexpr (ymm_sized<T> && any_i32<S>) return _mm256_unpackhi_epi32(a, b);
					else if constexpr (ymm_sized<T> && any_i16<S>) return _mm256_unpackhi_epi16(a, b);
					else if constexpr (ymm_sized<T> && any_i8<S>) return _mm256_unpackhi_epi8(a, b);
					else static_assert(always_false_v<T>);
				}

				template<typename S, size_t N>
				requires (sizeof(SIMD_Vector<S, N>) > 16 && any_int<S>)
				static SIMD_Mask<S,N> eval(op_cmpeq, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					using M = SIMD_Mask<S, N>;
					if constexpr (sizeof(T) > 32) return { cmp_equal(a.lo(),b.lo()), cmp_equal(a.hi(),b.hi()) };
					else if constexpr (ymm_sized<T> && any_i64<S>) return _mm256_cmpeq_epi64(a, b);
					else if constexpr (ymm_sized<T> && any_i32<S>) return _mm256_cmpeq_epi32(a, b);
					else if constexpr (ymm_sized<T> && any_i16<S>) return _mm256_cmpeq_epi16(a, b);
					else if constexpr (ymm_sized<T> && any_i8<S>) return _mm256_cmpeq_epi8(a, b);
					else static_assert(always_false_v<T>);
				}

				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 16 && any_int<S>)
				static SIMD_Mask<S,N> eval(op_cmpneq, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					return ~cmp_equal(a, b);
				}

				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 16 && any_int<S>)
				static SIMD_Mask<S,N> eval(op_cmpgt, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					using namespace concepts;
					using T = SIMD_Vector<S, N>;
					using I = same_size_int_t<S>::type;
					if constexpr (std::is_unsigned_v<S>)
					{
						S xorv = S(1) << (sizeof(S) * 8 - 1); //xor with 0x800..000 before comparison
						return cmp_greater(vcvt<I>(a) ^ xorv, vcvt<I>(b) ^ xorv);
					}
					else if constexpr (sizeof(T) > 32) return { cmp_greater(a.lo(),b.lo()), cmp_greater(a.hi(),b.hi()) };
					else if constexpr (ymm_sized<T> && is_i64<S>) return _mm256_cmpgt_epi64(a, b);
					else if constexpr (ymm_sized<T> && is_i32<S>) return _mm256_cmpgt_epi32(a, b);
					else if constexpr (ymm_sized<T> && is_i16<S>) return _mm256_cmpgt_epi16(a, b);
					else if constexpr (ymm_sized<T> && is_i8<S>) return _mm256_cmpgt_epi8(a, b);
					else static_assert(always_false_v<T>);
				}
				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 16 && any_int<S>)
				static SIMD_Mask<S,N> eval(op_cmplt, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					return cmp_greater(b, a); //flip arguments
				}

				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 16 && any_int<S>)
				static SIMD_Mask<S,N> eval(op_cmple, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					return ~cmp_greater(a, b);
				}

				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) > 16 && any_int<S>)
				static SIMD_Mask<S,N> eval(op_cmpge, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					return ~cmp_less(a, b);
				}
				/*
				template<typename S, size_t N>
				requires (any_small_int<S> && sizeof(SIMD_Vector<S,N>) >= 17) //|| (FS.has(SSSE3) && any_i16<S>))
				static SIMD_Mask<S,N>::UintT eval(op_maskvec2uint, const SIMD_Vector<S, N>& a)
				{
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(T) > 32) return { vec2mask(a.lo()),vec2mask(a.hi()) };
					else if constexpr (ymm_sized<T> && any_i8<S>) return _mm256_movemask_epi8(a);
					else if constexpr (FS.has(SSSE3) && ymm_sized<T> && any_i16<S>)
					{
						//AVX2 has no movemask_epi16 intrinsic, so we need to fall back to older shuffle+movemask
						//only care about upper bytes of each 16-bit word. 
						//concentrate upper bytes of each word into lower (x) or upper (y) half of 128-bit vector
						//then merge them and return 16 bit mask
						//cvt to 32 bits + domain cross + movemask_ps may be faster if domain penalties don't apply (Skylake and newer?)
						__m128i x = _mm_shuffle_epi8(a.lo(), _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1));
						__m128i y = _mm_shuffle_epi8(a.hi(), _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 1, 3, 5, 7, 9, 11, 13, 15));
						return _mm_movemask_epi8(_mm_or_si128(x, y));
					}
					else static_assert(always_false_v<T>);
				}
				
				template<typename S, size_t N>
					requires (sizeof(SIMD_Vector<S, N>) >= 17)
				static SIMD_Vector<S,N> eval(op_uint2maskvec<S,N>, const SIMD_Mask<S,N>::UintT& a)
				{
					using T = SIMD_Vector<S, N>;
					SIMD_Vector<S, N> ret;
					if constexpr (sizeof(T) > 32) return { mask2vec<S>(a.lo()),mask2vec<S>(a.hi()) };
					else if constexpr (ymm_sized<T> && sizeof(S) == 8)
					{
						__m256i broadcasted = _mm256_set1_epi64x(a);
						__m256i x = _mm256_andnot_si256(broadcasted, _mm256_setr_epi64x(1, 2, 4, 8));
						auto y = _mm256_cmpeq_epi64(x, _mm256_set1_epi64x(0));
						memcpy(&ret, &y, std::min(sizeof(ret), sizeof(y)));
						return ret;
					}
					else if constexpr (ymm_sized<T> && sizeof(S) == 4)
					{
						__m256i broadcasted = _mm256_set1_epi32(a);
						__m256i x = _mm256_andnot_si256(broadcasted, _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128));
						auto y = _mm256_cmpeq_epi32(x, _mm256_set1_epi32(0));
						memcpy(&ret, &y, std::min(sizeof(ret), sizeof(y)));
						return ret;
					}
					else if constexpr (ymm_sized<T> && sizeof(S) == 2)
					{
						__m256i broadcasted = _mm256_set1_epi16(a);
						__m256i x = _mm256_andnot_si256(broadcasted, _mm256_setr_epi16(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768));
						auto y = _mm256_cmpeq_epi16(x, _mm256_set1_epi16(0));
						memcpy(&ret, &y, std::min(sizeof(ret), sizeof(y)));
						return ret;
					}
					else if constexpr (ymm_sized<T> && any_i8<S>)
					{
						//https://stackoverflow.com/questions/21622212/how-to-perform-the-inverse-of-mm256-movemask-epi8-vpmovmskb
						__m256i vmask(_mm256_set1_epi32(a));
						const __m256i shuffle(_mm256_setr_epi64x(0x0000000000000000,
							0x0101010101010101, 0x0202020202020202, 0x0303030303030303));
						vmask = _mm256_shuffle_epi8(vmask, shuffle);
						const __m256i bit_mask(_mm256_set1_epi64x(0x7fbfdfeff7fbfdfe));
						vmask = _mm256_or_si256(vmask, bit_mask);
						auto y = _mm256_cmpeq_epi8(vmask, _mm256_set1_epi64x(-1));
						memcpy(&ret, &y, std::min(sizeof(ret), sizeof(y)));
						return ret;
					}
					else static_assert(always_false_v<T>);
				}
				*/
				template<typename S, size_t N, typename I>
					requires (sizeof(S) >= 4 && concepts::any_int<I> && sizeof(SIMD_Vector<S, N>) >= 17)
				static SIMD_Vector<S, N> eval(op_permx, const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& ind)
				{
					using namespace concepts;
					using canon_t = typename same_size_uint_t<S>::type;
					using T = SIMD_Vector<S, N>;
					if constexpr (sizeof(I) != sizeof(S)) return permx(a, vcvt<canon_t>(ind));
					else if constexpr (sizeof(T) > 32)
					{
						auto alo = a.lo();
						auto ahi = a.hi();
						return { permx2(alo, ahi, ind.lo()), permx2(alo, ahi, ind.hi()) };
					}
					else if constexpr (ymm_sized<T> && any_i32<S>) return _mm256_permutevar8x32_epi32(a, ind);
					else if constexpr (ymm_sized<T> && is_f32<S>) return _mm256_permutevar8x32_ps(a, ind);
					else if constexpr (ymm_sized<T> && sizeof(S) == 8)
					{
						//TODO: perhaps AVX _mm256_permutevar_pd + blend is better despite cross-domain and intralane limitations
						//TODO: check if this works
						using X = SIMD_Vector<int32_t, N * 2>;
						X i32_a = vcast<X>(a); //reinterpret a as i32's
						X i32_ind = vcast<X>(ind); //and ind too
						X dup;
						for (size_t i = 0; i < N * 2; ++i) dup[i] = i % 2;
						return vcast<T>(permx(i32_a, i32_ind << 1 | dup)); //emulate 64-bit permute via 32-bits. I.e. permx(a, i64x4(0, 3, 1, 2)) will become permx(i32x_(a), i32x8(0, 1, 6, 7, 2, 3, 4, 5))
					}
				}

				template<typename S, size_t N>
					requires (sizeof(S) == 4 && sizeof(SIMD_Vector<S, N>) >= 17)
				static SIMD_Vector<S, N> eval(op_compress, const SIMD_Mask<S,N>& mask, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& src = 0)
				{
					using T = SIMD_Vector<S, N>;
					using canon_t = same_size_int_t<S>::type;
					if constexpr (sizeof(T) > 32)
					{
						//TODO: check if src is passed properly. Also, can write it out at the end
						T ret = src;
						auto cl = compress(mask.lo(), a.lo(), src.lo());
						auto ch = compress(mask.hi(), a.hi(), src.lo()); //doesn't matter which src, since that's useless anyway
						size_t popcnt_lo = std::popcount(mask.lo().as_uint()); //TODO: _mm_popcnt_u* if is supported?
						size_t popcnt_hi = std::popcount(mask.hi().as_uint());

						static_assert(sizeof(S) == 4);
						float* p = (float*)&ret;
						store(cl, p);
						SIMD_Mask<canon_t, N / 2> cm = (uint64_t(1) << popcnt_hi) - 1;
						store(ch, p + popcnt_lo, cm); //don't overwrite src remains
						//_mm256_maskstore_ps(p + popcnt_lo, mask2vec<int32_t, N / 2>(cm), vreinterpret<__m256>(ch)); //don't overwrite src remains
						return ret;
					}
					else if constexpr (ymm_sized<T> && sizeof(S) == 4)
					{
						auto permx_ind = vcvt<canon_t>(SIMD_Vector<int8_t, N>(_mm_loadu_si64(&tables::compress_to_permx8[mask.as_uint()])));
						auto tmp = permx(a, permx_ind); //permx_ind is setup in such a way that is can be used both as index register and blend mask without extra conversions
						if constexpr (is_f32<S>) return _mm256_blendv_ps(tmp, src, _mm256_castsi256_ps(permx_ind));
						else if constexpr (any_i32<S>) return _mm256_blendv_epi8(tmp, src, permx_ind);
						else static_assert(always_false_v<T>);
					}
					else static_assert(always_false_v<T>);
				}

				template<typename S, size_t N>
					requires (any_i32<S> || any_i64<S>)
				static void eval(op_store, SIMD_Vector<S, N> vec, void* p, const SIMD_Mask<S,N>& mask = SIMD_Mask<S,N>::AllOnes)
				{
					using T = SIMD_Vector<S, N>;
					using I = same_size_int_t<S>::type;
					I* sp = reinterpret_cast<I*>(p);
					if constexpr (sizeof(T) > 32) 
					{
						store(vec.lo(), sp, mask.lo());
						store(vec.hi(), sp + N/2, mask.hi());
					}
					else
					{
						auto vec_mask = vcast<SIMD_Vector<I,N>>(mask.as_vector()); 
						if constexpr (ymm_sized<T> && any_i64<S>) _mm256_maskstore_epi64(sp, vec_mask, vec);
						else if constexpr (ymm_sized<T> && any_i32<S>) _mm256_maskstore_epi32(sp, vec_mask, vec);
						else if constexpr (xmm_sized<T> && any_i64<S>) _mm_maskstore_epi64(sp, vec_mask, vec);
						else if constexpr (xmm_sized<T> && any_i32<S>) _mm_maskstore_epi32(sp, vec_mask, vec);
						else static_assert(always_false_v<T>);
					}
				}

				template<typename S, size_t N>
					requires (any_i32<S> || any_i64<S>)
				static SIMD_Vector<S,N> eval(op_load<S, N>, const void* p, const SIMD_Mask<S,N>& mask = SIMD_Mask<S,N>::AllOnes, const SIMD_Vector<S, N>& src = 0)
				{
					using T = SIMD_Vector<S, N>;
					using I = same_size_int_t<S>::type;
					const I* sp = reinterpret_cast<const I*>(p);
					if constexpr (sizeof(T) > 32) return { load<S,N / 2>(sp, mask.lo(), src.lo()), load<S,N / 2>(sp + N / 2, mask.hi(),src.hi()) };
					else
					{
						auto vec_mask = vcast<SIMD_Vector<I, N>>(mask.as_vector());
						if constexpr (ymm_sized<T> && any_i64<S>) return _mm256_maskload_epi64(sp, vec_mask);
						else if constexpr (ymm_sized<T> && any_i32<S>) return _mm256_maskload_epi32(sp, vec_mask);
						else if constexpr (xmm_sized<T> && any_i64<S>) return _mm_maskload_epi64(sp, vec_mask);
						else if constexpr (xmm_sized<T> && any_i32<S>) return _mm_maskload_epi32(sp, vec_mask);
						else static_assert(always_false_v<T>);
					}
				}
			};
		}
	}
}