#pragma once
#include "Vec.h"
#include <string>
#include <immintrin.h>

template <typename InterpolandType, typename InterpolationValueType>
__forceinline InterpolandType lerp(const InterpolandType& start, const InterpolandType& end, const InterpolationValueType& amount)
{
	return start + (end - start) * amount;
}

__forceinline float inverse_lerp(float from, float to, float value)
{
	return (value - from) / (to - from);
}

__forceinline float32x16 inverse_lerp(float32x16 from, float32x16 to, float32x16 value)
{
	return (value - from) / (to - from);
}

__forceinline Vec4f getFaceNormalForTriangle(const Vec4f& v0, const Vec4f& v1, const Vec4f& v2)
{
	Vec4f n = (v2 - v0).cross3d(v1 - v0);
	n.w = 0;
	return n / n.len();
}
__forceinline Vec4_f32x16 getFaceNormalsForTriangles16(const Vec4_f32x16& v0, const Vec4_f32x16& v1, const Vec4_f32x16& v2)
{
	Vec4_f32x16 n = (v2 - v0).cross3d(v1 - v0);
	return n / n.len3d();
}

inline std::string toThousandsSeparatedString(int64_t value, std::string sep = ",")
{
	if (value == 0) return "0";
	uint64_t positive = abs(value);
	std::string ret;
	while (positive > 0)
	{
		int mod = positive % 1000;
		std::string modStr = std::to_string(mod);

		if (positive > 999) //prepend the resulting string part with zeros if it's not the leftmost part
		{
			if (mod < 10) modStr = "00" + modStr;
			else if (mod < 100) modStr = "0" + modStr;
		}

		ret = modStr + sep + ret;

		positive /= 1000;
	}

	ret.pop_back(); //due to how algorithm works, there's always an unnecessary trailing separator after last digit. Just remove it
	if (value < 0) ret = "-" + ret;
	return ret;
}

#ifdef VS_CLANG
__forceinline __m512i _mm512_setr_epi16(int16_t i0, int16_t i1, int16_t i2, int16_t i3, int16_t i4, int16_t i5, int16_t i6, int16_t i7, int16_t i8, int16_t i9, int16_t i10, int16_t i11, int16_t i12, int16_t i13, int16_t i14, int16_t i15, int16_t i16, int16_t i17, int16_t i18, int16_t i19, int16_t i20, int16_t i21, int16_t i22, int16_t i23, int16_t i24, int16_t i25, int16_t i26, int16_t i27, int16_t i28, int16_t i29, int16_t i30, int16_t i31)
{
	return _mm512_set_epi16(i31, i30, i29, i28, i27, i26, i25, i24, i23, i22, i21, i20, i19, i18, i17, i16, i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0);
}
#endif

__forceinline void interleaved_ph_to_ps(__m512i inp, __m512& retLow, __m512& retHigh)
{
	int32x16 x = _mm512_permutexvar_epi16(_mm512_setr_epi16(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31), inp);
	retLow = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(x, 0));
	retHigh = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(x, 1));
}

__forceinline void interleaved_ph_to_ps(__m512i inp, float32x16& retLow, float32x16& retHigh)
{
	interleaved_ph_to_ps(inp, retLow.zmm, retHigh.zmm);
}

//Returns 64 bit mask that has all bits of m duplicated four times, i.e: mask with bits 0123456789abcd will become 000011112222...dddd
__forceinline __mmask64 duplicate_mmask_bits_16_to_64(__mmask16 m)
{
	__m512i a = _mm512_movm_epi32(m);
	return _mm512_movepi8_mask(a);
}


__forceinline __m512i helper_mm512_setr_epi8(int8_t i0, int8_t i1, int8_t i2, int8_t i3, int8_t i4, int8_t i5, int8_t i6, int8_t i7, int8_t i8, int8_t i9, int8_t i10, int8_t i11, int8_t i12, int8_t i13, int8_t i14, int8_t i15, int8_t i16, int8_t i17, int8_t i18, int8_t i19, int8_t i20, int8_t i21, int8_t i22, int8_t i23, int8_t i24, int8_t i25, int8_t i26, int8_t i27, int8_t i28, int8_t i29, int8_t i30, int8_t i31, int8_t i32, int8_t i33, int8_t i34, int8_t i35, int8_t i36, int8_t i37, int8_t i38, int8_t i39, int8_t i40, int8_t i41, int8_t i42, int8_t i43, int8_t i44, int8_t i45, int8_t i46, int8_t i47, int8_t i48, int8_t i49, int8_t i50, int8_t i51, int8_t i52, int8_t i53, int8_t i54, int8_t i55, int8_t i56, int8_t i57, int8_t i58, int8_t i59, int8_t i60, int8_t i61, int8_t i62, int8_t i63)
{
	return _mm512_set_epi8(i63, i62, i61, i60, i59, i58, i57, i56, i55, i54, i53, i52, i51, i50, i49, i48, i47, i46, i45, i44, i43, i42, i41, i40, i39, i38, i37, i36, i35, i34, i33, i32, i31, i30, i29, i28, i27, i26, i25, i24, i23, i22, i21, i20, i19, i18, i17, i16, i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0);
}

//Returns 64 bit mask that has all bits of m duplicated three times, i.e: mask with bits 0123456789abcd will become 000111222...ddd. Returned mask's bits 48-63 are set to zero
__forceinline __mmask64 duplicate_mmask_bits_16_to_48(__mmask16 m)
{
	__m512i a = _mm512_movm_epi32(m);
	__m512i b = _mm512_maskz_compress_epi8(0x7777777777777777, a);
	return _mm512_movepi8_mask(b);
	/*
	__m128i a = _mm_movm_epi8(m);
	__m512i b = _mm512_maskz_permutexvar_epi8(0x0000FFFFFFFFFFFF, helper_mm512_setr_epi8(0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), _mm512_castsi128_si512(a));
	return _mm512_movepi8_mask(b);*/
}

//Returns 32 bit mask that has all bits of m duplicated four times, i.e: mask with bits 0123456789abcd will become 001122...dd
__forceinline __mmask32 duplicate_mmask_bits_16_to_32(__mmask16 m)
{
	__m256i a = _mm256_movm_epi16(m);
	return _mm256_movepi8_mask(a);
}

struct FixedPoint
{
	static inline constexpr int64_t TOTAL_BITS = 21;
	static inline constexpr int64_t FRAC_BITS = 4;
	static inline constexpr int64_t INTEGER_BITS = TOTAL_BITS - FRAC_BITS - 1;
	static inline constexpr int64_t MULT = 1 << FRAC_BITS;
	static inline constexpr int64_t UNSIGNED_PART_MASK = (1 << (TOTAL_BITS - 1)) - 1;
	static inline constexpr int64_t SIGN_MASK = (1 << (TOTAL_BITS - 1));
	static inline constexpr int64_t FULL_MASK = (1 << (TOTAL_BITS)) - 1;

	static uint64_t encode_3pack(double a, double b, double c)
	{
		int64_t ua = std::round(a * MULT);
		int64_t ub = std::round(b * MULT);
		int64_t uc = std::round(c * MULT);

		int64_t signA = a >= 0 ? 0 : SIGN_MASK;
		int64_t signB = b >= 0 ? 0 : SIGN_MASK;
		int64_t signC = c >= 0 ? 0 : SIGN_MASK;

		int64_t magA = abs(ua) & UNSIGNED_PART_MASK;
		int64_t magB = abs(ub) & UNSIGNED_PART_MASK;
		int64_t magC = abs(uc) & UNSIGNED_PART_MASK;

		int64_t p1 = magA | signA;
		int64_t p2 = magB | signB;
		int64_t p3 = magC | signC;

		return (p1 | (p2 << TOTAL_BITS) | (p3 << (TOTAL_BITS * 2))) & ~0x8000000000000000; //force uppermost bit to 0 to avoid garbage in there
	}

	static std::array<__m512, 3> decode_3pack(__m512i u64lo, __m512i u64hi)
	{
		u64lo = _mm512_and_epi64(u64lo, _mm512_set1_epi64(~0x8000000000000000));
		u64hi = _mm512_and_epi64(u64hi, _mm512_set1_epi64(~0x8000000000000000));
		std::array<__m512, 3> ret;
		for (int i = 0; i < 3; ++i)
		{
			__m512i lo = _mm512_and_si512(_mm512_srlv_epi64(u64lo, _mm512_set1_epi64(TOTAL_BITS * i)), _mm512_set1_epi64(FULL_MASK));
			__m512i hi = _mm512_and_si512(_mm512_srlv_epi64(u64hi, _mm512_set1_epi64(TOTAL_BITS * i)), _mm512_set1_epi64(FULL_MASK));
			__m256i lo32 = _mm512_cvtepi64_epi32(lo);
			__m256i hi32 = _mm512_cvtepi64_epi32(hi);
			__m512i united = _mm512_inserti32x8(_mm512_castsi256_si512(lo32), hi32, 1);
			__m512i sign = _mm512_and_si512(united, _mm512_set1_epi32(SIGN_MASK));
			__m512i mag = _mm512_and_si512(united, _mm512_set1_epi32(UNSIGNED_PART_MASK));

			//if sign bit is set, negate magnitude
			__m512i composed = _mm512_mask_sub_epi32(mag, _mm512_cmpneq_epi32_mask(sign, _mm512_setzero_si512()), _mm512_set1_epi32(0), mag);
			__m512 ps = _mm512_cvtepi32_ps(composed);
			ret[i] = _mm512_mul_ps(ps, _mm512_set1_ps(1.0 / MULT));
		}
		return ret;
	}
};