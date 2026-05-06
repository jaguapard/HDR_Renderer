#pragma once
#include "Vec.h"
#include <string>

template <typename T>
__forceinline T lerp(const T& start, const T& end, float amount)
{
	return start + (end - start) * amount;
}

__forceinline float32x16 lerp(const float32x16& start, const float32x16& end, float32x16 amount)
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
	//return _mm512_cmpeq_epi8_mask(a, _mm512_setzero_si512());
}
//Returns 32 bit mask that has all bits of m duplicated four times, i.e: mask with bits 0123456789abcd will become 001122...dd
__forceinline __mmask32 duplicate_mmask_bits_16_to_32(__mmask16 m)
{
	__m256i a = _mm256_movm_epi16(m);
	return _mm256_movepi8_mask(a);
	//return _mm512_cmpeq_epi8_mask(a, _mm512_setzero_si512());
}

//Loads 16 16-byte elements from base using mask and ind, then transposes and stores them into 4 SoA vectors.
//If an element is masked off, it is not loaded and it's returned value is undefined.
//A, B, C and D must be 64-byte types.
//for i in [0,15]:
//    if mask[i]:
//        tmp = load contigious 16 bytes starting at byte size_t(base) + ind[i]*16
//        ret1[i*4..i*4+3] = tmp[0..3]
//        ret2[i*4..i*4+3] = tmp[4..7]
//        ret3[i*4..i*4+3] = tmp[8..11]
//        ret4[i*4..i*4+3] = tmp[12..15]
template <typename A, typename B, typename C, typename D>
	requires (sizeof(A) == 64 && sizeof(B) == 64 && sizeof(C) == 64 && sizeof(D) == 64)
__forceinline void masked_16x4aos_to_4x16soa_gather_and_transpose(int32x16 ind, Mask16 mask, const void* base, A& ret1, B& ret2, C& ret3, D& ret4)
{
	ind *= 4;
	float r0[16], r1[16], r2[16], r3[16];
	uint32_t* uind = (uint32_t*)&ind;
	const float* fp = (const float*)base;
	__mmask32 m = duplicate_mmask_bits_16_to_32(mask);
	for (int i = 0; i < 16; i += 4)
	{
		__m128 v0 = _mm_castpd_ps(_mm_maskz_loadu_pd(m >> (i * 2), fp + uind[i]));
		__m128 v1 = _mm_castpd_ps(_mm_maskz_loadu_pd(m >> (i * 2 + 2), fp + uind[i + 1]));
		__m128 v2 = _mm_castpd_ps(_mm_maskz_loadu_pd(m >> (i * 2 + 4), fp + uind[i + 2]));
		__m128 v3 = _mm_castpd_ps(_mm_maskz_loadu_pd(m >> (i * 2 + 6), fp + uind[i + 3]));

		_mm_storeu_ps(&r0[i], v0); //r0 = abcd0,abcd4,abcd8,abcd12
		_mm_storeu_ps(&r1[i], v1); //r1 = abcd1,abcd5,abcd9,abcd13
		_mm_storeu_ps(&r2[i], v2); //r2 = abcd2,abcd6,abcd10,abcd14
		_mm_storeu_ps(&r3[i], v3); //r3 = abcd3,abcd7,abcd11,abcd15
	}

	__m512 aabb01 = _mm512_unpacklo_ps(_mm512_loadu_ps(r0), _mm512_loadu_ps(r1));
	__m512 aabb23 = _mm512_unpacklo_ps(_mm512_loadu_ps(r2), _mm512_loadu_ps(r3));
	__m512 ccdd01 = _mm512_unpackhi_ps(_mm512_loadu_ps(r0), _mm512_loadu_ps(r1));
	__m512 ccdd23 = _mm512_unpackhi_ps(_mm512_loadu_ps(r2), _mm512_loadu_ps(r3));
	_mm512_storeu_pd(&ret1, _mm512_unpacklo_pd(_mm512_castps_pd(aabb01), _mm512_castps_pd(aabb23)));
	_mm512_storeu_pd(&ret2, _mm512_unpackhi_pd(_mm512_castps_pd(aabb01), _mm512_castps_pd(aabb23)));
	_mm512_storeu_pd(&ret3, _mm512_unpacklo_pd(_mm512_castps_pd(ccdd01), _mm512_castps_pd(ccdd23)));
	_mm512_storeu_pd(&ret4, _mm512_unpackhi_pd(_mm512_castps_pd(ccdd01), _mm512_castps_pd(ccdd23)));
}