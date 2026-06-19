#pragma once
#include "Vec.h"
#include <string>
#include <immintrin.h>

template <typename InterpolandType, typename InterpolationValueType>
__forceinline InterpolandType lerp(const InterpolandType& start, const InterpolandType& end, const InterpolationValueType& amount)
{
	return start + (end - start) * amount;
}

template <typename InterpolandType>
__forceinline float inverse_lerp(const InterpolandType& from, const InterpolandType& to, const InterpolandType& value)
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

__forceinline void interleaved_ph_to_ps(__m512i inp, float32x16& retLow, float32x16& retHigh)
{
	u16x32 x = permx(u16x32(inp), u16x32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31));
	retLow = vcvt_fp16_fp32(x.lo());
	retHigh = vcvt_fp16_fp32(x.hi());
}

//Returns 64 bit mask that has all bits of m duplicated four times, i.e: mask with bits 0123456789abcd will become 000011112222...dddd
__forceinline SIMD_BitMask<64> duplicate_mmask_bits_16_to_64(SIMD_BitMask<16> m)
{
	i32x16 a = mask2vec<int, 16>(m);
	return vec2mask(vcast<u8x64>(a));
}


__forceinline __m512i helper_mm512_setr_epi8(int8_t i0, int8_t i1, int8_t i2, int8_t i3, int8_t i4, int8_t i5, int8_t i6, int8_t i7, int8_t i8, int8_t i9, int8_t i10, int8_t i11, int8_t i12, int8_t i13, int8_t i14, int8_t i15, int8_t i16, int8_t i17, int8_t i18, int8_t i19, int8_t i20, int8_t i21, int8_t i22, int8_t i23, int8_t i24, int8_t i25, int8_t i26, int8_t i27, int8_t i28, int8_t i29, int8_t i30, int8_t i31, int8_t i32, int8_t i33, int8_t i34, int8_t i35, int8_t i36, int8_t i37, int8_t i38, int8_t i39, int8_t i40, int8_t i41, int8_t i42, int8_t i43, int8_t i44, int8_t i45, int8_t i46, int8_t i47, int8_t i48, int8_t i49, int8_t i50, int8_t i51, int8_t i52, int8_t i53, int8_t i54, int8_t i55, int8_t i56, int8_t i57, int8_t i58, int8_t i59, int8_t i60, int8_t i61, int8_t i62, int8_t i63)
{
	return _mm512_set_epi8(i63, i62, i61, i60, i59, i58, i57, i56, i55, i54, i53, i52, i51, i50, i49, i48, i47, i46, i45, i44, i43, i42, i41, i40, i39, i38, i37, i36, i35, i34, i33, i32, i31, i30, i29, i28, i27, i26, i25, i24, i23, i22, i21, i20, i19, i18, i17, i16, i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0);
}

//Returns 64 bit mask that has all bits of m duplicated three times, i.e: mask with bits 0123456789abcdef will become 000111222...eeefff. Returned mask's bits 48-63 are set to zero
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

//Returns 32 bit mask that has all bits of m duplicated four times, i.e: mask with bits 0123456789abcdef will become 001122...ddeeff
__forceinline __mmask32 duplicate_mmask_bits_16_to_32(__mmask16 m)
{
	i16x16 a = mask2vec<uint16_t, 16>(m);
	return vec2mask(vcast<u8x32>(a));
}

/**
 * Removes duplicate active values from a 16-lane vector and packs the unique values contiguously.
 *
 * Only lanes enabled by `activeMask` participate in deduplication. Inactive lanes are replaced
 * with `invalidValue` before conflict detection so that undefined data in masked-off lanes cannot
 * suppress valid entries.
 *
 * The output vector contains only the first occurrence of each active value, compressed toward
 * the lower lanes. Only the first `outUniqueCount` lanes are defined; remaining lanes are undefined.
 *
 * @param inputValues     Source values to deduplicate.
 * @param invalidValue    Value guaranteed not to appear in active lanes. Used to sanitize inactive lanes.
 * @param activeMask      Mask identifying which input lanes are valid.
 * @param outUniqueValues Receives the compressed unique values.
 * @param outUniqueCount  Optional. Receives the number of unique active values.
 * @param outUniqueMask   Optional. Receives a mask of lanes corresponding to first occurrences.
 */
__forceinline void deduplicate_epi32x16(int32x16 inputValues, int32_t invalidValue, Mask16 activeMask, int32x16& outUniqueValues, uint32_t* outUniqueCount = nullptr, Mask16* outUniqueMask = nullptr)
{
	//Replace inactive lanes with guaranteed-invalid values so masked-off garbage
	//cannot participate in conflict detection.
	int32x16 cleaned = mask_mov(i32x16(invalidValue), activeMask, inputValues);
	int32x16 conflicts = conflict(cleaned);
	Mask16 uniqueMask = activeMask & (conflicts == 0); 	//Keep only active lanes that have no prior occurrence.

	outUniqueValues = compress(uniqueMask, cleaned);
	if (outUniqueCount) *outUniqueCount = std::popcount((uint32_t)uniqueMask);
	if (outUniqueMask) *outUniqueMask = uniqueMask;
}

/**
Removes duplicate active values from a 16-lane vector and packs the unique values contiguously.
Does NOT check for NANs or infinities, operates only on bitwise representations!
Check deduplicate_epi32x16 for more details
*/
__forceinline void deduplicate_ps512(float32x16 inputValues, float invalidValue, Mask16 activeMask, float32x16& outUniqueValues, uint32_t* outUniqueCount = nullptr, Mask16* outUniqueMask = nullptr)
{
	int32x16 unique;
	deduplicate_epi32x16(_mm512_castps_si512(inputValues), std::bit_cast<int32_t>(invalidValue), activeMask, unique, outUniqueCount, outUniqueMask);
	outUniqueValues = _mm512_castsi512_ps(unique);
}

/*
Builds 512-bit vector from 128-bit chunks.
ret[0..127] = xmm0
ret[128..255] = xmm1
ret[256..383] = xmm2
ret[384..511] = xmm3
*/
__forceinline __m512 xmm_x4_to_zmm(__m128 xmm0, __m128 xmm1, __m128 xmm2, __m128 xmm3)
{
	__m256 ymm0 = _mm256_insertf128_ps(_mm256_castps128_ps256(xmm0), xmm1, 1);
	__m256 ymm1 = _mm256_insertf128_ps(_mm256_castps128_ps256(xmm2), xmm3, 1);
	return _mm512_insertf32x8(_mm512_castps256_ps512(ymm0), ymm1, 1);
}
/*
Builds 512-bit vector from 256-bit chunks.
ret[0..255] = ymm0
ret[256..511] = ymm1
*/
__forceinline __m512 ymm_x2_to_zmm(__m256 ymm0, __m256 ymm1)
{
	return _mm512_insertf32x8(_mm512_castps256_ps512(ymm0), ymm1, 1);
}

static constexpr double PI = 3.1415926535897932384626433;