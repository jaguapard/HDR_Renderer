#define _USE_MATH_DEFINES
#include "LUTMan.h"
#include <cmath>

float32x16 LUTMan::sin(float32x16 x)
{
	return LUTMan::cos(x - M_PI / 2);
}
float32x16 LUTMan::cos(float32x16 x)
{
	float32x16 periods = x * float(1.0 / (2 * M_PI));
	periods = floor(periods);
	x -= periods * float(2 * M_PI); //now x is 0..2_PI range
	float32x16 lutIndex = x * float(tables.cos_fp32.size() / (2 * M_PI));
	
	int32x16 lutIndexFirst = lutIndex;
	int32x16 lutIndexSecond = lutIndexFirst + 1;

	float32x16 lut0 = load<f32x16>(&tables.cos_fp32[0]);
	float32x16 lut1 = load<f32x16>(&tables.cos_fp32[16]);
	//permutes already cut off MSB's, so we can use them without change
	float32x16 v1 = permx2(lut0, lut1, lutIndexFirst);
	float32x16 v2 = permx2(lut0, lut1, lutIndexSecond);
	float32x16 lerpT = lutIndex - floor(lutIndex);
	return v1 + (v2 - v1) * lerpT;
}

//Vectorized version of: https://innovation.ebayinc.com/stories/fast-approximate-logarithms-part-i-the-basics/
// compute log2(x) by reducing x to [0.75, 1.5)
float32x16 LUTMan::log2(float32x16 x)
{
	// a*(x-1)^2 + b*(x-1) approximates log2(x) when 0.75 <= x < 1.5
	__m512 a = _mm512_set1_ps(-0.6296735);
	__m512 b = _mm512_set1_ps(1.466967);

	/*
	 * Assume IEEE representation, which is sgn(1):exp(8):frac(23)
	 * representing (1+frac)*2^(exp-127)  Call 1+frac the significand
	 */
	__m512i ux1i = _mm512_castps_si512(x);
	// actual exponent is exp-127, will subtract 127 later
	__m512i exp = _mm512_srli_epi32(_mm512_and_si512(ux1i, _mm512_set1_epi32(0x7F800000)), 23);

	__mmask16 greater = _mm512_cmpneq_epi32_mask(_mm512_and_si512(ux1i, _mm512_set1_epi32(0x00400000)), _mm512_set1_epi32(0)); // nonzero if signif >= 1.5
	// if greater then:
	// signif >= 1.5 so need to divide by 2.  Accomplish this by 
	// stuffing exp = 126 which corresponds to an exponent of -1
	//else get signif by stuffing exp = 127 which corresponds to an exponent of 0

	__m512i orc = _mm512_mask_mov_epi32(_mm512_set1_epi32(0x3f800000), greater, _mm512_set1_epi32(0x3f000000));
	__m512i andc = _mm512_set1_epi32(0x007FFFFF);

	__m512i ux2i = _mm512_or_si512(_mm512_and_si512(ux1i, andc), orc);
	__m512 fexp = _mm512_cvtepi32_ps(_mm512_sub_epi32(exp, _mm512_mask_mov_epi32(_mm512_set1_epi32(127), greater, _mm512_set1_epi32(126))));  // 126 instead of 127 compensates for division by 2
	__m512 signif = _mm512_sub_ps(_mm512_castsi512_ps(ux2i), _mm512_set1_ps(1));

	__m512 finA = _mm512_fmadd_ps(b, signif, fexp);
	__m512 finB = _mm512_mul_ps(signif, signif);
	return _mm512_fmadd_ps(a, finB, finA);
}

LUTMan::__LutMan_tables_t::__LutMan_tables_t()
{
	for (int i = 0; i < 256; ++i)
	{
		float f32 = pow(i / 255.0, 2.2);
		this->rgbToLinear_fp32[i] = f32;
		this->rgbToLinear_fp16[i] = _mm_extract_epi16(_mm_cvtps_ph(_mm_set1_ps(f32), _MM_FROUND_TO_NEAREST_INT), 0);
	}
	for (double i = 0; i < this->cos_fp32.size(); ++i) this->cos_fp32[i] = std::cos(2 * M_PI * i / this->cos_fp32.size());
	for (double i = 0; i < this->sin_fp32.size(); ++i) this->sin_fp32[i] = std::sin(2 * M_PI * i / this->sin_fp32.size());
}
