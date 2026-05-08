#define _USE_MATH_DEFINES
#include "LUTMan.h"
#include <cmath>
void LUTMan::init()
{
	//for (int i = 0; i < sineLUT_fp32.size(); ++i) sineLUT_fp32[i] = std::sin(M_PI)
	for (double i = 0; i < cosLUT_fp32.size(); ++i) cosLUT_fp32[i] = std::cos(2 * M_PI * i / cosLUT_fp32.size());
}

float32x16 LUTMan::sin(float32x16 x)
{
	return LUTMan::cos(x - M_PI / 2);
}
float32x16 LUTMan::cos(float32x16 x)
{
	float32x16 periods = x * (1.0 / (2 * M_PI));
	periods = _mm512_floor_ps(periods);
	x -= periods * (2 * M_PI); //now x is 0..2_PI range
	float32x16 lutIndex = x * (cosLUT_fp32.size() / (2 * M_PI));
	
	int32x16 lutIndexFirst = lutIndex.trunc();
	int32x16 lutIndexSecond = lutIndexFirst + 1;

	float32x16 lut0 = _mm512_loadu_ps(&cosLUT_fp32);
	float32x16 lut1 = _mm512_loadu_ps(&cosLUT_fp32[16]);
	//permutes already cut off MSB's, so we can use them without change
	float32x16 v1 = _mm512_permutex2var_ps(lut0, lutIndexFirst, lut1);
	float32x16 v2 = _mm512_permutex2var_ps(lut0, lutIndexSecond, lut1);
	float32x16 lerpT = lutIndex - float32x16(_mm512_floor_ps(lutIndex));
	return v1 + (v2 - v1) * lerpT;
}
