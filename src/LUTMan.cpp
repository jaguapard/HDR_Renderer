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

LUTMan::__LutMan_tables_t::__LutMan_tables_t()
{
	for (int i = 0; i < 256; ++i)
	{
		float f32 = pow(i / 255.0, 2.2);
		this->rgbToLinear_fp32[i] = f32;
		this->rgbToLinear_fp16[i] = f32;
	}
	for (double i = 0; i < this->cos_fp32.size(); ++i) this->cos_fp32[i] = std::cos(2 * M_PI * i / this->cos_fp32.size());
	for (double i = 0; i < this->sin_fp32.size(); ++i) this->sin_fp32[i] = std::sin(2 * M_PI * i / this->sin_fp32.size());
}
