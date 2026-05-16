#pragma once
#include "Vec.h"

//Provides very fast low-accuracy approximations of various functions
class LUTMan
{
public:
	static void init();
	static float32x16 sin(float32x16 x);
	static float32x16 cos(float32x16 x);
	static float32x16 log2(float32x16 x);

	static const std::array<int16_t, 256> rgbToLinear_fp16; //unlike other LUTs here, this is exact aside rounding differences
private:
	static inline std::array<float, 32> sineLUT_fp32, cosLUT_fp32;
};