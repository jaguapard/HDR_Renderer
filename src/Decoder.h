#pragma once
#include "Vec.h"

struct Decoder
{
	static Vec4_f32x16 R10G11B10A1_gamma2_to_linear(int32x16 packed);
	static Vec4_f32x16 RGBA8888_to_linear_using_FP16_LUT(const u32x16& packed);
};