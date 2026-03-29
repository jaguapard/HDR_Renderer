#pragma once
#include "Vec.h"

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
