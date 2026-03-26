#pragma once


template <typename T>
inline T lerp(const T& start, const T& end, float amount)
{
	return start + (end - start) * amount;
}

inline float inverse_lerp(float from, float to, float value)
{
	return (value - from) / (to - from);
}