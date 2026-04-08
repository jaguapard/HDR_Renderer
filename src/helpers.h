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
	int64_t positive = abs(value);
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