#pragma once
#include <immintrin.h>
#include "SSE_Vec4.h"

/*
Struct representing 8 packed  independent floats.
Acts as a wrapper on top of AVX(2) intrinsics.
Freely convertible from/to __m256 and supports bitwise operations
*/
namespace bob
{
	struct alignas(32) float32x8
	{
		union {
			float f[8];
			__m256 ymm;
		};

		float32x8() = default;
		float32x8(const float x);
		float32x8(const __m256& m);
		float32x8(const float* p);
		float32x8(float f1, float f2, float f3, float f4, float f5, float f6, float f7, float f8);

		float32x8 operator+(const float other) const;
		float32x8 operator-(const float other) const;
		float32x8 operator*(const float other) const;
		float32x8 operator/(const float other) const;

		float32x8 operator+=(const float other);
		float32x8 operator-=(const float other);
		float32x8 operator*=(const float other);
		float32x8 operator/=(const float other);

		float32x8 operator+(const float32x8& other) const;
		float32x8 operator-(const float32x8& other) const;
		float32x8 operator*(const float32x8& other) const;
		float32x8 operator/(const float32x8& other) const;
		float32x8& operator+=(const float32x8& other);
		float32x8& operator-=(const float32x8& other);
		float32x8& operator*=(const float32x8& other);
		float32x8& operator/=(const float32x8& other);

		float32x8 operator>(const float32x8& other) const;
		float32x8 operator>=(const float32x8& other) const;
		float32x8 operator<(const float32x8& other) const;
		float32x8 operator<=(const float32x8& other) const;
		float32x8 operator==(const float32x8& other) const;
		float32x8 operator!=(const float32x8& other) const;


		float32x8 operator&(const float32x8& other) const;
		float32x8 operator|(const float32x8& other) const;
		float32x8 operator^(const float32x8& other) const;
		float32x8& operator&=(const float32x8& other);
		float32x8& operator|=(const float32x8& other);
		float32x8& operator^=(const float32x8& other);

		float32x8 operator-() const;
		float32x8 operator~() const;
		operator __m256() const;

		explicit operator bool() const
		{
			return !_mm256_testz_ps(*this, *this);
		}

		float32x8 clamp(float min, float max) const;
		int moveMask() const;

		static float32x8 sequence(float mult = 1.0);
	};

	inline float32x8::float32x8(const float x)
	{
		ymm = _mm256_set1_ps(x);
	}

	inline float32x8::float32x8(const __m256& m)
	{
		ymm = m;
	}

	inline float32x8::float32x8(const float* p)
	{
		ymm = *reinterpret_cast<const __m256*>(p);
	}

	inline float32x8::float32x8(float f1, float f2, float f3, float f4, float f5, float f6, float f7, float f8)
	{
		*this = _mm256_setr_ps(f1, f2, f3, f4, f5, f6, f7, f8);
	}

	inline float32x8 float32x8::operator+(const float other) const
	{
		return _mm256_add_ps(ymm, _mm256_set1_ps(other));
	}

	inline float32x8 float32x8::operator-(const float other) const
	{
		return _mm256_sub_ps(ymm, _mm256_set1_ps(other));
	}

	inline float32x8 float32x8::operator*(const float other) const
	{
		return _mm256_mul_ps(ymm, _mm256_set1_ps(other));
	}

	inline float32x8 float32x8::operator/(const float other) const
	{
		return _mm256_div_ps(ymm, _mm256_set1_ps(other));
	}

	inline float32x8 float32x8::operator+=(const float other)
	{
		return *this = *this + other;
	}

	inline float32x8 float32x8::operator-=(const float other)
	{
		return *this = *this - other;
	}

	inline float32x8 float32x8::operator*=(const float other)
	{
		return *this = *this * other;
	}

	inline float32x8 float32x8::operator/=(const float other)
	{
		return *this = *this / other;
	}

	inline float32x8 float32x8::operator+(const float32x8& other) const
	{
		return _mm256_add_ps(ymm, other.ymm);
	}

	inline float32x8 float32x8::operator-(const float32x8& other) const
	{
		return _mm256_sub_ps(ymm, other.ymm);
	}

	inline float32x8 float32x8::operator*(const float32x8& other) const
	{
		return _mm256_mul_ps(ymm, other.ymm);
	}

	inline float32x8 float32x8::operator/(const float32x8& other) const
	{
		return _mm256_div_ps(ymm, other.ymm);
	}

	inline float32x8 float32x8::operator>(const float32x8& other) const
	{
		return _mm256_cmp_ps(ymm, other.ymm, _CMP_GT_OQ);
	}

	inline float32x8 float32x8::operator>=(const float32x8& other) const
	{
		return _mm256_cmp_ps(ymm, other.ymm, _CMP_GE_OQ);
	}

	inline float32x8 float32x8::operator<(const float32x8& other) const
	{
		return _mm256_cmp_ps(ymm, other.ymm, _CMP_LT_OQ);
	}

	inline float32x8 float32x8::operator<=(const float32x8& other) const
	{
		return _mm256_cmp_ps(ymm, other.ymm, _CMP_LE_OQ);
	}

	inline float32x8 float32x8::operator==(const float32x8& other) const
	{
		return _mm256_cmp_ps(ymm, other.ymm, _CMP_EQ_OQ);
	}

	inline float32x8 float32x8::operator!=(const float32x8& other) const
	{
		return _mm256_cmp_ps(ymm, other.ymm, _CMP_NEQ_OQ);
	}

	inline float32x8 float32x8::operator&(const float32x8& other) const
	{
		return _mm256_and_ps(ymm, other.ymm);
	}

	inline float32x8 float32x8::operator|(const float32x8& other) const
	{
		return _mm256_or_ps(ymm, other.ymm);
	}

	inline float32x8 float32x8::operator^(const float32x8& other) const
	{
		return _mm256_xor_ps(ymm, other.ymm);
	}

	inline float32x8& float32x8::operator&=(const float32x8& other)
	{
		return *this = *this & other;
	}

	inline float32x8& float32x8::operator|=(const float32x8& other)
	{
		return *this = *this | other;
	}

	inline float32x8& float32x8::operator^=(const float32x8& other)
	{
		return *this = *this ^ other;
	}

	inline float32x8& float32x8::operator+=(const float32x8& other)
	{
		return *this = *this + other;
	}

	inline float32x8& float32x8::operator-=(const float32x8& other)
	{
		return *this = *this - other;
	}

	inline float32x8& float32x8::operator*=(const float32x8& other)
	{
		return *this = *this * other;
	}

	inline float32x8& float32x8::operator/=(const float32x8& other)
	{
		return *this = *this / other;
	}

	inline float32x8 float32x8::operator-() const
	{
		return _mm256_sub_ps(_mm256_setzero_ps(), ymm);
	}

	inline float32x8 float32x8::operator~() const
	{
		return _mm256_xor_ps(ymm, _mm256_cmp_ps(ymm, ymm, _CMP_EQ_OQ));
	}

	inline float32x8::operator __m256() const
	{
		return ymm;
	}

	inline float32x8 float32x8::clamp(float min, float max) const
	{
		__m256 c = _mm256_min_ps(ymm, _mm256_set1_ps(max));
		return _mm256_max_ps(c, _mm256_set1_ps(min));
	}

	inline int float32x8::moveMask() const
	{
		return _mm256_movemask_ps(*this);
	}

	inline float32x8 float32x8::sequence(float mult)
	{
		return float32x8(0, 1, 2, 3, 4, 5, 6, 7) * mult;
	}
}