#pragma once
#include <immintrin.h>
#include "SSE_Vec4.h"
#include "Mask16.h"

namespace bob
{
	struct alignas(64) float32x16
	{
		union {
			float f[16];
			__m512 zmm;
		};

		float32x16();
		float32x16(const float x, Mask16 mask = 0xFFFF, const float32x16& fillerVal = {});
		float32x16(const __m512& m, Mask16 mask = 0xFFFF, const float32x16& fillerVal = {});
		float32x16(const float* p, Mask16 mask = 0xFFFF, const float32x16& fillerVal = {});
		float32x16(const float32x16& other, Mask16 mask = 0xFFFF, const float32x16& fillerVal = {});
		float32x16(float f1, float f2, float f3, float f4, float f5, float f6, float f7, float f8, float f9, float f10, float f11, float f12, float f13, float f14, float f15, float f16);		

		float32x16 operator+(const float other) const;
		float32x16 operator-(const float other) const;
		float32x16 operator*(const float other) const;
		float32x16 operator/(const float other) const;

		float32x16& operator+=(const float other);
		float32x16& operator-=(const float other);
		float32x16& operator*=(const float other);
		float32x16& operator/=(const float other);

		float32x16 operator+(const float32x16& other) const;
		float32x16 operator-(const float32x16& other) const;
		float32x16 operator*(const float32x16& other) const;
		float32x16 operator/(const float32x16& other) const;
		float32x16& operator+=(const float32x16& other);
		float32x16& operator-=(const float32x16& other);
		float32x16& operator*=(const float32x16& other);
		float32x16& operator/=(const float32x16& other);

		Mask16 operator>(const float32x16& other) const;
		Mask16 operator>=(const float32x16& other) const;
		Mask16 operator<(const float32x16& other) const;
		Mask16 operator<=(const float32x16& other) const;
		Mask16 operator==(const float32x16& other) const;
		Mask16 operator!=(const float32x16& other) const;


		float32x16 operator&(const float32x16& other) const;
		float32x16 operator|(const float32x16& other) const;
		float32x16 operator^(const float32x16& other) const;
		float32x16& operator&=(const float32x16& other);
		float32x16& operator|=(const float32x16& other);
		float32x16& operator^=(const float32x16& other);

		float32x16 operator-() const;
		float32x16 operator~() const;
		operator __m512() const;

		float32x16 clamp(float min, float max) const;
		float32x16 clamp(const float32x16& min, const float32x16& max) const;
		float32x16 sqrt() const;
		float32x16 rsqrt14() const;
		float32x16 rsqrt28() const;
		__m512i trunc() const;

		static float32x16 sequence(float mult = 1.0);

		const float& operator[](size_t i) const;
		float& operator[](size_t i);
	};

	inline float32x16::float32x16()
	{
	}

	inline float32x16::float32x16(const float x, Mask16 mask, const float32x16& fillerVal)
	{
		zmm = _mm512_mask_mov_ps(fillerVal, mask, _mm512_set1_ps(x));
	}

	inline float32x16::float32x16(const __m512& m, Mask16 mask, const float32x16& fillerVal)
	{
		zmm = _mm512_mask_mov_ps(fillerVal, mask, m);
	}

	inline float32x16::float32x16(const float* p, Mask16 mask, const float32x16& fillerVal)
	{
		zmm = _mm512_mask_loadu_ps(fillerVal, mask, p);
	}

	inline float32x16::float32x16(float f1, float f2, float f3, float f4, float f5, float f6, float f7, float f8, float f9, float f10, float f11, float f12, float f13, float f14, float f15, float f16)
	{
		*this = _mm512_setr_ps(f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16);
	}

	inline float32x16::float32x16(const float32x16& other, Mask16 mask, const float32x16& fillerVal)
	{
		zmm = float32x16(other.zmm, mask, fillerVal);
	}

	inline float32x16 float32x16::operator+(const float other) const
	{
		return _mm512_add_ps(zmm, _mm512_set1_ps(other));
	}

	inline float32x16 float32x16::operator-(const float other) const
	{
		return _mm512_sub_ps(zmm, _mm512_set1_ps(other));
	}

	inline float32x16 float32x16::operator*(const float other) const
	{
		return _mm512_mul_ps(zmm, _mm512_set1_ps(other));
	}

	inline float32x16 float32x16::operator/(const float other) const
	{
		return _mm512_div_ps(zmm, _mm512_set1_ps(other));
	}

	inline float32x16& float32x16::operator+=(const float other)
	{
		return *this = *this + other;
	}

	inline float32x16& float32x16::operator-=(const float other)
	{
		return *this = *this - other;
	}

	inline float32x16& float32x16::operator*=(const float other)
	{
		return *this = *this * other;
	}

	inline float32x16& float32x16::operator/=(const float other)
	{
		return *this = *this / other;
	}

	inline float32x16 float32x16::operator+(const float32x16& other) const
	{
		return _mm512_add_ps(zmm, other.zmm);
	}

	inline float32x16 float32x16::operator-(const float32x16& other) const
	{
		return _mm512_sub_ps(zmm, other.zmm);
	}

	inline float32x16 float32x16::operator*(const float32x16& other) const
	{
		return _mm512_mul_ps(zmm, other.zmm);
	}

	inline float32x16 float32x16::operator/(const float32x16& other) const
	{
		return _mm512_div_ps(zmm, other.zmm);
	}

	inline Mask16 float32x16::operator>(const float32x16& other) const
	{
		return _mm512_cmp_ps_mask(zmm, other.zmm, _CMP_GT_OQ);
	}

	inline Mask16 float32x16::operator>=(const float32x16& other) const
	{
		return _mm512_cmp_ps_mask(zmm, other.zmm, _CMP_GE_OQ);
	}

	inline Mask16 float32x16::operator<(const float32x16& other) const
	{
		return _mm512_cmp_ps_mask(zmm, other.zmm, _CMP_LT_OQ);
	}

	inline Mask16 float32x16::operator<=(const float32x16& other) const
	{
		return _mm512_cmp_ps_mask(zmm, other.zmm, _CMP_LE_OQ);
	}

	inline Mask16 float32x16::operator==(const float32x16& other) const
	{
		return _mm512_cmp_ps_mask(zmm, other.zmm, _CMP_EQ_OQ);
	}

	inline Mask16 float32x16::operator!=(const float32x16& other) const
	{
		return _mm512_cmp_ps_mask(zmm, other.zmm, _CMP_NEQ_OQ);
	}

	inline float32x16 float32x16::operator&(const float32x16& other) const
	{
		return _mm512_and_ps(zmm, other.zmm);
	}

	inline float32x16 float32x16::operator|(const float32x16& other) const
	{
		return _mm512_or_ps(zmm, other.zmm);
	}

	inline float32x16 float32x16::operator^(const float32x16& other) const
	{
		return _mm512_xor_ps(zmm, other.zmm);
	}

	inline float32x16& float32x16::operator&=(const float32x16& other)
	{
		return *this = *this & other;
	}

	inline float32x16& float32x16::operator|=(const float32x16& other)
	{
		return *this = *this | other;
	}

	inline float32x16& float32x16::operator^=(const float32x16& other)
	{
		return *this = *this ^ other;
	}

	inline float32x16& float32x16::operator+=(const float32x16& other)
	{
		return *this = *this + other;
	}

	inline float32x16& float32x16::operator-=(const float32x16& other)
	{
		return *this = *this - other;
	}

	inline float32x16& float32x16::operator*=(const float32x16& other)
	{
		return *this = *this * other;
	}

	inline float32x16& float32x16::operator/=(const float32x16& other)
	{
		return *this = *this / other;
	}

	inline float32x16 float32x16::operator-() const
	{
		return _mm512_sub_ps(_mm512_setzero_ps(), zmm);
	}

	inline float32x16 float32x16::operator~() const
	{
		return _mm512_xor_ps(zmm, _mm512_castsi512_ps(_mm512_set1_epi32(-1)));
	}

	inline float32x16::operator __m512() const
	{
		return zmm;
	}

	inline float32x16 float32x16::clamp(float min, float max) const
	{
		__m512 c = _mm512_min_ps(zmm, _mm512_set1_ps(max));
		return _mm512_max_ps(c, _mm512_set1_ps(min));
	}

	inline float32x16 float32x16::clamp(const float32x16& min, const float32x16& max) const
	{
		__m512 c = _mm512_min_ps(zmm, max);
		return _mm512_max_ps(c, min);
	}

	inline float32x16 float32x16::sqrt() const
	{
		return _mm512_sqrt_ps(*this);
	}

	inline float32x16 float32x16::rsqrt14() const
	{
		return _mm512_rsqrt14_ps(*this);
	}

	/*/
	inline float32x16 float32x16::rsqrt28() const
	{
		return _mm512_rsqrt28_ps(*this);
	}*/

	inline __m512i float32x16::trunc() const
	{
		return _mm512_cvttps_epi32(zmm);
	}

	inline float32x16 float32x16::sequence(float mult)
	{
		return float32x16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15) * mult;
	}

	inline const float& float32x16::operator[](size_t i) const
	{
		return f[i];
	}

	inline float& float32x16::operator[](size_t i)
	{
		return f[i];
	}
}