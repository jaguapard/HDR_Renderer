#pragma once
#include <stdint.h>
#include "Mask16.h"
#include "float32x16.h"
#include "AVX512_MaskedOp.h"

namespace bob
{
	struct int32x16;

	typedef AVX512_MaskedOp<int32x16, Mask16, int32_t> MaskedOp_i32x16;
	struct alignas(64) int32x16
	{
		union {
			int32_t el[16];
			__m512i zmm;
		};

		int32x16();
		int32x16(const __m512i& m, Mask16 mask = 0xFFFF, const int32x16& fillerVal = {});
		int32x16(const int32_t x, Mask16 mask = 0xFFFF, const int32x16& fillerVal = {});

		int32x16(const int32_t* p, Mask16 mask = 0xFFFF, const int32x16& fillerVal = {});
		//int32x16(const float32x16& f, Mask16 mask = 0xFFFF, const int32x16& fillerVal = {});
		int32x16(const int32x16& other, Mask16 mask = 0xFFFF, const int32x16& fillerVal = {});
		int32x16(int32_t i1, int32_t i2, int32_t i3, int32_t i4, int32_t i5, int32_t i6, int32_t i7, int32_t i8, int32_t i9, int32_t i10, int32_t i11, int32_t i12, int32_t i13, int32_t i14, int32_t i15, int32_t i16);

		__forceinline void store(void* dst, Mask16 mask = 0xFFFF);

		__forceinline friend int32x16 operator+(const int32x16& lhs, const MaskedOp_i32x16& rhs);
		__forceinline friend int32x16 operator-(const int32x16& lhs, const MaskedOp_i32x16& rhs);
		__forceinline friend int32x16 operator*(const int32x16& lhs, const MaskedOp_i32x16& rhs);
		__forceinline friend int32x16 operator/(const int32x16& lhs, const MaskedOp_i32x16& rhs); //AVX512 does not have integer division, so it is emulated by double-precision division.
		__forceinline int32x16& operator+=(const MaskedOp_i32x16& other);
		__forceinline int32x16& operator-=(const MaskedOp_i32x16& other);
		__forceinline int32x16& operator*=(const MaskedOp_i32x16& other);
		__forceinline int32x16& operator/=(const MaskedOp_i32x16& other);

		__forceinline Mask16 operator>(const int32x16& other) const;
		__forceinline Mask16 operator>=(const int32x16& other) const;
		__forceinline Mask16 operator<(const int32x16& other) const;
		__forceinline Mask16 operator<=(const int32x16& other) const;
		__forceinline Mask16 operator==(const int32x16& other) const;
		__forceinline Mask16 operator!=(const int32x16& other) const;


		__forceinline friend int32x16 operator&(const int32x16& lhs, const MaskedOp_i32x16& rhs);
		__forceinline friend int32x16 operator|(const int32x16& lhs, const MaskedOp_i32x16& rhs);
		__forceinline friend int32x16 operator^(const int32x16& lhs, const MaskedOp_i32x16& rhs);
		__forceinline int32x16& operator&=(const MaskedOp_i32x16& other);
		__forceinline int32x16& operator|=(const MaskedOp_i32x16& other);
		__forceinline int32x16& operator^=(const MaskedOp_i32x16& other);

		__forceinline int32x16 operator<<(const MaskedOp_i32x16& other) const;
		__forceinline int32x16 operator>>(const MaskedOp_i32x16& other) const;

		__forceinline int32x16& operator<<=(const MaskedOp_i32x16& other);
		__forceinline int32x16& operator>>=(const MaskedOp_i32x16& other);


		__forceinline int32x16 operator-() const;
		__forceinline int32x16 operator~() const;

		__forceinline operator __m512i() const;
		__forceinline operator float32x16() const;
		__forceinline operator __m512() const;

		__forceinline int32x16 clamp(const int32x16& min, const int32x16& max) const;
		__forceinline int32x16 abs() const;
		//int32x16 sqrt() const;
		//int32x16 rsqrt14() const;
		//int32x16 rsqrt28() const;

		__forceinline static int32x16 sequence(int32_t mult = 1);

		__forceinline const int32_t& operator[](size_t i) const;
		__forceinline int32_t& operator[](size_t i);
	};

	inline int32x16::int32x16(const int32_t x, Mask16 mask, const int32x16& fillerVal)
	{
		zmm = _mm512_mask_mov_epi32(fillerVal, mask, _mm512_set1_epi32(x));
	}

	inline int32x16::int32x16()
	{
		zmm = _mm512_set1_epi32(0);
	}

	inline int32x16::int32x16(const __m512i& m, Mask16 mask, const int32x16& fillerVal)
	{
		zmm = _mm512_mask_mov_epi32(fillerVal, mask, m);
	}

	inline int32x16::int32x16(const int32_t* p, Mask16 mask, const int32x16& fillerVal)
	{
		zmm = _mm512_mask_loadu_epi32(fillerVal, mask, p);
	}

	/*
	inline int32x16::int32x16(const float32x16& f, Mask16 mask, const int32x16& fillerVal)
	{
		zmm = _mm512_mask_cvttps_epi32(fillerVal, mask, f);
	}*/

	inline int32x16::int32x16(const int32x16& other, Mask16 mask, const int32x16& fillerVal)
	{
		zmm = _mm512_mask_mov_epi32(fillerVal, mask, other);
	}

	inline int32x16::int32x16(int32_t i1, int32_t i2, int32_t i3, int32_t i4, int32_t i5, int32_t i6, int32_t i7, int32_t i8, int32_t i9, int32_t i10, int32_t i11, int32_t i12, int32_t i13, int32_t i14, int32_t i15, int32_t i16)
	{
		zmm = _mm512_setr_epi32(i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, i16);
	}

	inline void int32x16::store(void* dst, Mask16 mask)
	{
		_mm512_mask_store_epi32(dst, mask, *this);
	}

	inline int32x16 operator+(const int32x16& lhs, const MaskedOp_i32x16& rhs)
	{
		switch (rhs.mode)
		{
		case MaskedOp_i32x16::Mode::UNCONDITIONAL:
			return _mm512_add_epi32(lhs, rhs.a);
		case MaskedOp_i32x16::Mode::CONDITIONAL:
			return _mm512_mask_add_epi32(lhs, rhs.mask, lhs, rhs.a);
		case MaskedOp_i32x16::Mode::BLEND:
			return _mm512_add_epi32(lhs, _mm512_mask_blend_epi32(rhs.mask, rhs.b, rhs.a));
		case MaskedOp_i32x16::Mode::ZERO_MASKING:
			return _mm512_maskz_add_epi32(rhs.mask, lhs, rhs.a);
		case MaskedOp_i32x16::Mode::MERGE_MASKING:
			return _mm512_mask_add_epi32(rhs.src, rhs.mask, lhs, rhs.a);
		default:
			break;
		}		
	}

	inline int32x16 operator-(const int32x16& lhs, const MaskedOp_i32x16& rhs)
	{
		switch (rhs.mode)
		{
		case MaskedOp_i32x16::Mode::UNCONDITIONAL:
			return _mm512_sub_epi32(lhs, rhs.a);
		case MaskedOp_i32x16::Mode::CONDITIONAL:
			return _mm512_mask_sub_epi32(lhs, rhs.mask, lhs, rhs.a);
		case MaskedOp_i32x16::Mode::BLEND:
			return _mm512_sub_epi32(lhs, _mm512_mask_blend_epi32(rhs.mask, rhs.b, rhs.a));
		case MaskedOp_i32x16::Mode::ZERO_MASKING:
			return _mm512_maskz_sub_epi32(rhs.mask, lhs, rhs.a);
		case MaskedOp_i32x16::Mode::MERGE_MASKING:
			return _mm512_mask_sub_epi32(rhs.src, rhs.mask, lhs, rhs.a);
		default:
			break;
		}
	}

	inline int32x16 operator*(const int32x16& lhs, const MaskedOp_i32x16& rhs)
	{
		switch (rhs.mode)
		{
		case MaskedOp_i32x16::Mode::UNCONDITIONAL:
			return _mm512_mullo_epi32(lhs, rhs.a);
		case MaskedOp_i32x16::Mode::CONDITIONAL:
			return _mm512_mask_mullo_epi32(lhs, rhs.mask, lhs, rhs.a);
		case MaskedOp_i32x16::Mode::BLEND:
			return _mm512_mullo_epi32(lhs, _mm512_mask_blend_epi32(rhs.mask, rhs.b, rhs.a));
		case MaskedOp_i32x16::Mode::ZERO_MASKING:
			return _mm512_maskz_mullo_epi32(rhs.mask, lhs, rhs.a);
		case MaskedOp_i32x16::Mode::MERGE_MASKING:
			return _mm512_mask_mullo_epi32(rhs.src, rhs.mask, lhs, rhs.a);
		default:
			break;
		}
	}

	inline int32x16 operator/(const int32x16& lhs, const MaskedOp_i32x16& rhs)
	{
		__m512i arg = rhs.mode != MaskedOp_i32x16::Mode::BLEND ? __m512i(rhs.a) : _mm512_mask_blend_epi32(rhs.mask, rhs.b, rhs.a);
		__m512d lhsDouble1 = _mm512_cvtepi32_pd(_mm512_extracti32x8_epi32(lhs.zmm, 0));
		__m512d lhsDouble2 = _mm512_cvtepi32_pd(_mm512_extracti32x8_epi32(lhs.zmm, 1));
		__m512d rhsDouble1 = _mm512_cvtepi32_pd(_mm512_extracti32x8_epi32(arg, 0));
		__m512d rhsDouble2 = _mm512_cvtepi32_pd(_mm512_extracti32x8_epi32(arg, 1));

		__m512d div1 = _mm512_div_pd(lhsDouble1, rhsDouble1);
		__m512d div2 = _mm512_div_pd(lhsDouble2, rhsDouble2);
		__m256i ret1 = _mm512_cvttpd_epi32(div1);
		__m256i ret2 = _mm512_cvttpd_epi32(div2);
		__m512i result =  _mm512_inserti32x8(_mm512_castsi256_si512(ret1), ret2, 1);

		switch (rhs.mode)
		{
		case MaskedOp_i32x16::Mode::UNCONDITIONAL:
		case MaskedOp_i32x16::Mode::BLEND:
			return result;
		case MaskedOp_i32x16::Mode::CONDITIONAL:
			return _mm512_mask_blend_epi32(rhs.mask, lhs, result);
		case MaskedOp_i32x16::Mode::ZERO_MASKING:
			return _mm512_maskz_mov_epi32(rhs.mask, result);
		case MaskedOp_i32x16::Mode::MERGE_MASKING:
			return _mm512_mask_blend_epi32(rhs.mask, rhs.src, result);
		default:
			break;
		}
	}

	inline int32x16 int32x16::operator-() const
	{
		return 0 - *this;
	}

	inline int32x16 int32x16::operator~() const
	{
		return _mm512_xor_epi32(zmm, _mm512_set1_epi32(-1));
	}

	inline int32x16::operator __m512i() const
	{
		return zmm;
	}

	inline int32x16::operator float32x16() const
	{
		return _mm512_cvtepi32_ps(zmm);
	}

	inline int32x16::operator __m512() const
	{
		return _mm512_cvtepi32_ps(zmm);
	}

	inline int32x16 int32x16::clamp(const int32x16& min, const int32x16& max) const
	{
		__m512i clampLo = _mm512_max_epi32(zmm, min);
		return _mm512_min_epi32(clampLo, max);
	}

	inline int32x16 int32x16::sequence(int32_t mult)
	{
		return int32x16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15) * int32x16(mult);
	}

	inline const int32_t& int32x16::operator[](size_t i) const
	{
		return el[i];
	}

	inline int32_t& int32x16::operator[](size_t i)
	{
		return el[i];
	}

	/*
	inline int32x16& int32x16::operator+=(const int32_t other)
	{
		return *this = *this + other;
	}

	inline int32x16& int32x16::operator-=(const int32_t other)
	{
		return *this = *this - other;
	}

	inline int32x16& int32x16::operator*=(const int32_t other)
	{
		return *this = *this * other;
	}

	inline int32x16& int32x16::operator/=(const int32_t other)
	{
		return *this = *this / other;
	}

	inline int32x16 int32x16::operator<<(const int32_t other) const
	{
		return _mm512_sllv_epi32(zmm, _mm512_set1_epi32(other));
	}

	

	inline int32x16 int32x16::operator>>(const int32_t other) const
	{
		return _mm512_srlv_epi32(zmm, _mm512_set1_epi32(other));
	}
	*/
	inline int32x16 int32x16::operator<<(const MaskedOp_i32x16& rhs) const
	{
		const auto& lhs = *this;
		switch (rhs.mode)
		{
		case MaskedOp_i32x16::Mode::UNCONDITIONAL:
			return _mm512_sllv_epi32(lhs, rhs.a);
		case MaskedOp_i32x16::Mode::CONDITIONAL:
			return _mm512_mask_sllv_epi32(lhs, rhs.mask, lhs, rhs.a);
		case MaskedOp_i32x16::Mode::BLEND:
			return _mm512_sllv_epi32(lhs, _mm512_mask_blend_epi32(rhs.mask, rhs.b, rhs.a));
		case MaskedOp_i32x16::Mode::ZERO_MASKING:
			return _mm512_maskz_sllv_epi32(rhs.mask, lhs, rhs.a);
		case MaskedOp_i32x16::Mode::MERGE_MASKING:
			return _mm512_mask_sllv_epi32(rhs.src, rhs.mask, lhs, rhs.a);
		default:
			break;
		}
	}

	inline int32x16 int32x16::operator>>(const MaskedOp_i32x16& rhs) const
	{
		const auto& lhs = *this;
		switch (rhs.mode)
		{
		case MaskedOp_i32x16::Mode::UNCONDITIONAL:
			return _mm512_srlv_epi32(lhs, rhs.a);
		case MaskedOp_i32x16::Mode::CONDITIONAL:
			return _mm512_mask_srlv_epi32(lhs, rhs.mask, lhs, rhs.a);
		case MaskedOp_i32x16::Mode::BLEND:
			return _mm512_srlv_epi32(lhs, _mm512_mask_blend_epi32(rhs.mask, rhs.b, rhs.a));
		case MaskedOp_i32x16::Mode::ZERO_MASKING:
			return _mm512_maskz_srlv_epi32(rhs.mask, lhs, rhs.a);
		case MaskedOp_i32x16::Mode::MERGE_MASKING:
			return _mm512_mask_srlv_epi32(rhs.src, rhs.mask, lhs, rhs.a);
		default:
			break;
		}
	}

	inline int32x16& int32x16::operator<<=(const MaskedOp_i32x16& other)
	{
		return *this = *this << other;
	}
	inline int32x16& int32x16::operator>>=(const MaskedOp_i32x16& other)
	{
		return *this = *this >> other;
	}

	inline int32x16& int32x16::operator+=(const MaskedOp_i32x16& other)
	{
		return *this = *this + other;
	}

	inline int32x16& int32x16::operator-=(const MaskedOp_i32x16& other)
	{
		return *this = *this - other;
	}

	inline int32x16& int32x16::operator*=(const MaskedOp_i32x16& other)
	{
		return *this = *this * other;
	}

	inline int32x16& int32x16::operator/=(const MaskedOp_i32x16& other)
	{
		return *this = *this / other;
	}

	inline Mask16 int32x16::operator>(const int32x16& other) const
	{
		return _mm512_cmpgt_epi32_mask(zmm, other);
	}

	inline Mask16 int32x16::operator>=(const int32x16& other) const
	{
		return _mm512_cmpge_epi32_mask(zmm, other);
	}

	inline Mask16 int32x16::operator<(const int32x16& other) const
	{
		return _mm512_cmplt_epi32_mask(zmm, other);
	}

	inline Mask16 int32x16::operator<=(const int32x16& other) const
	{
		return _mm512_cmple_epi32_mask(zmm, other);
	}

	inline Mask16 int32x16::operator==(const int32x16& other) const
	{
		return _mm512_cmpeq_epi32_mask(zmm, other);
	}

	inline Mask16 int32x16::operator!=(const int32x16& other) const
	{
		return _mm512_cmpneq_epi32_mask(zmm, other);
	}

	inline int32x16 operator&(const int32x16& lhs, const MaskedOp_i32x16& rhs)
	{
		switch (rhs.mode)
		{
		case MaskedOp_i32x16::Mode::UNCONDITIONAL:
			return _mm512_and_epi32(lhs, rhs.a);
		case MaskedOp_i32x16::Mode::CONDITIONAL:
			return _mm512_mask_and_epi32(lhs, rhs.mask, lhs, rhs.a);
		case MaskedOp_i32x16::Mode::BLEND:
			return _mm512_and_epi32(lhs, _mm512_mask_blend_epi32(rhs.mask, rhs.b, rhs.a));
		case MaskedOp_i32x16::Mode::ZERO_MASKING:
			return _mm512_maskz_and_epi32(rhs.mask, lhs, rhs.a);
		case MaskedOp_i32x16::Mode::MERGE_MASKING:
			return _mm512_mask_and_epi32(rhs.src, rhs.mask, lhs, rhs.a);
		default:
			break;
		}
	}

	inline int32x16 operator|(const int32x16& lhs, const MaskedOp_i32x16& rhs)
	{
		switch (rhs.mode)
		{
		case MaskedOp_i32x16::Mode::UNCONDITIONAL:
			return _mm512_or_epi32(lhs, rhs.a);
		case MaskedOp_i32x16::Mode::CONDITIONAL:
			return _mm512_mask_or_epi32(lhs, rhs.mask, lhs, rhs.a);
		case MaskedOp_i32x16::Mode::BLEND:
			return _mm512_or_epi32(lhs, _mm512_mask_blend_epi32(rhs.mask, rhs.b, rhs.a));
		case MaskedOp_i32x16::Mode::ZERO_MASKING:
			return _mm512_maskz_or_epi32(rhs.mask, lhs, rhs.a);
		case MaskedOp_i32x16::Mode::MERGE_MASKING:
			return _mm512_mask_or_epi32(rhs.src, rhs.mask, lhs, rhs.a);
		default:
			break;
		}
	}

	inline int32x16 operator^(const int32x16& lhs, const MaskedOp_i32x16& rhs)
	{
		switch (rhs.mode)
		{
		case MaskedOp_i32x16::Mode::UNCONDITIONAL:
			return _mm512_xor_epi32(lhs, rhs.a);
		case MaskedOp_i32x16::Mode::CONDITIONAL:
			return _mm512_mask_xor_epi32(lhs, rhs.mask, lhs, rhs.a);
		case MaskedOp_i32x16::Mode::BLEND:
			return _mm512_xor_epi32(lhs, _mm512_mask_blend_epi32(rhs.mask, rhs.b, rhs.a));
		case MaskedOp_i32x16::Mode::ZERO_MASKING:
			return _mm512_maskz_xor_epi32(rhs.mask, lhs, rhs.a);
		case MaskedOp_i32x16::Mode::MERGE_MASKING:
			return _mm512_mask_xor_epi32(rhs.src, rhs.mask, lhs, rhs.a);
		default:
			break;
		}
	}

	inline int32x16& int32x16::operator&=(const MaskedOp_i32x16& other)
	{
		return *this = *this & other;
	}

	inline int32x16& int32x16::operator|=(const MaskedOp_i32x16& other)
	{
		return *this = *this | other;
	}

	inline int32x16& int32x16::operator^=(const MaskedOp_i32x16& other)
	{
		return *this = *this ^ other;
	}

	inline int32x16 int32x16::abs() const
	{
		return _mm512_abs_epi32(*this);
	}

}