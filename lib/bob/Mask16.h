#pragma once
#include <immintrin.h>

namespace bob
{
	struct Mask16
	{
		__mmask16 mask;
		Mask16() = default;
		Mask16(const __mmask16& m);

		Mask16 operator&(const Mask16& other) const;
		Mask16 operator|(const Mask16& other) const;
		Mask16 operator^(const Mask16& other) const;
		Mask16 operator~() const;
		Mask16& operator&=(const Mask16& other);
		Mask16& operator|=(const Mask16& other);
		Mask16& operator^=(const Mask16& other);

		//static Mask16 allOnes();
		bool allOnes() const;
		bool allZeros() const;
		operator __mmask16() const;
		explicit operator bool() const;
	};

	__forceinline Mask16::Mask16(const __mmask16& m)
	{
		mask = m;
	}

	__forceinline Mask16 Mask16::operator&(const Mask16& other) const
	{
		return _kand_mask16(mask, other.mask);
	}

	__forceinline Mask16 Mask16::operator|(const Mask16& other) const
	{
		return _kor_mask16(mask, other.mask);
	}

	__forceinline Mask16 Mask16::operator^(const Mask16& other) const
	{
		return _kxor_mask16(mask, other.mask);
	}

	__forceinline Mask16 Mask16::operator~() const
	{
		return _knot_mask16(*this);
	}

	__forceinline Mask16& Mask16::operator&=(const Mask16& other)
	{
		*this = *this & other;
		return *this;
	}

	__forceinline Mask16& Mask16::operator|=(const Mask16& other)
	{
		*this = *this | other;
		return *this;
	}

	__forceinline Mask16& Mask16::operator^=(const Mask16& other)
	{
		*this = *this ^ other;
		return *this;
	}

	__forceinline bool Mask16::allOnes() const
	{
		return *this == 0xFFFF;
	}

	__forceinline bool Mask16::allZeros() const
	{
		return *this == 0;
	}

	__forceinline Mask16::operator __mmask16() const
	{
		return mask;
	}

	__forceinline Mask16::operator bool() const
	{
		return !_ktestz_mask16_u8(mask, mask);
	}

	/*
	__forceinline Mask16 Mask16::allOnes()
	{
		return 0xFFFF;
	}*/
}