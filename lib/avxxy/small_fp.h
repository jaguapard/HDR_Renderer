#pragma once
#include "namespace.h"
#include <stdint.h>

namespace AVXXY_NAMESPACE
{
	static float _scalar_fp16_to_fp32(uint16_t h);
	static uint16_t _scalar_fp32_to_fp16(float x);

	struct fp16_t
	{
		uint16_t value;

		fp16_t() {};
		fp16_t(float x)
		{
			value = _scalar_fp32_to_fp16(x);
		}

		operator float() const
		{
			return _scalar_fp16_to_fp32(value);
		}
	};
	struct bf16_t
	{
		uint16_t value;
		bf16_t() {};
		//bf16_t(float f);
		//operator float() const;
	};






	//TODO: verify that it works. Maybe a 2^16 entry LUT is better?
	//It does work based on surface-tests, but NaNs and other quirks are untested
	static float _scalar_fp16_to_fp32(uint16_t h)
	{
		uint32_t sign = (h & 0x8000) << 16;
		uint32_t exp = (h >> 10) & 0x1F;
		uint32_t frac = h & 0x03FF;

		uint32_t f;

		if (exp == 0)
		{
			if (frac == 0)
			{
				// Zero
				f = sign;
			}
			else
			{
				// Subnormal half -> normalized float

				// Normalize mantissa
				exp = 1;

				while ((frac & 0x0400) == 0)
				{
					frac <<= 1;
					exp--;
				}

				frac &= 0x03FF;

				uint32_t fp32_exp = exp + (127 - 15);
				uint32_t fp32_frac = frac << 13;

				f = sign | (fp32_exp << 23) | fp32_frac;
			}
		}
		else if (exp == 31)
		{
			// Inf or NaN
			uint32_t fp32_exp = 0xFF;
			uint32_t fp32_frac = frac << 13;

			f = sign | (fp32_exp << 23) | fp32_frac;
		}
		else
		{
			// Normalized number
			uint32_t fp32_exp = exp + (127 - 15);
			uint32_t fp32_frac = frac << 13;

			f = sign | (fp32_exp << 23) | fp32_frac;
		}

		float result;
		memcpy(&result, &f, sizeof(result));
		return result;
	}

	//TODO: verify that it works
	//It does work based on surface-tests, but NaNs and other quirks are untested
	static uint16_t _scalar_fp32_to_fp16(float x)
	{
		uint32_t f;
		memcpy(&f, &x, sizeof(f));

		uint32_t sign = (f >> 16) & 0x8000;
		uint32_t exp = (f >> 23) & 0xFF;
		uint32_t frac = f & 0x7FFFFF;

		// NaN or Inf
		if (exp == 0xFF)
		{
			if (frac == 0)
			{
				// Infinity
				return sign | 0x7C00;
			}
			else
			{
				// NaN
				uint16_t nan = frac >> 13;

				// Ensure mantissa nonzero
				if (nan == 0)
					nan = 1;

				return sign | 0x7C00 | nan;
			}
		}

		// Rebias exponent
		int32_t new_exp = (int32_t)exp - 127 + 15;

		// Overflow -> Inf
		if (new_exp >= 31)
		{
			return sign | 0x7C00;
		}

		// Underflow / subnormal
		if (new_exp <= 0)
		{
			// Too small -> zero
			if (new_exp < -10)
			{
				return sign;
			}

			// Produce subnormal FP16
			frac |= 0x800000;

			int shift = 14 - new_exp;

			uint32_t mant = frac >> shift;

			// Round to nearest even
			uint32_t round_bit = 1u << (shift - 1);

			if ((frac & round_bit) &&
				((frac & (round_bit - 1)) || (mant & 1)))
			{
				mant++;
			}

			return sign | (uint16_t)mant;
		}

		// Normalized FP16
		uint16_t half_exp = (uint16_t)(new_exp << 10);
		uint16_t half_frac = (uint16_t)(frac >> 13);

		// Round to nearest even
		uint32_t round_bits = frac & 0x1FFF;

		if (round_bits > 0x1000 || (round_bits == 0x1000 && (half_frac & 1)))
		{
			half_frac++;

			// Mantissa overflow
			if (half_frac == 0x400)
			{
				half_frac = 0;
				half_exp += 0x0400;

				// Exponent overflow
				if (half_exp >= 0x7C00)
				{
					half_exp = 0x7C00;
				}
			}
		}

		return sign | half_exp | half_frac;
	}
}