#pragma once
#include "SIMD_Mask.h"
#include <iostream>
#include "funcs.h"

namespace AVXXY_NAMESPACE
{
	template<typename S, size_t N>
	inline constexpr SIMD_Mask<S, N> SIMD_Mask<S, N>::AllOnes()
	{
		return (N == sizeof(UintT) * 8) ? ~UintT(0) : ((UintT(1) << N) - 1);
	}
	template<typename S, size_t N>
	inline SIMD_Mask<S, N>::SIMD_Mask(UintT bits)
	{
		if constexpr (IsBitMask) this->underlying = bits;
		else
		{
			//TODO: bits to vector mask conversion here!
			//this->underlying = Default
		}
	}
}