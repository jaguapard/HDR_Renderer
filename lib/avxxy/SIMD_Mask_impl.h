#pragma once
#include "SIMD_Mask.h"
#include <iostream>
//#include "funcs.h"

namespace AVXXY_NAMESPACE
{
	template<meta::ScalarSizeClassEnum LS, size_t N> requires IsValid_SIMD_Mask<N>
	inline SIMD_Mask<LS,N>::VecT SIMD_Mask<LS, N>::_movm(UintT bits)
	{
		using T = SIMD_Mask<LS, N>::SizeTraits;
		SIMD_Mask<LS, N>::VecT ret;
		//scalar movm
		for (size_t i = 0; i < N; ++i)
		{
			ret[i] = bits & (UintT(1) << i) ? T::AllOnesUint : 0;
		}
		return ret;
		
	}

	template<meta::ScalarSizeClassEnum LS, size_t N> requires IsValid_SIMD_Mask<N>
	inline SIMD_Mask<LS,N>::UintT SIMD_Mask<LS, N>::_movemask() const
	{
		if constexpr (IsBitMask) return underlying & AllOnesUint;
		//scalar movemask
		UintT ret = 0;
		for (size_t i = 0; i < N; ++i)
		{
			if (underlying[i] < 0) ret |= UintT(1) << i;
		}
		return ret;
	}

	template<meta::ScalarSizeClassEnum LS, size_t N> requires IsValid_SIMD_Mask<N>
	inline SIMD_Mask<LS, N>::SIMD_Mask(UintT bits)
	{
		if constexpr (IsBitMask) underlying = bits & AllOnesUint;
		else underlying = SIMD_Mask<LS, N>::_movm(bits);
	}

	template<meta::ScalarSizeClassEnum LS, size_t N>  requires IsValid_SIMD_Mask<N>
	inline SIMD_Mask<LS, N>::SIMD_Mask(const SIMD_Mask<LS, N / 2>& lo, const SIMD_Mask<LS, N / 2>& hi)
	{
		if constexpr (IsBitMask) return UintT(lo) | (UintT(hi) << (N / 2));
		else 
		{
			SIMD_Mask<LS, N> ret;
			ret.underlying = { lo.underlying, hi.underlying };
			return ret;
		}
	}

	template<meta::ScalarSizeClassEnum LS, size_t N> requires IsValid_SIMD_Mask<N>
	inline SIMD_Mask<LS, N / 2> SIMD_Mask<LS, N>::lo() const
	{
		static_assert(N % 2 == 0);
		if constexpr (IsBitMask) return underlying;
		else return underlying.lo();
	}
	template<meta::ScalarSizeClassEnum LS, size_t N> requires IsValid_SIMD_Mask<N>
	inline SIMD_Mask<LS, N / 2> SIMD_Mask<LS, N>::hi() const
	{
		static_assert(N % 2 == 0);
		if constexpr (IsBitMask) return underlying >> (N/2);
		else return underlying.hi();
	}

	template<meta::ScalarSizeClassEnum LS, size_t N> requires IsValid_SIMD_Mask<N>
	inline SIMD_Mask<LS, N>::operator UintT() const
	{
		if constexpr (IsBitMask) return underlying & AllOnesUint;
		else return this->_movemask();
	}

	template<meta::ScalarSizeClassEnum LS, size_t N> requires IsValid_SIMD_Mask<N>
	template<meta::ScalarSizeClassEnum LS2, size_t N2> requires (N >= N2)
	inline SIMD_Mask<LS, N>::SIMD_Mask(const SIMD_Mask<LS2, N2>& other)
	{
		*this = other._movemask();
	}
}