#pragma once
#include "SIMD_Mask.h"
#include <iostream>
//#include "funcs.h"

namespace AVXXY_NAMESPACE
{
	template<meta::ScalarSizeClassEnum LS, size_t N> requires IsValid_SIMD_Mask<N>
	inline SIMD_Mask<LS,N>::VecT SIMD_Mask<LS, N>::_movm(BitsUintT bits)
	{
		using T = SIMD_Mask<LS, N>::SizeTraits;
		SIMD_Mask<LS, N>::VecT ret;
		//scalar movm
		for (size_t i = 0; i < N; ++i)
		{
			ret[i] = bits & (BitsUintT(1) << i) ? T::AllOnesUint : 0;
		}
		return ret;
		
	}

	template<meta::ScalarSizeClassEnum LS, size_t N> requires IsValid_SIMD_Mask<N>
	inline SIMD_Mask<LS,N>::BitsUintT SIMD_Mask<LS, N>::_movemask() const
	{
		if constexpr (IsBitMask) return underlying & AllOnesUint;
		else
		{
			//scalar movemask
			BitsUintT ret = 0;
			for (size_t i = 0; i < N; ++i)
			{
				if (underlying[i] < 0) ret |= BitsUintT(1) << i;
			}
			return ret;
		}
	}

	template<meta::ScalarSizeClassEnum LS, size_t N> requires IsValid_SIMD_Mask<N>
	inline SIMD_Mask<LS, N>::SIMD_Mask(BitsUintT bits)
	{
		if constexpr (IsBitMask) underlying = bits & AllOnesUint;
		else underlying = SIMD_Mask<LS, N>::_movm(bits);
	}

	template<meta::ScalarSizeClassEnum LS, size_t N> requires IsValid_SIMD_Mask<N>
	template<typename T>
	inline SIMD_Mask<LS, N>::SIMD_Mask(const SIMD_Vector<T, N>& vec)
	{
		*this = movemask(vec);
	}

	template<meta::ScalarSizeClassEnum LS, size_t N>  requires IsValid_SIMD_Mask<N>
	template<size_t N2> requires (N2 >= 2 && N2 * 2 == N)
	inline SIMD_Mask<LS, N>::SIMD_Mask(const SIMD_Mask<LS, N2>& lo, const SIMD_Mask<LS, N2>& hi)
	{
		if constexpr (IsBitMask) underlying = BitsUintT(lo) | (BitsUintT(hi) << (N / 2));
		else underlying = { lo.underlying, hi.underlying };
	}

	template<meta::ScalarSizeClassEnum LS, size_t N> requires IsValid_SIMD_Mask<N>
	inline bool SIMD_Mask<LS, N>::operator[](size_t i) const
	{
		return BitsUintT(*this) & (BitsUintT(1) << i);
	}

	template<meta::ScalarSizeClassEnum LS, size_t N>  requires IsValid_SIMD_Mask<N>
	inline void SIMD_Mask<LS, N>::setBit(size_t i, bool value)
	{
		BitsUintT u = *this;
		u &= ~(BitsUintT(1) << i);
		*this = u | BitsUintT(value) << i;
	}

	template<meta::ScalarSizeClassEnum LS, size_t N> requires IsValid_SIMD_Mask<N>
	inline auto SIMD_Mask<LS, N>::lo() const
		requires (N % 2 == 0 && IsValid_SIMD_Mask<N/2>)
	{
		static_assert(N % 2 == 0);
		SIMD_Mask<LS, N / 2> ret;
		if constexpr (IsBitMask) ret.underlying = underlying;
		else ret.underlying = underlying.lo();
		return ret;
	}
	template<meta::ScalarSizeClassEnum LS, size_t N> requires IsValid_SIMD_Mask<N>
	inline auto SIMD_Mask<LS, N>::hi() const
		requires (N % 2 == 0 && IsValid_SIMD_Mask<N / 2>)
	{
		static_assert(N % 2 == 0);
		SIMD_Mask<LS, N / 2> ret;
		if constexpr (IsBitMask) ret.underlying = underlying >> (N/2);
		else ret.underlying = underlying.hi();
		return ret;
	}

	template<meta::ScalarSizeClassEnum LS, size_t N> requires IsValid_SIMD_Mask<N>
	inline SIMD_Mask<LS, N>::operator BitsUintT() const
	{
		if constexpr (IsBitMask) return underlying & AllOnesUint;
		else return this->_movemask();
	}

	template<meta::ScalarSizeClassEnum LS, size_t N> requires IsValid_SIMD_Mask<N>
	template<typename T> requires (meta::IsIntrinsicVector<T> && (meta::ScalarSizeTraits<LS>::ByteSize* N == sizeof(T)))
	inline SIMD_Mask<LS, N>::operator T() const
	{
		if constexpr (IsBitMask) return movm<VecIntT>(*this);
		else return this->_movm(*this);
	}

	template<meta::ScalarSizeClassEnum LS, size_t N> requires IsValid_SIMD_Mask<N>
	inline SIMD_Mask<LS, N> SIMD_Mask<LS, N>::operator&(const SIMD_Mask<LS, N>& other) const
	{
		SIMD_Mask<LS, N> ret;
		ret.underlying = underlying & other.underlying;
		return ret;
	}
	template<meta::ScalarSizeClassEnum LS, size_t N> requires IsValid_SIMD_Mask<N>
	inline SIMD_Mask<LS, N> SIMD_Mask<LS, N>::operator|(const SIMD_Mask<LS, N>& other) const
	{
		SIMD_Mask<LS, N> ret;
		ret.underlying = underlying | other.underlying;
		return ret;
	}
	template<meta::ScalarSizeClassEnum LS, size_t N> requires IsValid_SIMD_Mask<N>
	inline SIMD_Mask<LS, N> SIMD_Mask<LS, N>::operator^(const SIMD_Mask<LS, N>& other) const
	{
		SIMD_Mask<LS, N> ret;
		ret.underlying = underlying ^ other.underlying;
		return ret;
	}

	template<meta::ScalarSizeClassEnum LS, size_t N>  requires IsValid_SIMD_Mask<N>
	inline SIMD_Mask<LS, N> SIMD_Mask<LS, N>::operator~() const
	{
		SIMD_Mask<LS, N> ret;
		ret.underlying = ~underlying;
		return ret;
	}

	template<meta::ScalarSizeClassEnum LS, size_t N>  requires IsValid_SIMD_Mask<N>
	inline SIMD_Mask<LS, N>& SIMD_Mask<LS, N>::operator&=(const SIMD_Mask<LS, N>& other)
	{
		*this = *this & other;
		return *this;
	}
	template<meta::ScalarSizeClassEnum LS, size_t N>  requires IsValid_SIMD_Mask<N>
	inline SIMD_Mask<LS, N>& SIMD_Mask<LS, N>::operator|=(const SIMD_Mask<LS, N>& other)
	{
		*this = *this | other;
		return *this;
	}
	template<meta::ScalarSizeClassEnum LS, size_t N>  requires IsValid_SIMD_Mask<N>
	inline SIMD_Mask<LS, N>& SIMD_Mask<LS, N>::operator^=(const SIMD_Mask<LS, N>& other)
	{
		*this = *this ^ other;
		return *this;
	}

	template<meta::ScalarSizeClassEnum LS, size_t N> requires IsValid_SIMD_Mask<N>
	template<meta::ScalarSizeClassEnum LS2, size_t N2> requires (N >= N2)
	inline SIMD_Mask<LS, N>::SIMD_Mask(const SIMD_Mask<LS2, N2>& other)
	{
		*this = other._movemask();
	}

	template<meta::ScalarSizeClassEnum LS, size_t N>
	std::ostream& operator<<(std::ostream& os, const SIMD_Mask<LS, N>& mask)
	{
		for (size_t i = 0; i < mask; ++i)
		{
			os << int(mask[i]);
			if (i < N - 1) os << ",";
		}
		return os;
	}
}