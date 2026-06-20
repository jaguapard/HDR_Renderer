#pragma once
#include "SIMD_Mask.h"
#include <iostream>
#include "funcs.h"

namespace AVXXY_NAMESPACE
{
	template<typename S, size_t N>
	inline SIMD_Mask<S, N> SIMD_Mask<S, N>::AllOnes()
	{
		return (N == sizeof(UintT) * 8) ? ~UintT(0) : ((UintT(1) << N) - 1);
	}
	template<typename S, size_t N>
	inline SIMD_Mask<S, N>::SIMD_Mask(UintT bits)
	{
		if constexpr (IsBitMask) this->underlying = bits & AllOnesUint;
		else
		{
			this->underlying = movm<S, N>(bits);
			//TODO: bits to vector mask conversion here!
			//this->underlying = Default
		}
	}

	template<typename S, size_t N>
	inline SIMD_Mask<S, N>::SIMD_Mask(const SIMD_Mask<S, N / 2>& lo, const SIMD_Mask<S, N / 2>& hi)
	{
		if constexpr (IsBitMask) this->underlying = (UintT(lo.underlying) | (U(hi.underlying) << (N / 2))) & AllOnesUint;
		else this->underlying = { lo.underlying, hi.underlying };
	}

	template<typename S, size_t N>
	inline SIMD_Mask<S, N>::SIMD_Mask(const SIMD_Vector<S, N>& v)
	{
		if constexpr (IsBitMask) this->underlying = movemask(v);
		else this->underlying = v;
	}

	template<typename S, size_t N>
	inline SIMD_Mask<S, N>::operator UintT() const
	{
		return this->as_uint();
	}

	template<typename S, size_t N>
	inline typename SIMD_Mask<S, N>::UintT SIMD_Mask<S, N>::as_uint() const
	{
		if constexpr (IsBitMask) return underlying & AllOnesUint;
		else return movemask(underlying);
	}
	template<typename S, size_t N>
	inline typename SIMD_Mask<S, N>::IntT SIMD_Mask<S, N>::as_int() const
	{
		return as_uint();
	}

	template<typename S, size_t N>
	inline typename SIMD_Mask<S,N>::VecT SIMD_Mask<S, N>::as_vector() const
	{
		if constexpr (IsBitMask) return movm(underlying & AllOnesUint);
		else return underlying; //TODO: clean it (i.e. ensure all values are 0 or 0xFFFFFFF)
	}

	template<typename S, size_t N>
	inline bool SIMD_Mask<S, N>::operator[](size_t i) const
	{
		using U = typename VecT::UintScalarT;
		if constexpr (IsBitMask) return underlying & (UintT(1) << i) & AllOnesUint;
		else
		{
			UintT u = movemask(underlying);
			return u & (U(1) << i) & AllOnesUint;
		}
	}

	template<typename S, size_t N>
	inline SIMD_Mask<S, N> SIMD_Mask<S, N>::operator&(const SIMD_Mask<S, N>& other) const
	{
		return underlying & other.underlying; 
	}

	template<typename S, size_t N>
	inline SIMD_Mask<S, N> SIMD_Mask<S, N>::operator|(const SIMD_Mask<S, N>& other) const
	{
		return underlying | other.underlying;
	}

	template<typename S, size_t N>
	inline SIMD_Mask<S, N> SIMD_Mask<S, N>::operator^(const SIMD_Mask<S, N>& other) const
	{
		return underlying ^ other.underlying;
	}

	template<typename S, size_t N>
	template<typename S2>
	inline SIMD_Mask<S, N>::SIMD_Mask(const SIMD_Mask<S2, N>& other)
	{
		if constexpr (IsBitMask) underlying = other.underlying;
		else if constexpr (sizeof(S) == sizeof(S2)) underlying = vcast<VecT>(other.underlying);
		else *this = UintT(other);
	}

	template<typename S, size_t N>
	template<typename T>
		requires (std::is_convertible_v<T, SIMD_Vector<S, N>>&& concepts::IsIntrinsicVector<T>)
	inline SIMD_Mask<S, N>::SIMD_Mask(const T& intrVec)
	{
		*this = SIMD_Vector<S, N>(intrVec);
	}

	template<typename S, size_t N>
	inline SIMD_Mask<S, N> SIMD_Mask<S, N>::operator~() const
	{
		if constexpr (IsBitMask) return ~underlying;
		else return logic_not(underlying);
	}

	template<typename S, size_t N>
	inline SIMD_Mask<S, N>& SIMD_Mask<S, N>::operator&=(const SIMD_Mask<S, N>& other)
	{
		*this = *this & other;
		return *this;
	}
	template<typename S, size_t N>
	inline SIMD_Mask<S, N>& SIMD_Mask<S, N>::operator|=(const SIMD_Mask<S, N>& other)
	{
		*this = *this | other;
		return *this;
	}
	template<typename S, size_t N>
	inline SIMD_Mask<S, N>& SIMD_Mask<S, N>::operator^=(const SIMD_Mask<S, N>& other)
	{
		*this = *this ^ other;
		return *this;
	}

	template<typename S, size_t N>
	static std::ostream& operator<<(std::ostream& os, const SIMD_Mask<S, N>& mask)
	{
		for (int i = 0; i < N; ++i) os << (mask[i] ? 1 : 0) << ",";
		os << (mask[N - 1] ? 1 : 0);
		return os;
	}
}