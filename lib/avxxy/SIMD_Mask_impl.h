#pragma once
#include "SIMD_Mask.h"
#include <iostream>
#include "funcs.h"

namespace AVXXY_NAMESPACE
{
	template<concepts::LaneSizeEnum LS, size_t N>
	inline SIMD_Mask<LS, N> SIMD_Mask<LS, N>::AllOnes()
	{
		return (N == sizeof(UintT) * 8) ? ~UintT(0) : ((UintT(1) << N) - 1);
	}
	template<concepts::LaneSizeEnum LS, size_t N>
	inline SIMD_Mask<LS, N>::SIMD_Mask(UintT bits)
	{
		if constexpr (IsBitMask) this->underlying = bits & AllOnesUint;
		else
		{
			this->underlying = movm<IntT, N>(bits);
		}
	}

	template<concepts::LaneSizeEnum LS, size_t N>
	inline SIMD_Mask<LS, N>::SIMD_Mask(const SIMD_Mask<LS, N / 2>& lo, const SIMD_Mask<LS, N / 2>& hi)
	{
		if constexpr (IsBitMask) this->underlying = concat_bitmasks<N / 2>(lo.underlying, hi.underlying) & AllOnesUint;
		else this->underlying = { lo.underlying, hi.underlying };
	}

	/*
	template<concepts::LaneSizeEnum LS, size_t N>
	template <typename S>
	inline SIMD_Mask<LS, N>::SIMD_Mask(const SIMD_Vector<S, N>& v)
	{
		if constexpr (IsBitMask) this->underlying = movemask(v);
		else *this = movemask(v.underlying);
	}*/

	template<concepts::LaneSizeEnum LS, size_t N>
	inline SIMD_Mask<LS, N>::operator UintT() const
	{
		return this->as_uint();
	}

	template<concepts::LaneSizeEnum LS, size_t N>
	inline SIMD_Mask<LS, N>::UintT SIMD_Mask<LS, N>::as_uint() const
	{
		if constexpr (IsBitMask) return underlying & AllOnesUint;
		else return movemask(underlying);
	}
	template<concepts::LaneSizeEnum LS, size_t N>
	inline SIMD_Mask<LS, N>::IntT SIMD_Mask<LS, N>::as_int() const
	{
		return as_uint();
	}

	template<concepts::LaneSizeEnum LS, size_t N>
	inline bool SIMD_Mask<LS, N>::operator[](size_t i) const
	{
		using U = typename VecT::UintScalarT;
		if constexpr (IsBitMask) return underlying & (UintT(1) << i) & AllOnesUint;
		else
		{
			UintT u = movemask(underlying);
			return u & (U(1) << i) & AllOnesUint;
		}
	}

	template<concepts::LaneSizeEnum LS, size_t N>
	inline SIMD_Mask<LS, N> SIMD_Mask<LS, N>::operator&(const SIMD_Mask<LS, N>& other) const
	{
		return underlying & other.underlying;
	}

	template<concepts::LaneSizeEnum LS, size_t N>
	inline SIMD_Mask<LS, N> SIMD_Mask<LS, N>::operator|(const SIMD_Mask<LS, N>& other) const
	{
		return underlying | other.underlying;
	}

	template<concepts::LaneSizeEnum LS, size_t N>
	inline SIMD_Mask<LS, N> SIMD_Mask<LS, N>::operator^(const SIMD_Mask<LS, N>& other) const
	{
		return underlying ^ other.underlying;
	}

	template<concepts::LaneSizeEnum LS, size_t N>
	template<concepts::LaneSizeEnum LS2>
	inline SIMD_Mask<LS, N>::SIMD_Mask(const SIMD_Mask<LS2, N>& other)
	{
		if constexpr (IsBitMask) underlying = other.underlying;
		else if constexpr (LS == LS2) underlying = vcast<VecT>(other.underlying);
		else *this = UintT(other);
	}
	/*

	template<concepts::LaneSizeEnum LS, size_t N>
	template<typename T>
		requires (concepts::IsIntrinsicVector<T>&& std::is_convertible_v<SIMD_Vector<S, N>, T>)
	inline SIMD_Mask<LS, N>::operator T() const
	{
		return this->as_vector();
	}

	template<concepts::LaneSizeEnum LS, size_t N>
	template<typename T>
		requires (std::is_convertible_v<T, SIMD_Vector<S, N>>&& concepts::IsIntrinsicVector<T>)
	inline SIMD_Mask<LS, N>::SIMD_Mask(const T& intrVec)
	{
		*this = SIMD_Vector<S, N>(intrVec);
	}*/

	template<concepts::LaneSizeEnum LS, size_t N>
	template<typename S>
	inline SIMD_Vector<S, N> SIMD_Mask<LS, N>::as_vector() const
	{
		if constexpr (IsBitMask) return movm<S, N>(underlying & AllOnesUint);
		else
		{
			/*
			if constexpr (sizeof(S) == sizeof(IntT))
			{
				return vcast<SIMD_Vector<S, N>>(underlying < 0); //TODO: check all of it ensure strict masks! (elements in each lane are all zeroes or all ones)
			}
			else*/
				return movm<S, N>(movemask(underlying) & AllOnesUint);
		}
	}

	template<concepts::LaneSizeEnum LS, size_t N>
	template<typename T>
		requires (concepts::IsIntrinsicTypeThatCanHold<T, typename SIMD_Mask<LS,N>::VecT>)
	inline SIMD_Mask<LS, N> SIMD_Mask<LS, N>::constructNoClean(const T& intr)
	{
		SIMD_Mask<LS, N> ret;
		if constexpr (IsBitMask) ret.underlying = movemask(VecT(intr));
		else memcpy(&ret.underlying, &intr, std::min(sizeof(ret), sizeof(intr)));
	}

	/*
	template<concepts::LaneSizeEnum LS, size_t N>
	template<typename T>
		requires (concepts::SameRegisterSizeClass<T, typename SIMD_Mask<LS,N>::VecT> && !concepts::IsScalarType<T>)
	inline SIMD_Mask<LS, N> SIMD_Mask<LS, N>::constructNoClean(const T& intr)
	{
		SIMD_Mask<LS, N> ret;
		if constexpr (IsBitMask) ret.underlying = movemask(VecT(intr)) & ret.AllOnesUint;
		else return ret.underlying = intr;
	}*/

	template<concepts::LaneSizeEnum LS, size_t N>
	inline SIMD_Mask<LS, N> SIMD_Mask<LS, N>::operator~() const
	{
		if constexpr (IsBitMask) return ~underlying;
		else return logic_not(underlying);
	}

	template<concepts::LaneSizeEnum LS, size_t N>
	inline SIMD_Mask<LS, N>& SIMD_Mask<LS, N>::operator&=(const SIMD_Mask<LS, N>& other)
	{
		*this = *this & other;
		return *this;
	}
	template<concepts::LaneSizeEnum LS, size_t N>
	inline SIMD_Mask<LS, N>& SIMD_Mask<LS, N>::operator|=(const SIMD_Mask<LS, N>& other)
	{
		*this = *this | other;
		return *this;
	}
	template<concepts::LaneSizeEnum LS, size_t N>
	inline SIMD_Mask<LS, N>& SIMD_Mask<LS, N>::operator^=(const SIMD_Mask<LS, N>& other)
	{
		*this = *this ^ other;
		return *this;
	}

	template<concepts::LaneSizeEnum LS, size_t N>
	inline SIMD_Mask<LS, N / 2> SIMD_Mask<LS, N>::lo() const
	{
		static_assert(N % 2 == 0);
		if constexpr (IsBitMask) return underlying;
		else return SIMD_Mask<LS,N/2>::constructNoClean(underlying.lo());
	}
	template<concepts::LaneSizeEnum LS, size_t N>
	inline SIMD_Mask<LS, N / 2> SIMD_Mask<LS, N>::hi() const
	{
		static_assert(N % 2 == 0);
		if constexpr (IsBitMask) return underlying >> (sizeof(UintT) * 4);
		else return SIMD_Mask<LS, N / 2>::constructNoClean(underlying.hi());
	}

	template<concepts::LaneSizeEnum LS, size_t N>
	static std::ostream& operator<<(std::ostream& os, const SIMD_Mask<LS, N>& mask)
	{
		for (int i = 0; i < N; ++i) os << (mask[i] ? 1 : 0) << ",";
		os << (mask[N - 1] ? 1 : 0);
		return os;
	}
}