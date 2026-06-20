#pragma once
#include "SIMD_BitMask.h"
#include <iostream>
#include "SIMD_Mask.h"
#if 0
namespace AVXXY_NAMESPACE
{
	//AllOnes anding is to clear out garbage from upper unused bits
	template<size_t N>
	__forceinline AVXXY_NAMESPACE::SIMD_Mask<S,N>::SIMD_BitMask(UintT value)
	{
		underlying = value & AllOnes;
	}
	template<size_t N>
	inline SIMD_Mask<S,N>::SIMD_BitMask(const SIMD_BitMask<N / 2>& lo, const SIMD_BitMask<N / 2>& hi)
	{
		static_assert(N % 2 == 0);
		underlying = UintT(lo) | (UintT(hi) << N/2);
	}

	/*
	template<size_t N>
	template<typename T>
		requires (concepts::is_any_of_v<T, __m128, __m256, __m128d, __m256d>)
	inline SIMD_Mask<S,N>::SIMD_BitMask(const T& intrinsicVec)
	{
		*this = backends::current::intrinsic_vec_to_mask<N>(intrinsicVec);
	}

	template<size_t N>
	template<typename S, typename T>
		requires (concepts::is_any_of_v<T, __m128i, __m256i>&& std::is_integral_v<S>)
	inline SIMD_Mask<S,N>::SIMD_BitMask(const T& intrinsicVec)
	{
		*this = backends::current::intrinsic_vec_to_mask<N, S>(intrinsicVec);
	}*/

	template<size_t N>
	inline SIMD_Mask<S,N>::operator UintT() const
	{
		return underlying & AllOnes;
	}
	template<size_t N>
	inline bool SIMD_Mask<S,N>::operator[](size_t i) const
	{
		return underlying & (UintT(1) << i);
	}

	template<size_t N>
	inline void SIMD_Mask<S,N>::setBit(size_t i, bool value)
	{
		underlying &= ~(UintT(1) << i); //clear bit i
		underlying |= UintT(value) << i;
		underlying &= AllOnes;
	}
	
	template<size_t N>
	inline SIMD_Mask<S,N>::IntT SIMD_Mask<S,N>::as_int() const
	{
		return UintT(*this);
	}
	template<size_t N>
	inline SIMD_BitMask<N / 2> SIMD_Mask<S,N>::lo() const
	{
		return underlying;
	}
	template<size_t N>
	inline SIMD_BitMask<N / 2> SIMD_Mask<S,N>::hi() const
	{
		return underlying >> (N / 2);
	}
	
	template<size_t N>
	inline SIMD_Mask<S,N>& SIMD_Mask<S,N>::operator&=(const SIMD_Mask<S,N>& other)
	{
		*this = *this & other;
		return *this;
	}
	template<size_t N>
	inline SIMD_Mask<S,N>& SIMD_Mask<S,N>::operator|=(const SIMD_Mask<S,N>& other)
	{
		*this = *this | other;
		return *this;
	}
	template<size_t N>
	inline SIMD_Mask<S,N>& SIMD_Mask<S,N>::operator^=(const SIMD_Mask<S,N>& other)
	{
		*this = *this ^ other;
		return *this;
	}

	template <size_t N>
	static std::ostream& operator<<(std::ostream& os, const SIMD_Mask<S,N>& mask)
	{
		for (int i = 0; i < N; ++i) os << (mask[i] ? 1 : 0) << ",";
		os << (mask[N - 1] ? 1 : 0);
		return os;
	}
}
#endif