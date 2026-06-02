#pragma once
#include "namespace.h"
#include "SIMD_Mask.h"
namespace AVXXY_NAMESPACE
{
	template <size_t N>
	requires (N > 0 && N <= 64)
	__forceinline bool SIMD_Mask<N>::operator[](size_t i) const
	{
		return bits & (SIMD_Mask<N>::UintType(1) << i);
	}
	/*
	template <size_t N>
	requires (N > 64)
	__forceinline bool SIMD_Mask<N>::operator[](size_t i) const
	{
		return bits[i / 64] & (1ull << (i % 64));
	}*/

	template <size_t N>
	std::ostream& operator<<(std::ostream& os, const SIMD_Mask<N>& mask)
	{
		for (int i = 0; i < N; ++i) os << (mask[i] ? 1 : 0) << ",";
		os << (mask[N - 1] ? 1 : 0);
		return os;
	}
}