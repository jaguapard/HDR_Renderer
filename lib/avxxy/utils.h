#pragma once
#include "namespace.h"
namespace AVXXY_NAMESPACE
{
	namespace utils
	{
		static constexpr bool isPowerOf2(size_t N)
		{
			if (N <= 2) return true;
			if (N % 2 != 0) return false;
			return isPowerOf2(N / 2);
		}
		static constexpr bool inRange(size_t val, size_t min, size_t max)
		{
			return val >= min && val <= max;
		}

		//Returns true if the given size (in bytes) fits into XMM register
		static constexpr bool is_xmm_size(size_t N) { return inRange(N, 0, 16); }
		//Returns true if the given size (in bytes) fits into YMM register, but doesn't fit into XMM register
		static constexpr bool is_ymm_size(size_t N) { return inRange(N, 17, 32); }
		//Returns true if the given size (in bytes) fits into ZMM register, but doesn't fit into XMM or YMM registers
		static constexpr bool is_zmm_size(size_t N) { return inRange(N, 33, 64); }
		//Returns true if the given size (in bytes) is too large to be held by ZMM register.
		static constexpr bool is_XL_size(size_t N) { return !(is_xmm_size(N) || is_ymm_size(N) || is_zmm_size(N)); }
	}
}