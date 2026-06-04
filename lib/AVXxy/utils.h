#pragma once
#include "namespace.h"
#include <type_traits>

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

		static constexpr size_t aligned_size(size_t size, size_t alignment)
		{
			if (size % alignment == 0) return size;
			return ((size / alignment) + 1) * alignment;
		}

		static constexpr size_t ceil_div(size_t numerator, size_t denominator)
		{
			if (numerator % denominator == 0) return numerator / denominator;
			return numerator / denominator + 1;
		}

		template<typename T, size_t count, size_t alignmentRequirement>
		static constexpr size_t padded_element_count()
		{
			constexpr size_t sz = sizeof(T) * count;
			if (sz % alignmentRequirement == 0) return count;
			return (((sz / alignmentRequirement) + 1) * sz) / sizeof(alignmentRequirement);
		}

		static constexpr bool inRange(size_t val, size_t min, size_t max)
		{
			return val >= min && val <= max;
		}
	}
}