#pragma once
#include <array>
#include "namespace.h"
namespace AVXXY_NAMESPACE
{
	namespace tables
	{
		//Lookup table that maps 8-bit compress mask to 8-element permutexvar index register
				//It it compressed to only take up 1 byte per index, thus, it needs to be expanded after loading at the call site (from int8_t's to required signed(!!!) type)
				//Negative indices (when treated as int8_t) stored here means that value is masked out and should be passed through from source register.
				//It is uncertain if 4 bit LUT would be better. 2 KiB size is decently large, but greatly simplifies the compress emulation for 8-element vectors, replacing it just with 1 mask extraction + lookup + cvt + cross-lane permute
		static constexpr std::array<uint64_t, 256> compress_to_permx8 = []() {
			std::array<uint64_t, 256> ret;
			for (int i = 0; i < 256; ++i)
			{
				uint64_t val = 0xFFFFFFFFFFFFFFFF;
				int pivot = 0;
				for (int j = 0; j < 8; ++j)
				{
					if (i & (1 << j))
					{
						val &= ~(uint64_t(0xFF) << (8 * pivot));
						val |= uint64_t(j) << (8 * (pivot++));
					}
				}
				ret[i] = val;
			}
			return ret;
			}();
	}
}