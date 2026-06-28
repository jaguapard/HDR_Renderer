#pragma once
#include <array>
#include "namespace.h"
namespace AVXXY_NAMESPACE
{
	namespace tables
	{
		//Lookup table that maps 8-bit compress mask to 8-element permutexvar index register
		//It it compressed to only take up 1 byte per index, thus, it needs to be expanded after loading at the call site (from int8_t's to required signed(!!!) type)
		//Negative indices (when treated as int8_t) stored here mean that value is masked out and should be passed through from source register.
		//It is uncertain if 4 bit LUT would be better. 2 KiB size is decently large, but greatly simplifies the compress emulation for 8-element vectors, replacing it just with 1 mask extraction + lookup + cvt + cross-lane permute
		alignas(32) static constexpr std::array<uint64_t, 256> compress_to_permx8 = []() {
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


		//Maps mask (4 bits) to index registers for xmm pshufb for compress emulation (16 bytes)
		//Negative values indicate that value should be passed through from src.
		//Due to small size, this table is not compressed, unlike the one for 8 elements
		//pshufb shuffling bytes instead of native elements also means that this table cannot be used for other sizes without change
		alignas(16) static constexpr std::array<int8_t, 256> compress_dwords_pshufb = []() {
			std::array<int8_t, 256> ret;
			std::fill(ret.begin(), ret.end(), -1);
			for (int i = 0; i < 16; ++i)
			{
				int pivot = 0;
				for (int j = 0; j < 4; ++j)
				{
					bool bit = i & (1 << j);
					if (bit)
					{
						for (int k = 0; k < 4; ++k)
						{
							ret[i * 16 + pivot++] = j * 4 + k;
						}
					}
				}
			}
			return ret;
			}();

		//64-entry popcnt table for nibbles (duplicated 4 times)
		//element at index i equals to popcnt(i % 16). This property allows this table to be used in _mm*_shuffle_epi8 family of instructions
		alignas(64) static constexpr std::array<int8_t, 64> popcnt_table_for_nibbles_as_epi8 = []() {
			std::array<int8_t, 64> ret;
			for (size_t i = 0; i < 64; ++i) ret[i] = std::popcount(i % 16);
			return ret;
			}();
	}
}