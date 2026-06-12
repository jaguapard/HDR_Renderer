#pragma once
#include "namespace.h"
#include <stdint.h>

namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		enum Feature : uint64_t
		{
			MMX = 1ull << 0,
			SSE = 1ull << 1,
			SSE2 = 1ull << 2,
			SSE3 = 1ull << 3,
			SSSE3 = 1ull << 4,
			SSE4A = 1ull << 5,
			SSE41 = 1ull << 6,
			SSE42 = 1ull << 7,
			AES = 1ull << 8,
			AVX = 1ull << 9,
			F16C = 1ull << 10,
			FMA3 = 1ull << 11,
			AVX2 = 1ull << 12,

			AVX512_F = 1ull << 13,
			AVX512_CD = 1ull << 14,
			AVX512_VL = 1ull << 15,
			AVX512_DQ = 1ull << 16,
			AVX512_BW = 1ull << 17,
			AVX512_IFMA = 1ull << 18,
			AVX512_VBMI = 1ull << 19,
			AVX512_VBMI2 = 1ull << 20,
			AVX512_VPOPCNTDQ = 1ull << 21,
			AVX512_BITALG = 1ull << 22,
			AVX512_VNNI = 1ull << 23,
			AVX512_VPCLMULQDQ = 1ull << 24,
			AVX512_GFNI = 1ull << 25,
			AVX512_VAES = 1ull << 26,
			AVX512_BF16 = 1ull << 27,
			AVX512_VP2INTERSECT = 1ull << 28,
			AVX512_FP16 = 1ull << 29,
			AVX512_BMM = 1ull << 30,
		};

		struct FeatureSet
		{
			uint64_t _bits = 0;
			constexpr bool has(Feature feature) const
			{
				return (_bits & static_cast<uint64_t>(feature)) != 0;
			}
		};

		static constexpr FeatureSet FS_zen4 = {
			Feature::MMX |
			Feature::SSE |
			Feature::SSE2 |
			Feature::SSE3 |
			Feature::SSSE3 |
			Feature::SSE4A |
			Feature::SSE41 |
			Feature::SSE42 |
			Feature::AES |
			Feature::AVX |
			Feature::F16C |
			Feature::FMA3 |
			Feature::AVX2 |
			Feature::AVX512_F |
			Feature::AVX512_CD |
			Feature::AVX512_VL |
			Feature::AVX512_DQ |
			Feature::AVX512_BW |
			Feature::AVX512_IFMA |
			Feature::AVX512_VBMI |
			Feature::AVX512_VBMI2 |
			Feature::AVX512_VPOPCNTDQ |
			Feature::AVX512_BITALG |
			Feature::AVX512_VNNI |
			Feature::AVX512_VPCLMULQDQ |
			Feature::AVX512_GFNI |
			Feature::AVX512_VAES |
			Feature::AVX512_BF16 };

		static constexpr FeatureSet FS_current = FS_zen4;// FeatureSet();
	}
}