#pragma once
#include "namespace.h"

namespace AVXXY_NAMESPACE
{
	namespace capabilities
	{
		struct Arch
		{
			struct _AVX512
			{
				bool BMM = false;
				bool FP16 = false;
				bool VP2INTERSECT = false;
				bool BF16 = false;
				bool VAES = false;
				bool GFNI = false;
				bool VPCLMULQDQ = false;
				bool VNNI = false;
				bool BITALG = false;
				bool VPOPCNTDQ = false;
				bool VBMI2 = false;
				bool VBMI = false;
				bool IFMA = false;
				bool BW = false;
				bool DQ = false;
				bool VL = false;
				bool CD = false;
				bool F = false;

				constexpr bool operator==(const _AVX512& other) const = default;
			};
			_AVX512 AVX512;

			bool AVX2 = false;
			bool FMA3 = false;
			bool FP16C = false;

			bool AVX = false;
			bool AES = false;
			bool SSE42 = false;
			bool SSE41 = false;
			bool SSE4A = false;
			bool SSSE3 = false;
			bool SSE3 = false;
			bool SSE2 = false;
			bool SSE = false;
			bool MMX = false;
			//bool FPU = false;

			constexpr bool operator==(const Arch& other) const = default;
		};

		static constexpr Arch zen4 = []() {
			Arch a;
			Arch::_AVX512 x;
			x.BF16 = x.VAES = x.GFNI = x.VPCLMULQDQ = x.VNNI = x.BITALG = x.VPOPCNTDQ = x.VBMI2 = x.VBMI = x.IFMA = x.BW = x.DQ = x.VL = x.CD = x.F = true;
			a.AVX512 = x;

			//a.FPU = 
			a.MMX = a.SSE = a.SSE2 = a.SSE3 = a.SSE41 = a.SSE42 = a.SSE4A = a.SSSE3 = a.AES = a.AVX = a.AVX2 = a.FMA3 = true;
			a.FP16C = true;
			return a;
			}();

		static constexpr Arch current = Arch();
	}
}