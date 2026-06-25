#pragma once
#include "namespace.h"
#include <stdint.h>
#include <string>
#include <iostream>
#include <vector>

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

			std::string toString(std::string sep = ", ") const
			{
				const FeatureSet& fs = *this;
				std::vector<std::string> features, features512;
				if (fs.has(MMX)) features.emplace_back("MMX");
				if (fs.has(SSE)) features.emplace_back("SSE");
				if (fs.has(SSE)) features.emplace_back("SSE2");
				if (fs.has(SSE)) features.emplace_back("SSE3");
				if (fs.has(SSE)) features.emplace_back("SSSE3");
				if (fs.has(SSE4A)) features.emplace_back("SSE4A");
				if (fs.has(SSE41)) features.emplace_back("SSE4.1");
				if (fs.has(SSE42)) features.emplace_back("SSE4.2");
				if (fs.has(AES)) features.emplace_back("AES");
				if (fs.has(AVX)) features.emplace_back("AVX");
				if (fs.has(F16C)) features.emplace_back("F16C");
				if (fs.has(FMA3)) features.emplace_back("FMA3");
				if (fs.has(AVX2)) features.emplace_back("AVX2");

				if (fs.has(AVX512_F)) features512.emplace_back("F");
				if (fs.has(AVX512_CD)) features512.emplace_back("CD");
				if (fs.has(AVX512_VL)) features512.emplace_back("VL");
				if (fs.has(AVX512_DQ)) features512.emplace_back("DQ");
				if (fs.has(AVX512_BW)) features512.emplace_back("BW");
				if (fs.has(AVX512_IFMA)) features512.emplace_back("IFMA");
				if (fs.has(AVX512_VBMI)) features512.emplace_back("VBMI");
				if (fs.has(AVX512_VBMI2)) features512.emplace_back("VBMI2");
				if (fs.has(AVX512_VPOPCNTDQ)) features512.emplace_back("VPOPCNTDQ");
				if (fs.has(AVX512_BITALG)) features512.emplace_back("BITALG");
				if (fs.has(AVX512_VNNI)) features512.emplace_back("VNNI");
				if (fs.has(AVX512_VPCLMULQDQ)) features512.emplace_back("VPCLMULQDQ");
				if (fs.has(AVX512_GFNI)) features512.emplace_back("GFNI");
				if (fs.has(AVX512_VAES)) features512.emplace_back("VAES");
				if (fs.has(AVX512_BF16)) features512.emplace_back("BF16");
				if (fs.has(AVX512_VP2INTERSECT)) features512.emplace_back("VP2INTERSECT");
				if (fs.has(AVX512_FP16)) features512.emplace_back("FP16");
				if (fs.has(AVX512_BMM)) features512.emplace_back("BMM");

				std::string ret;
				for (size_t i = 0; i < features.size(); ++i)
				{
					ret += features[i];
					if (i < features.size() - 1) ret += sep;
				}

				ret += '\n';
				if (!features512.empty()) ret += "AVX512: ";
				for (size_t i = 0; i < features512.size(); ++i)
				{
					ret += features512[i];
					if (i < features512.size() - 1) ret += sep;
				}
				return ret;
			}
		};



		static std::ostream& operator<<(std::ostream& os, const FeatureSet& fs)
		{
			os << fs.toString();
			return os;
		}
		static constexpr FeatureSet FS_nothing = { 0 };
		static constexpr FeatureSet FS_SSE = {
			Feature::MMX |
			Feature::SSE,
		};
		static constexpr FeatureSet FS_SSE2 = {
			FS_SSE._bits | Feature::SSE2
		};
		static constexpr FeatureSet FS_SSE41 = {
			FS_SSE2._bits | Feature::SSE3 | Feature::SSSE3 |
			Feature::SSE4A | Feature::SSE41
		};
		static constexpr FeatureSet FS_AVX = {
			FS_SSE41._bits | Feature::AES |
			Feature::AVX |
			Feature::F16C
		};
		static constexpr FeatureSet FS_AVX2 = {
			FS_AVX._bits | Feature::FMA3 | Feature::AVX2
		};

		static constexpr FeatureSet FS_SkylakeX = {
			FS_AVX2._bits |
			Feature::AVX512_F |
			Feature::AVX512_CD |
			Feature::AVX512_VL |
			Feature::AVX512_DQ |
			Feature::AVX512_BW
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
			Feature::AVX512_BF16 |
			0
		};

		static constexpr FeatureSet FS_compile_target = {
		#ifdef __MMX__
					Feature::MMX |
		#endif

		#ifdef __SSE__
					Feature::SSE |
		#endif

		#ifdef __SSE2__
					Feature::SSE2 |
		#endif

		#ifdef __SSE3__
					Feature::SSE3 |
		#endif

		#ifdef __SSSE3__
					Feature::SSSE3 |
		#endif

		#ifdef __SSE4A__
					Feature::SSE4A |
		#endif

		#ifdef __SSE4_1__
					Feature::SSE41 |
		#endif

		#ifdef __SSE4_2__
					Feature::SSE42 |
		#endif

		#ifdef __AES__
					Feature::AES |
		#endif

		#ifdef __AVX__
					Feature::AVX |
		#endif

		#ifdef __F16C__
					Feature::F16C |
		#endif

		#ifdef __FMA__
					Feature::FMA3 |
		#endif

		#ifdef __AVX2__
					Feature::AVX2 |
		#endif

		#ifdef __AVX512F__
					Feature::AVX512_F |
			#ifdef __AVX512CD__
					Feature::AVX512_CD |
		#endif

		#ifdef __AVX512VL__
					Feature::AVX512_VL |
		#endif

		#ifdef __AVX512DQ__
					Feature::AVX512_DQ |
		#endif

		#ifdef __AVX512BW__
					Feature::AVX512_BW |
		#endif

		#ifdef __AVX512IFMA__
					Feature::AVX512_IFMA |
		#endif

		#ifdef __AVX512VBMI__
					Feature::AVX512_VBMI |
		#endif

		#ifdef __AVX512VBMI2__
					Feature::AVX512_VBMI2 |
		#endif

		#ifdef __AVX512VPOPCNTDQ__
					Feature::AVX512_VPOPCNTDQ |
		#endif

		#ifdef __AVX512BITALG__
					Feature::AVX512_BITALG |
		#endif

		#ifdef __AVX512VNNI__
					Feature::AVX512_VNNI |
		#endif

		#ifdef __VPCLMULQDQ__
					Feature::AVX512_VPCLMULQDQ |
		#endif

		#ifdef __GFNI__
				Feature::AVX512_GFNI |
		#endif

		#ifdef __VAES__
				Feature::AVX512_VAES |
		#endif

		#ifdef __AVX512BF16__
				Feature::AVX512_BF16 |
		#endif

		#ifdef __AVX512VP2INTERSECT__ //TODO: unchecked
				Feature::AVX512_VP2INTERSECT |
		#endif

		#ifdef __AVX512FP16__ //TODO: unchecked
				Feature::AVX512_FP16 |
		#endif
		#ifdef __AVX512BMM__ //TODO: unchecked
				Feature::AVX512_BMM |
		#endif
		#endif
				0
		};
		static constexpr FeatureSet FS_current = FS_compile_target;
		static constexpr FeatureSet FS = FS_current;
	}
}