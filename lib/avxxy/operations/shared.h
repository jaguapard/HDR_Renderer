#pragma once
#include "../namespace.h"
#include "../FeatureSet.h"
#include "../SIMD_Vector.h"
#include "../SIMD_Mask.h"
#include "../meta/meta.h"
#include <source_location>

namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		using namespace meta;
		struct OperationBase
		{
			static inline constexpr internals::FeatureSet FS = internals::FS_current;
			//Maximum vector size of bytes, or 0 if no vector operations are supported
			//Value of 0 can be here only if scalar operations supported
			static inline constexpr size_t MaxVectorSize = []() {
				if constexpr (FS.has(AVX512_F)) return 64;
				else if constexpr (FS.has(AVX)) return 32;
				else if constexpr (FS.has(SSE)) return 16;
				else return 0;
				}();
			/*
			template<size_t VecSizeThreshold, typename... Args>
			static auto split_trivial(Args&&... args)
			{
				if constexpr (requires {split_trivial(std::forward<Args>(args)...); }
			}*/

			//scream your lungs out if scalar fallback is reached and this function is enabled via AVXXY_NOISY_SCALAR define
			static void scream(std::source_location loc = std::source_location::current())
			{
#ifdef AVXXY_NOISY_SCALAR
				std::cout << "\nScalar fallback reached:" << loc.function_name() << "\n";
#endif
			}

			template<typename S, size_t N, bool Lo>
			static SIMD_Vector<S, N> scalar_unpack_base(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				SIMD_Vector<S, N> ret;
				constexpr size_t pairs_per_xmm = 8 / sizeof(S); //8, since unpack only processes lower/upper half of each input
				constexpr size_t elements_per_xmm = 16 / sizeof(S); //how much elements of type S fit into one 128 bit lane
				constexpr size_t xmm_count = sizeof(ret) / 16;
				for (size_t xmm_i = 0; xmm_i < xmm_count; ++xmm_i) //for each 128-bit lane
				{
					for (size_t i = 0; i < elements_per_xmm; i += 2)
					{
						size_t srcI = xmm_i * elements_per_xmm + i / 2 + (Lo ? 0 : elements_per_xmm / 2);
						ret[xmm_i * elements_per_xmm + i] = a[srcI];
						ret[xmm_i * elements_per_xmm + i + 1] = b[srcI];
					}
				}
				return ret;
			}
		};
	}
}