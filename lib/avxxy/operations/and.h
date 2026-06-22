#pragma once
#include "shared.h"
namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		struct op_and : OperationBase
		{
			template<typename S, size_t N>
			static SIMD_Vector<S, N> run(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				using U = meta::ScalarTraits<S>::UintT;
				for (size_t i = 0; i < N; ++i) ret[i] = std::bit_cast<S>(U(std::bit_cast<U>(a[i]) & std::bit_cast<U>(b[i])));
				return ret;
			}
		};
	}
}