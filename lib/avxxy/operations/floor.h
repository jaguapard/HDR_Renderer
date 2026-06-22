#pragma once
#include "shared.h"
namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		struct op_floor : OperationBase
		{
			template<typename S, size_t N>
			requires (any_float<S>)
			static SIMD_Vector<S, N> run(const SIMD_Vector<S, N>& a)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = std::floor(a[i]);
				return ret;
			}
		};
	}
}