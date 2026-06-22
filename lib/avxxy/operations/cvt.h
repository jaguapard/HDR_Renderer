#pragma once
#include "shared.h"

namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		struct op_cvt : OperationBase
		{
			template<typename To, size_t N, typename From>
			static SIMD_Vector<To, N> run(const SIMD_Vector<From, N>& a)
			{
				scream();
				SIMD_Vector<To, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = a[i];
				return ret;
			}

		private:
			
		};
	}
}