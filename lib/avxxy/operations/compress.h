#pragma once
#include "shared.h"

namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		struct op_compress : OperationBase
		{
			template<typename S, size_t N>
			static SIMD_Vector<S, N> run(const mask_t<S,N>& mask, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& src)
			{
				scream();
				SIMD_Vector<S, N> ret;
				size_t j = 0;
				for (size_t i = 0; i < N; ++i) if (mask[i]) ret[j++] = a[i];
				for (; j < N; ++j) ret[j] = src[j];
				return ret;
			}

		private:
			
		};
	}
}