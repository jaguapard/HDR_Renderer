#pragma once
#include "shared.h"
namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		struct op_cmplt : OperationBase
		{
			template<typename S, size_t N>
			static mask_t<S, N> run(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				typename SIMD_Vector<S, N>::MaskT ret = 0;
				for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] < b[i]);
				return ret;
			}
		};
	}
}