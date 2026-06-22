#pragma once
#include "../namespace.h"

namespace AVXXY_NAMESPACE
{
	namespace meta
	{
		enum class VectorSizeClassEnum
		{
			XMM = 16, YMM = 32, ZMM = 64, XL = 65,
		};

		enum class ScalarSizeClassEnum
		{
			byte = 1, word = 2, dword = 4, qword = 8,
		};
	}
}