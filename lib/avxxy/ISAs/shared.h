#pragma once
#include "../SIMD_Mask.h"
#include "../SIMD_Vector.h"
#include "../Dispatcher.h"
#include "../op_tags.h"
#include <source_location>

namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		struct null_t {};
		struct alive_sentinel_t {};
	}
}