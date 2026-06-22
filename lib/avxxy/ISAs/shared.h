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
		//This type is returned when implementation was considered for evaluation, but didn't return anything else (i.e. failed due to size, type, accompanying ISA absent or other requirements)
		struct fail_ack_t {};
		//This type is returned when a function doesn't return a value, but needs to indicate success to the dispatcher (for example, scatter or store operations)
		struct success_ack_t {};
	}
}