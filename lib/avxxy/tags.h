#pragma once
#include "namespace.h"

namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		template <typename To> struct op_cvt {};
		struct op_add {};
		struct op_sub {};
		struct op_mul {};
		struct op_div {};
		struct op_mod {};
		struct op_or {};
		struct op_and {};
		struct op_xor {};
		struct op_not {};
		struct op_shr {};
		struct op_shl {};
		struct op_sqrtf {};
		struct op_sqrtd {};
		struct op_permx {};
		struct op_permx2 {};
		struct op_floor {};
		struct op_ceil {};
		template <typename S, size_t N> struct op_load {};
		struct op_store {};
		template <typename S, size_t N, size_t Scale = sizeof(S)> struct op_gather {};
		template <size_t Scale> struct op_scatter{};
		struct op_compress {};
		struct op_mask_mov {};
		struct op_maskz_mov {};
		struct op_unpacklo {};
		struct op_unpackhi {};
		struct op_cmpeq {};
		struct op_cmpneq {};
		struct op_cmplt {};
		struct op_cmple {};
		struct op_cmpgt {};
		struct op_cmpge {};
		struct op_abs {};
		struct op_min {};
		struct op_max {};
		struct op_fp16_to_fp32 {};
		struct op_fp32_to_fp16 {};

		struct op_vec2mask {};
		struct op_mask2vec {};
	}
}