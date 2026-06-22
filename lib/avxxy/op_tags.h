#pragma once
#include "namespace.h"

namespace AVXXY_NAMESPACE
{
#if 0
	namespace internals
	{
		struct op_abs {};
		struct op_add {};
		struct op_and {};
		struct op_ceil {};
		struct op_cmpeq {};
		struct op_cmpge {};
		struct op_cmpgt {};
		struct op_cmple {};
		struct op_cmplt {};
		struct op_cmpneq {};
		struct op_compress {};
		struct op_conflict {};
		template <typename To> struct op_cvt {};
		struct op_div {};
		struct op_floor {};
		struct op_fp16_to_fp32 {};
		struct op_fp32_to_fp16 {};
		template <typename S, size_t N, size_t Scale = sizeof(S)> struct op_gather {};
		template <typename S, size_t N> struct op_load {};
		struct op_mask_mov {};
		struct op_maskz_mov {};
		struct op_max {};
		struct op_min {};
		struct op_mod {};
		struct op_movemask {};
		template <typename S, size_t N> struct op_movm {};
		struct op_mul {};
		struct op_not {};
		struct op_or {};
		struct op_permx {};
		struct op_permx2 {};
		template <size_t Scale> struct op_scatter {};
		struct op_shl {};
		struct op_shr {};
		struct op_sqrtd {};
		struct op_sqrtf {};
		struct op_store {};
		struct op_sub {};
		struct op_unpackhi {};
		struct op_unpacklo {};
		struct op_xor {};
	}
#endif
}