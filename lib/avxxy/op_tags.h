#pragma once
#include "namespace.h"

namespace AVXXY_NAMESPACE
{
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
		template <typename To> struct op_cvt { using cvt_to_t = typename To; };
		struct op_div {};
		struct op_floor {};
		struct op_fp16_to_fp32 {};
		struct op_fp32_to_fp16 {};
		template <typename _S, size_t _N, size_t _Scale = sizeof(_S)> struct op_gather {
			using S = _S;
			//static constexpr size_t N = _N;
			static constexpr bool _avxxy_is_gather_tag = true;
		};
		template <typename _S, size_t _N> struct op_load {
			using S = _S;
			static constexpr size_t N = _N;
			static constexpr bool _avxxy_is_load_tag = true; 
		};
		struct op_mask_mov {};
		struct op_maskz_mov {};
		struct op_max {};
		struct op_min {};
		struct op_mod {};
		struct op_movemask {};
		template <typename _S> struct op_movm {
			using S = _S;
			static constexpr bool _avxxy_is_movm_tag = true;
		};
		struct op_mul {};
		struct op_not {};
		struct op_or {};
		struct op_permx {};
		struct op_permx2 {};
		template <size_t _Scale> struct op_scatter {
			static constexpr size_t Scale = _Scale;
			static constexpr bool _avxxy_is_scatter_tag = true;
		};
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
}