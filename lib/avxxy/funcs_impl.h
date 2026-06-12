#pragma once
#include "funcs.h"
#include "Dispatcher.h"

namespace AVXXY_NAMESPACE
{
	template<typename S, size_t N>
	inline SIMD_Vector<S, N> add(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::DefaultDispatcher::run(internals::op_add{}, a, b);
	}
	template<typename S, size_t N>
	inline SIMD_Vector<S, N> sub(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::DefaultDispatcher::run(internals::op_sub{}, a, b);
	}
	template<typename S, size_t N>
	inline SIMD_Vector<S, N> mul(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::DefaultDispatcher::run(internals::op_mul{}, a, b);
	}
	template<typename S, size_t N>
	inline SIMD_Vector<S, N> div(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::DefaultDispatcher::run(internals::op_div{}, a, b);
	}
	template<typename S, size_t N>
	inline SIMD_Vector<S, N> logic_and(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::DefaultDispatcher::run(internals::op_and{}, a, b);
	}
	template<typename S, size_t N>
	inline SIMD_Vector<S, N> logic_or(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::DefaultDispatcher::run(internals::op_or{}, a, b);
	}
	template<typename S, size_t N>
	inline SIMD_Vector<S, N> logic_xor(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::DefaultDispatcher::run(internals::op_xor{}, a, b);
	}
	template<typename S, size_t N>
	inline SIMD_Vector<S, N> logic_not(const SIMD_Vector<S, N>& a)
	{
		return internals::DefaultDispatcher::run(internals::op_not{}, a);
	}

	template<typename S, size_t N, typename I>
	inline SIMD_Vector<S, N> shift_left(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& amount)
	{
		return internals::DefaultDispatcher::run(internals::op_shl{}, a, amount);
	}
	template<typename S, size_t N, typename I>
	inline SIMD_Vector<S, N> shift_right(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& amount)
	{
		return internals::DefaultDispatcher::run(internals::op_shr{}, a, amount);
	}

	template<typename S, size_t N, typename I>
	inline SIMD_Vector<S, N> permx(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& ind)
	{
		return internals::DefaultDispatcher::run(internals::op_permx{}, a, ind);
	}

	template<typename S, size_t N, typename I>
	inline SIMD_Vector<S, N> permx2(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b, const SIMD_Vector<I, N>& ind)
	{
		return internals::DefaultDispatcher::run(internals::op_permx2{}, a, b, ind);
	}

	template<typename S, size_t N>
	inline SIMD_Vector<float, N> sqrtf(const SIMD_Vector<S, N>& a)
	{
		return internals::DefaultDispatcher::run(internals::op_sqrtf{}, a);
	}

	template<typename S, size_t N>
	inline SIMD_Vector<double, N> sqrtd(const SIMD_Vector<S, N>& a)
	{
		return internals::DefaultDispatcher::run(internals::op_sqrtd{}, a);
	}

	template<typename To, size_t N, typename From>
	inline SIMD_Vector<To, N> vcvt(const SIMD_Vector<From, N>& value)
	{
		return internals::DefaultDispatcher::run(internals::op_cvt<To>{}, value);
	}

	template<typename S, size_t N>
	inline SIMD_Vector<S, N> abs(const SIMD_Vector<S, N>& a)
	{
		return internals::DefaultDispatcher::run(internals::op_abs{}, a);
	}

	template<typename S, size_t N>
	inline SIMD_Vector<S, N> floor(const SIMD_Vector<S, N>& a)
	{
		return internals::DefaultDispatcher::run(internals::op_floor{}, a);
	}

	template<typename S, size_t N>
	inline SIMD_Vector<S, N> ceil(const SIMD_Vector<S, N>& a)
	{
		return internals::DefaultDispatcher::run(internals::op_ceil{}, a);
	}

	template<typename S, size_t N>
	inline SIMD_Vector<S, N> min(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::DefaultDispatcher::run(internals::op_min{}, a, b);
	}

	template<typename S, size_t N>
	inline SIMD_Vector<S, N> max(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::DefaultDispatcher::run(internals::op_max{}, a, b);
	}

	template<typename S, size_t N>
	inline SIMD_Vector<S, N> clamp(const SIMD_Vector<S, N>& val, const SIMD_Vector<S, N>& min, const SIMD_Vector<S, N>& max)
	{
		return AVXXY_NAMESPACE::max(min, AVXXY_NAMESPACE::min(val, max));
	}

	template<typename T, typename S, size_t N>
	//	requires (T::IsSimdVector)
	inline T vcast(const SIMD_Vector<S, N>& value)
	{
		T ret;
		memcpy(&ret, &value, std::min(sizeof(ret), sizeof(value)));
		return ret;
	}

	template<typename S, size_t N>
	inline SIMD_Vector<S, N> mask_mov(const SIMD_Vector<S, N>& ifBitClear, SIMD_BitMask<N>& mask, const SIMD_Vector<S, N>& ifBitSet)
	{
		return internals::DefaultDispatcher::run(internals::op_mask_mov{}, ifBitClear, mask, ifBitSet);
	}

	template<typename S, size_t N>
	inline SIMD_Vector<S, N> maskz_mov(const SIMD_BitMask<N>& mask, const SIMD_Vector<S, N>& ifBitSet)
	{
		return mask_mov(SIMD_Vector<S, N>(0), mask, ifBitSet);
	}

	template<typename S, size_t N>
	inline SIMD_Vector<S, N> blend(const SIMD_BitMask<N>& mask, const SIMD_Vector<S, N>& ifBitClear, const SIMD_Vector<S, N>& ifBitSet)
	{
		return mask_mov(ifBitClear, mask, ifBitSet);
	}

	template<typename S, size_t N>
	inline SIMD_Vector<S, N> load(const void* p, const SIMD_BitMask<N>& mask, const SIMD_Vector<S, N>& src)
	{
		return internals::DefaultDispatcher::run(internals::op_load<S, N>{}, p, mask, src);
	}

	template<typename S, size_t N>
	inline void store(const SIMD_Vector<S, N>& v, void* p, const SIMD_BitMask<N>& mask)
	{
		return internals::DefaultDispatcher::run(internals::op_store{}, v, p, mask);
	}

	template<typename S, size_t N, size_t Scale, typename I>
	inline SIMD_Vector<S, N> gather(const void* base, const SIMD_Vector<I, N>& ind, const SIMD_BitMask<N>& mask, const SIMD_Vector<S, N>& src)
	{
		return internals::DefaultDispatcher::run(internals::op_gather<S, N, Scale>{}, base, ind, mask, src);
	}

	template<typename S, size_t N, size_t Scale, typename I>
	inline void scatter(const SIMD_Vector<S, N>& vec, void* base, const SIMD_Vector<I, N>& ind, const SIMD_BitMask<N>& mask)
	{
		return internals::DefaultDispatcher::run(internals::op_scatter<Scale>{}, vec, base, ind, mask);
	}

	template<typename S, size_t N>
	inline SIMD_BitMask<N> cmp_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::DefaultDispatcher::run(internals::op_cmpeq{}, a, b);
	}
	template<typename S, size_t N>
	inline SIMD_BitMask<N> cmp_not_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::DefaultDispatcher::run(internals::op_cmpneq{}, a, b);
	}
	template<typename S, size_t N>
	inline SIMD_BitMask<N> cmp_less(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::DefaultDispatcher::run(internals::op_cmplt{}, a, b);
	}
	template<typename S, size_t N>
	inline SIMD_BitMask<N> cmp_less_or_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::DefaultDispatcher::run(internals::op_cmple{}, a, b);
	}
	template<typename S, size_t N>
	inline SIMD_BitMask<N> cmp_greater(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::DefaultDispatcher::run(internals::op_cmpgt{}, a, b);
	}
	template<typename S, size_t N>
	inline SIMD_BitMask<N> cmp_greater_or_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::DefaultDispatcher::run(internals::op_cmpge{}, a, b);
	}

	template<typename S, size_t N>
	inline SIMD_Vector<S, N> unpacklo(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::DefaultDispatcher::run(internals::op_unpacklo{}, a, b);
	}
	template<typename S, size_t N>
	inline SIMD_Vector<S, N> unpackhi(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		return internals::DefaultDispatcher::run(internals::op_unpackhi{}, a, b);
	}

	template<size_t N>
	inline SIMD_Vector<float, N> vcvt_fp16_fp32(const SIMD_Vector<uint16_t, N>& a)
	{
		return internals::DefaultDispatcher::run(internals::op_fp16_to_fp32{}, a);
	}
	template<size_t N>
	inline SIMD_Vector<uint16_t, N> vcvt_fp32_fp16(const SIMD_Vector<float, N>& a)
	{
		return internals::DefaultDispatcher::run(internals::op_fp32_to_fp16{}, a);
	}
	template<typename S, size_t N>
	inline SIMD_Vector<S, N> compress(const SIMD_BitMask<N>& mask, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& src)
	{
		return internals::DefaultDispatcher::run(internals::op_compress{}, mask, a, src);
	}
}