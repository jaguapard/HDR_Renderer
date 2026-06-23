#pragma once
#include "funcs.h"
#include "SIMD_Vector.h"
#include "Dispatcher.h"

namespace AVXXY_NAMESPACE
{
//#define AVXXY_RUN(op) internals::Dispatcher::run<internals::op>
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> add(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		return internals::Dispatcher::run<internals::op_add>(a, b);
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> sub(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		return internals::Dispatcher::run<internals::op_sub>(a, b);
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> mul(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		return internals::Dispatcher::run<internals::op_mul>(a, b);
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> div(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		return internals::Dispatcher::run<internals::op_div>(a, b);
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> logic_and(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) return vcast<S>(logic_and(vcast<U>(a), vcast<U>(b)));
		else return internals::Dispatcher::run<internals::op_and>(a, b);
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> logic_or(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) return vcast<S>(logic_or(vcast<U>(a), vcast<U>(b)));
		else return internals::Dispatcher::run<internals::op_or>(a, b);
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> logic_xor(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) return vcast<S>(logic_xor(vcast<U>(a), vcast<U>(b)));
		else return internals::Dispatcher::run<internals::op_xor>(a, b);
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> logic_not(const SIMD_Vector<S, N>& a)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) return vcast<S>(logic_not(vcast<U>(a)));
		else return internals::Dispatcher::run<internals::op_not>(a);
	}
	template<typename S, size_t N, typename I>
	__forceinline SIMD_Vector<S, N> shift_left(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& amount)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		return internals::Dispatcher::run<internals::op_shl>(a, amount);
	}
	template<typename S, size_t N, typename I>
	__forceinline SIMD_Vector<S, N> shift_right(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& amount)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		return internals::Dispatcher::run<internals::op_shr>(a, amount);
	}
	template<typename S, size_t N, typename I>
	__forceinline SIMD_Vector<S, N> permx(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& ind)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) return vcast<S>(permx(vcast<U>(a), ind));
		else return internals::Dispatcher::run<internals::op_permx>(a, ind);
	}
	template<typename S, size_t N, typename I>
	__forceinline SIMD_Vector<S, N> permx2(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b, const SIMD_Vector<I, N>& ind)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) return vcast<S>(permx2(vcast<U>(a), vcast<U>(b), ind));
		else return internals::Dispatcher::run<internals::op_permx2>(a, b, ind);
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<float, N> sqrtf(const SIMD_Vector<S, N>& a)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		return internals::Dispatcher::run<internals::op_sqrtf>(a);
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<double, N> sqrtd(const SIMD_Vector<S, N>& a)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		return internals::Dispatcher::run<internals::op_sqrtd>(a);
	}


	template<typename To, size_t N, typename From>
	__forceinline SIMD_Vector<To, N> vcvt(const SIMD_Vector<From, N>& value)
	{
		using namespace meta;
		if constexpr ((meta::is_fp16<From> && !meta::is_f32<To>) || (!meta::is_f32<From> && meta::is_fp16<To>)) return vcvt<To>(vcvt<float>(value));
		else return internals::Dispatcher::run<internals::op_cvt<To>>(value);
	}

	template<typename S2, typename S, size_t N> requires (meta::IsScalarType<S2> && (sizeof(SIMD_Vector<S, N>) % sizeof(S2) == 0))
	__forceinline SIMD_Vector<S2, sizeof(SIMD_Vector<S, N>) / sizeof(S2)> vcast(const SIMD_Vector<S, N>& a)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		return vreinterpret_us<SIMD_Vector<S2, sizeof(SIMD_Vector<S, N>) / sizeof(S2)>>(a);
	}
	template<typename T, typename S, size_t N>
		requires (meta::IsSimdVector<T> && (sizeof(SIMD_Vector<S, N>) % sizeof(typename T::ScalarT) == 0) && sizeof(SIMD_Vector<S, N>) == sizeof(T))
	__forceinline T vcast(const SIMD_Vector<S, N>& a)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		return vreinterpret_us<T>(a);
	}


	template<typename T, typename S, size_t N> requires (sizeof(T) == sizeof(SIMD_Vector<S, N>))
	__forceinline T vreinterpret(const SIMD_Vector<S, N>& value)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		return vreinterpret_us<T>(value);
	}
	template<typename T, typename S, size_t N>
	__forceinline T vreinterpret_us(const SIMD_Vector<S, N>& value)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		T ret;
		memcpy(&ret, &value, std::min(sizeof(ret), sizeof(value)));
		return ret;
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> mask_mov(const SIMD_Vector<S, N>& ifBitClear, const mask_t<S, N>& mask, const SIMD_Vector<S, N>& ifBitSet)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) return vcast<S>(mask_mov(vcast<U>(ifBitClear), mask, vcast<U>(ifBitSet)));
		else return internals::Dispatcher::run<internals::op_mask_mov>(ifBitClear, mask, ifBitSet);
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> maskz_mov(const mask_t<S, N>& mask, const SIMD_Vector<S, N>& ifBitSet)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		return mask_mov(SIMD_Vector<S, N>(0), mask, ifBitSet);
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> blend(const mask_t<S, N>& mask, const SIMD_Vector<S, N>& ifBitClear, const SIMD_Vector<S, N>& ifBitSet)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		return mask_mov(ifBitClear, mask, ifBitSet);
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> load(const void* p, const mask_t<S, N>& mask, const SIMD_Vector<S, N>& src)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) return vcast<S>(load<S, N>(p, mask, vcast<U>(src)));
		else return internals::Dispatcher::run<internals::op_load<S, N>>(p, mask, src);
	}
	template<typename S, size_t N>
	__forceinline void store(const SIMD_Vector<S, N>& v, void* p, const mask_t<S, N>& mask)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) store(vcast<U>(v), p, mask);
		else internals::Dispatcher::run<internals::op_store>(v, p, mask);
	}

	template<typename S, size_t N, size_t Scale, typename I>
	__forceinline void scatter(const SIMD_Vector<S, N>& vec, void* base, const SIMD_Vector<I, N>& ind, const mask_t<S, N>& mask)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) scatter<S, N, Scale, I>(vcast<U>(vec), base, ind, mask);
		else internals::Dispatcher::run<internals::op_scatter<Scale>>(vec, base, ind, mask);
	}
	template<typename S, size_t N>
	__forceinline mask_t<S, N> cmp_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		return internals::Dispatcher::run<internals::op_cmpeq>(a, b);
	}
	template<typename S, size_t N>
	__forceinline mask_t<S, N> cmp_not_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		return internals::Dispatcher::run<internals::op_cmpneq>(a, b);
	}
	template<typename S, size_t N>
	__forceinline mask_t<S, N> cmp_less(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		return internals::Dispatcher::run<internals::op_cmplt>(a, b);
	}
	template<typename S, size_t N>
	__forceinline mask_t<S, N> cmp_less_or_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		return internals::Dispatcher::run<internals::op_cmple>(a, b);
	}
	template<typename S, size_t N>
	__forceinline mask_t<S, N> cmp_greater(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		return internals::Dispatcher::run<internals::op_cmpgt>(a, b);
	}
	template<typename S, size_t N>
	__forceinline mask_t<S, N> cmp_greater_or_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		return internals::Dispatcher::run<internals::op_cmpge>(a, b);
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> abs(const SIMD_Vector<S, N>& a)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		return internals::Dispatcher::run<internals::op_abs>(a);
	}

	template<typename S, size_t N>
		requires (meta::any_float<S>)
	__forceinline SIMD_Vector<S, N> floor(const SIMD_Vector<S, N>& a)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		return internals::Dispatcher::run<internals::op_floor>(a);
	}
	template<typename S, size_t N>
		requires (meta::any_float<S>)
	__forceinline SIMD_Vector<S, N> ceil(const SIMD_Vector<S, N>& a)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		return internals::Dispatcher::run<internals::op_ceil>(a);
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> min(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		return internals::Dispatcher::run<internals::op_min>(a, b);
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> max(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		return internals::Dispatcher::run<internals::op_max>(a, b);
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> clamp(const SIMD_Vector<S, N>& val, const SIMD_Vector<S, N>& min, const SIMD_Vector<S, N>& max)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		return AVXXY_NAMESPACE::max(min, AVXXY_NAMESPACE::min(val, max));
	}


	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> unpacklo(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) return vcast<S>(unpacklo(vcast<U>(a), vcast<U>(b)));
		else return internals::Dispatcher::run<internals::op_unpacklo>(a, b);
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> unpackhi(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) return vcast<S>(unpackhi(vcast<U>(a), vcast<U>(b)));
		else return internals::Dispatcher::run<internals::op_unpackhi>(a, b);
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> compress(const mask_t<S, N>& mask, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& src)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) return vcast<S>(compress(mask, vcast<U>(a), vcast<U>(src)));
		else return internals::Dispatcher::run<internals::op_compress>(mask, a, src);
	}
	template<typename S, size_t N>
		requires (sizeof(S) * 8 >= N)
	__forceinline SIMD_Vector<typename meta::ScalarTraits<S>::UintT, N> conflict(const SIMD_Vector<S, N>& a)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		//TODO: allow bigger CD?
		return internals::Dispatcher::run<internals::op_conflict>(a);
	}

	template<typename S, size_t N>
	__forceinline mask_t<S, N> movemask(const SIMD_Vector<S, N>& v)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) return movemask(vcast<U>(v));
		else return internals::Dispatcher::run<internals::op_movemask>(v);
	}

	template<typename S, meta::ScalarSizeClassEnum C, size_t N>
	__forceinline SIMD_Vector<S, N> movm(const SIMD_Mask<C, N>& mask)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) return vcast<S>(movm<U>(mask));
		return internals::Dispatcher::run<internals::op_movm<S>>(mask);
	}

	template<typename S, size_t N, size_t Scale, typename I>
	__forceinline SIMD_Vector<S, N> __gather_impl(const void* base, const SIMD_Vector<I, N>& ind, const typename SIMD_Vector<S, N>::MaskT& mask, const SIMD_Vector<S, N>& src)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) return vcast<S>(__gather_impl<U, N, Scale, I>(base, ind, mask, vcast<U>(src)));
		else return internals::Dispatcher::run<internals::op_gather<S, N, Scale>>(base, ind, mask, src);
	}
}