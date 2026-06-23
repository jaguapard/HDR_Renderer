#pragma once
#include "funcs.h"
#include "SIMD_Vector.h"
#include "Dispatcher.h"

namespace AVXXY_NAMESPACE
{
	template<typename S, size_t N, bool Lo>
	static SIMD_Vector<S, N> unpack_base(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		SIMD_Vector<S, N> ret;
		constexpr size_t pairs_per_xmm = 8 / sizeof(S); //8, since unpack only processes lower/upper half of each input
		constexpr size_t elements_per_xmm = 16 / sizeof(S); //how much elements of type S fit into one 128 bit lane
		constexpr size_t xmm_count = sizeof(ret) / 16;
		for (size_t xmm_i = 0; xmm_i < xmm_count; ++xmm_i) //for each 128-bit lane
		{
			for (size_t i = 0; i < elements_per_xmm; i += 2)
			{
				size_t srcI = xmm_i * elements_per_xmm + i / 2 + (Lo ? 0 : elements_per_xmm / 2);
				ret[xmm_i * elements_per_xmm + i] = a[srcI];
				ret[xmm_i * elements_per_xmm + i + 1] = b[srcI];
			}
		}
		return ret;
	}

	//scream your lungs out if scalar fallback is reached and this function is enabled via AVXXY_NOISY_SCALAR define
	static void scream(std::source_location loc = std::source_location::current())
	{
#ifdef AVXXY_NOISY_SCALAR
		std::cout << "\nScalar fallback reached:" << loc.function_name() << "\n";
#endif
	}

	//#define AVXXY_RUN(op) internals::Dispatcher::run<internals::op>
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> add(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;

		scream();
		SIMD_Vector<S, N> ret;
		for (size_t i = 0; i < N; ++i) ret[i] = a[i] + b[i];
		return ret;
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> sub(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;

		scream();
		SIMD_Vector<S, N> ret;
		for (size_t i = 0; i < N; ++i) ret[i] = a[i] - b[i];
		return ret;
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> mul(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;

		scream();
		SIMD_Vector<S, N> ret;
		for (size_t i = 0; i < N; ++i) ret[i] = a[i] * b[i];
		return ret;
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> div(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		
		scream();
		SIMD_Vector<S, N> ret;
		for (size_t i = 0; i < N; ++i) ret[i] = a[i] / b[i];
		return ret;
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> logic_and(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		
		scream();
		SIMD_Vector<S, N> ret;
		using T = typename meta::ScalarTraits<S>::UintT;

		for (size_t i = 0; i < N; ++i) ret[i] = std::bit_cast<S>(T(std::bit_cast<T>(a[i]) & std::bit_cast<T>(b[i])));
		return ret;
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> logic_or(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		
		scream();
		SIMD_Vector<S, N> ret;
		using T = typename meta::ScalarTraits<S>::UintT;
		for (size_t i = 0; i < N; ++i) ret[i] = std::bit_cast<S>(T(std::bit_cast<T>(a[i]) | std::bit_cast<T>(b[i])));
		return ret;
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> logic_xor(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		
		scream();
		SIMD_Vector<S, N> ret;
		using T = typename meta::ScalarTraits<S>::UintT;
		for (size_t i = 0; i < N; ++i) ret[i] = std::bit_cast<S>(T(std::bit_cast<T>(a[i]) ^ std::bit_cast<T>(b[i])));
		return ret;
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> logic_not(const SIMD_Vector<S, N>& a)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		
		scream();
		SIMD_Vector<S, N> ret;
		using T = typename meta::ScalarTraits<S>::UintT;
		for (size_t i = 0; i < N; ++i) ret[i] = std::bit_cast<S>(T(~std::bit_cast<T>(a[i])));
		return ret;
	}
	template<typename S, size_t N, typename I>
	__forceinline SIMD_Vector<S, N> shift_left(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& b)
	{
		scream();
		SIMD_Vector<S, N> ret;
		//using T = typename concepts::same_size_uint_t<S>::type;
		for (size_t i = 0; i < N; ++i) ret[i] = a[i] << b[i];
		return ret;
	}
	template<typename S, size_t N, typename I>
	__forceinline SIMD_Vector<S, N> shift_right(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		
		scream();
		SIMD_Vector<S, N> ret;
		//using T = typename concepts::same_size_uint_t<S>::type;
		for (size_t i = 0; i < N; ++i) ret[i] = a[i] >> b[i];
		return ret;
	}
	template<typename S, size_t N, typename I>
	__forceinline SIMD_Vector<S, N> permx(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& ind)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		
		scream();
		SIMD_Vector<S, N> ret;
		for (size_t i = 0; i < N; ++i) ret[i] = a[ind[i] & (N - 1)];
		return ret;
	}
	template<typename S, size_t N, typename I>
	__forceinline SIMD_Vector<S, N> permx2(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b, const SIMD_Vector<I, N>& ind)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		
		scream();
		SIMD_Vector<S, N> ret;
		for (size_t i = 0; i < N; ++i)
		{
			auto j = ind[i] & (2 * N - 1);
			ret[i] = j < N ? a[j] : b[j - N];
		}
		return ret;
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<float, N> sqrtf(const SIMD_Vector<S, N>& a)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		
		scream();
		SIMD_Vector<float, N> ret;
		for (size_t i = 0; i < N; ++i) ret[i] = std::sqrt(float(a[i]));
		return ret;
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<double, N> sqrtd(const SIMD_Vector<S, N>& a)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		
		scream();
		SIMD_Vector<double, N> ret;
		for (size_t i = 0; i < N; ++i) ret[i] = std::sqrt(double(a[i]));
		return ret;
	}


	template<typename To, size_t N, typename From>
	__forceinline SIMD_Vector<To, N> vcvt(const SIMD_Vector<From, N>& a)
	{
		using namespace meta;
		//if constexpr ((meta::is_fp16<From> && !meta::is_f32<To>) || (!meta::is_f32<From> && meta::is_fp16<To>)) return vcvt<To>(vcvt<float>(value));
		//else return internals::Dispatcher::run<internals::op_cvt<To>>(value);

		scream();
		//using To = typename Op::cvt_to_t;
		SIMD_Vector<To, N> ret;
		for (size_t i = 0; i < N; ++i) ret[i] = a[i];
		return ret;
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
		//if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) return vcast<S>(mask_mov(vcast<U>(ifBitClear), mask, vcast<U>(ifBitSet)));
		//else return internals::Dispatcher::run<internals::op_mask_mov>(ifBitClear, mask, ifBitSet);

		scream();
		SIMD_Vector<S, N> ret;
		for (size_t i = 0; i < N; ++i) ret[i] = mask[i] ? ifBitSet[i] : ifBitClear[i];
		return ret;
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
		//if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) return vcast<S>(load<S, N>(p, mask, vcast<U>(src)));
		//else return internals::Dispatcher::run<internals::op_load<S, N>>(p, mask, src);

		SIMD_Vector<S, N> ret;
		const S* sp = static_cast<const S*>(p);
		for (size_t i = 0; i < N; ++i) ret[i] = mask[i] ? sp[i] : src[i];
		return ret;
	}
	template<typename S, size_t N>
	__forceinline void store(const SIMD_Vector<S, N>& v, void* p, const mask_t<S, N>& mask)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		//if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) store(vcast<U>(v), p, mask);
		
		scream();
		S* sp = static_cast<S*>(p);
		for (size_t i = 0; i < N; ++i) if (mask[i]) sp[i] = v[i];
	}

	template<typename S, size_t N, size_t Scale, typename I>
	__forceinline void scatter(const SIMD_Vector<S, N>& v, void* base, const SIMD_Vector<I, N>& ind, const mask_t<S, N>& mask)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;

		//if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) scatter<S, N, Scale, I>(vcast<U>(vec), base, ind, mask);
		//else internals::Dispatcher::run<internals::op_scatter<Scale>>(vec, base, ind, mask);
		scream();
		size_t addr = size_t(base);
		for (size_t i = 0; i < N; ++i) if (mask[i]) *(S*)(addr + Scale * ind[i]) = v[i];
	}
	template<typename S, size_t N>
	__forceinline mask_t<S, N> cmp_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		//return internals::Dispatcher::run<internals::op_cmpeq>(a, b);

		scream();
		typename SIMD_Vector<S, N>::MaskT ret = 0;
		for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] == b[i]);
		return ret;
	}
	template<typename S, size_t N>
	__forceinline mask_t<S, N> cmp_not_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		//return internals::Dispatcher::run<internals::op_cmpneq>(a, b);
		scream();
		typename SIMD_Vector<S, N>::MaskT ret = 0;
		for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] != b[i]);
		return ret;
	}
	template<typename S, size_t N>
	__forceinline mask_t<S, N> cmp_less(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;

		scream();
		typename SIMD_Vector<S, N>::MaskT ret = 0;
		for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] < b[i]);
		return ret;
	}
	template<typename S, size_t N>
	__forceinline mask_t<S, N> cmp_less_or_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;

		scream();
		typename SIMD_Vector<S, N>::MaskT ret = 0;
		for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] <= b[i]);
		return ret;
	}
	template<typename S, size_t N>
	__forceinline mask_t<S, N> cmp_greater(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		
		scream();
		typename SIMD_Vector<S, N>::MaskT ret = 0;
		for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] > b[i]);
		return ret;
	}
	template<typename S, size_t N>
	__forceinline mask_t<S, N> cmp_greater_or_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		
		typename SIMD_Vector<S, N>::MaskT ret = 0;
		for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] >= b[i]);
		return ret;
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> abs(const SIMD_Vector<S, N>& a)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		if constexpr (std::is_unsigned_v<S>) return a;
		else
		{
			scream();
			SIMD_Vector<S, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = std::abs(a[i]);
			return ret;
		}
	}

	template<typename S, size_t N>
		requires (meta::any_float<S>)
	__forceinline SIMD_Vector<S, N> floor(const SIMD_Vector<S, N>& a)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		
		scream();
		SIMD_Vector<S, N> ret;
		for (size_t i = 0; i < N; ++i) ret[i] = std::floor(a[i]);
		return ret;
	}
	template<typename S, size_t N>
		requires (meta::any_float<S>)
	__forceinline SIMD_Vector<S, N> ceil(const SIMD_Vector<S, N>& a)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		
		scream();
		SIMD_Vector<S, N> ret;
		for (size_t i = 0; i < N; ++i) ret[i] = std::ceil(a[i]);
		return ret;
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> min(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		
		scream();
		SIMD_Vector<S, N> ret;
		for (size_t i = 0; i < N; ++i) ret[i] = std::min(a[i], b[i]);
		return ret;
	}

	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> max(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		
		scream();
		SIMD_Vector<S, N> ret;
		for (size_t i = 0; i < N; ++i) ret[i] = std::max(a[i], b[i]);
		return ret;
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
		//if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) return vcast<S>(unpacklo(vcast<U>(a), vcast<U>(b)));
		//else return internals::Dispatcher::run<internals::op_unpacklo>(a, b);

		scream();
		return unpack_base<S, N, true>(a, b);
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> unpackhi(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		//if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) return vcast<S>(unpackhi(vcast<U>(a), vcast<U>(b)));
		//else return internals::Dispatcher::run<internals::op_unpackhi>(a, b);
		scream();
		return unpack_base<S, N, false>(a, b);
	}
	template<typename S, size_t N>
	__forceinline SIMD_Vector<S, N> compress(const mask_t<S, N>& mask, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& src)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		//if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) return vcast<S>(compress(mask, vcast<U>(a), vcast<U>(src)));
		
		scream();
		SIMD_Vector<S, N> ret;
		size_t j = 0;
		for (size_t i = 0; i < N; ++i) if (mask[i]) ret[j++] = a[i];
		for (; j < N; ++j) ret[j] = src[j];
		return ret;
	}
	template<typename S, size_t N>
		requires (sizeof(S) * 8 >= N)
	__forceinline SIMD_Vector<typename meta::ScalarTraits<S>::UintT, N> conflict(const SIMD_Vector<S, N>& a)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		//TODO: allow bigger CD?
		
		using UV = SIMD_Vector<U, N>;
		UV ret;
		for (size_t i = 0; i < N; ++i)
		{
			U acc = 0;
			for (size_t j = 0; j < i; ++j)
			{
				if (a[i] == a[j]) acc |= U(1) << j;
			}
			ret[i] = acc;
		}
		return ret;
	}

	template<typename S, size_t N>
	__forceinline mask_t<S, N> movemask(const SIMD_Vector<S, N>& v)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		//if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) return movemask(vcast<U>(v));
		
		mask_t<S, N> ret;
		using Tr = meta::ScalarTraits<S>;
		using U = Tr::UintT;
		for (size_t i = 0; i < N; ++i)
		{
			U sb = std::bit_cast<U>(v[i]) & Tr::SignMask;
			ret.setBit(i, sb);
		}
		return ret;
	}

	template<typename S, meta::ScalarSizeClassEnum C, size_t N>
	__forceinline SIMD_Vector<S, N> movm(const SIMD_Mask<C, N>& mask)
	{
		using namespace meta;
		using U = typename ScalarTraits<S>::UintT;
		//if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) return vcast<S>(movm<U>(mask));
		//return internals::Dispatcher::run<internals::op_movm<S>>(mask);

		scream();
		SIMD_Vector<S, N> ret;
		using Tr = meta::ScalarTraits<S>;
		for (size_t i = 0; i < N; ++i)
		{
			typename Tr::UintT u = mask[i] ? Tr::AllOnesUint : 0;
			ret[i] = std::bit_cast<S>(u);
		}
		return ret;
	}

	template<typename S, size_t N, size_t Scale, typename I>
	__forceinline SIMD_Vector<S, N> __gather_impl(const void* base, const SIMD_Vector<I, N>& ind, const typename SIMD_Vector<S, N>::MaskT& mask, const SIMD_Vector<S, N>& src)
	{
#if 0
		using namespace meta;
		using namespace internals;
		using U = typename ScalarTraits<S>::UintT;
		if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) return vcast<S>(__gather_impl<U, N, Scale, I>(base, ind, mask, vcast<U>(src)));
		else
		{
			//put everything up here to prevent else if chain breaks (since compilation gives useless errors by thinking unsanitized inputs surviving to native gathers
			using namespace meta;
			using CanonicalIndex_t = std::conditional_t<(sizeof(I) <= 4), int32_t, int64_t>;
			using RetVec_t = SIMD_Vector<S, N>;
			using IndVec_t = SIMD_Vector<I, N>;
			constexpr size_t MaxSize = std::max(sizeof(RetVec_t), sizeof(IndVec_t));

			//if scale is not native, emulate it by gathering with scale 1 and manually calculated byte offsets. 
			//TODO: Can optimize a little by checking if Scale*maxint(I) fits into smaller sizes
			if constexpr (Scale != 1 && Scale != 2 && Scale != 4 && Scale != 8) return gather<S, N, 1>(base, vcvt<int64_t>(ind) * Scale, mask, src);

			//TODO: emulation of small int gathers (where elements gathered are small ints)
			else if constexpr (!std::is_same_v<I, CanonicalIndex_t>) return gather<S, N, Scale>(base, vcvt<CanonicalIndex_t>(ind), mask, src);

			//SANITIZATION DONE
			//if we get here, means that indices are already in good format (4-byte or 8-byte)
			//and scale is good too.
			//break up large gather into halves
			else if constexpr (MaxSize > 64) return RetVec_t{
				gather<S, N / 2, Scale, I>(base, ind.lo(), mask.lo(), src.lo()),
				gather<S, N / 2, Scale, I>(base, ind.hi(), mask.hi(), src.hi()) };

			else
			{
				if constexpr (FS.has(AVX512_F))
				{
					if constexpr (is_zmm_size(MaxSize))
					{
						//clang is a cry-baby with ind here for some reason, so force convert it. Pay attention to size!
						std::conditional_t<(meta::zmm_sized<IndVec_t>), __m512i, __m256i> ni = ind;
						if constexpr (is_i64<I> && is_f64<S>) return _mm512_mask_i64gather_pd(src, mask, ni, base, Scale);
						else if constexpr (is_i64<I> && is_f32<S>) return _mm512_mask_i64gather_ps(src, mask, ni, base, Scale);
						else if constexpr (is_i64<I> && any_i64<S>) return _mm512_mask_i64gather_epi64(src, mask, ni, base, Scale);
						else if constexpr (is_i64<I> && any_i32<S>) return _mm512_mask_i64gather_epi32(src, mask, ni, base, Scale);

						else if constexpr (is_i32<I> && is_f64<S>) return _mm512_mask_i32gather_pd(src, mask, ni, base, Scale);
						else if constexpr (is_i32<I> && is_f32<S>) return _mm512_mask_i32gather_ps(src, mask, ni, base, Scale);
						else if constexpr (is_i32<I> && any_i64<S>) return _mm512_mask_i32gather_epi64(src, mask, ni, base, Scale);
						else if constexpr (is_i32<I> && any_i32<S>) return _mm512_mask_i32gather_epi32(src, mask, ni, base, Scale);
					}
					if constexpr (FS.has(AVX512_VL))
					{
						if constexpr (is_ymm_size(MaxSize))
						{
							std::conditional_t<(meta::ymm_sized<IndVec_t>), __m256i, __m128i> ni = ind;
							if constexpr (is_i64<I> && is_f64<S>) return _mm256_mmask_i64gather_pd(src, mask, ni, base, Scale);
							else if constexpr (is_i64<I> && is_f32<S>) return _mm256_mmask_i64gather_ps(src, mask, ni, base, Scale);
							else if constexpr (is_i64<I> && any_i64<S>) return _mm256_mmask_i64gather_epi64(src, mask, ni, base, Scale);
							else if constexpr (is_i64<I> && any_i32<S>) return _mm256_mmask_i64gather_epi32(src, mask, ni, base, Scale);

							else if constexpr (is_i32<I> && is_f64<S>) return _mm256_mmask_i32gather_pd(src, mask, ni, base, Scale);
							else if constexpr (is_i32<I> && is_f32<S>) return _mm256_mmask_i32gather_ps(src, mask, ni, base, Scale);
							else if constexpr (is_i32<I> && any_i64<S>) return _mm256_mmask_i32gather_epi64(src, mask, ni, base, Scale);
							else if constexpr (is_i32<I> && any_i32<S>) return _mm256_mmask_i32gather_epi32(src, mask, ni, base, Scale);
						}
						else if constexpr (is_xmm_size(MaxSize))
						{
							__m128i ni = ind;
							if constexpr (is_i64<I> && is_f64<S>) return _mm_mmask_i64gather_pd(src, mask, ni, base, Scale);
							else if constexpr (is_i64<I> && is_f32<S>) return _mm_mmask_i64gather_ps(src, mask, ni, base, Scale);
							else if constexpr (is_i64<I> && any_i64<S>) return _mm_mmask_i64gather_epi64(src, mask, ni, base, Scale);
							else if constexpr (is_i64<I> && any_i32<S>) return _mm_mmask_i64gather_epi32(src, mask, ni, base, Scale);

							else if constexpr (is_i32<I> && is_f64<S>) return _mm_mmask_i32gather_pd(src, mask, ni, base, Scale);
							else if constexpr (is_i32<I> && is_f32<S>) return _mm_mmask_i32gather_ps(src, mask, ni, base, Scale);
							else if constexpr (is_i32<I> && any_i64<S>) return _mm_mmask_i32gather_epi64(src, mask, ni, base, Scale);
							else if constexpr (is_i32<I> && any_i32<S>) return _mm_mmask_i32gather_epi32(src, mask, ni, base, Scale);
						}
					}
				}
				if constexpr (FS.has(AVX2))
				{
					if constexpr (MaxSize > 32)
					{
						return {
							gather<S, N / 2, Scale, I>(base, ind.lo(), mask.lo(), src.lo()),
							gather<S, N / 2, Scale, I>(base, ind.hi(), mask.hi(), src.hi())
						};
					}
					else if constexpr (is_ymm_size(MaxSize))
					{
						//clang is a cry-baby with ind here for some reason, so force convert it. Pay attention to size!
						std::conditional_t<(meta::ymm_sized<IndVec_t>), __m256i, __m128i> ni = ind;
						if constexpr (is_i64<I> && is_f64<S>) return _mm256_mask_i64gather_pd(src, base, ni, mask, Scale);
						else if constexpr (is_i64<I> && is_f32<S>) return _mm256_mask_i64gather_ps(src, base, ni, mask, Scale);
						else if constexpr (is_i64<I> && any_i64<S>) return _mm256_mask_i64gather_epi64(src, base, ni, mask, Scale);
						else if constexpr (is_i64<I> && any_i32<S>) return _mm256_mask_i64gather_epi32(src, base, ni, mask, Scale);

						else if constexpr (is_i32<I> && is_f64<S>) return _mm256_mask_i32gather_pd(src, base, ni, mask, Scale);
						else if constexpr (is_i32<I> && is_f32<S>) return _mm256_mask_i32gather_ps(src, base, ni, mask, Scale);
						else if constexpr (is_i32<I> && any_i64<S>) return _mm256_mask_i32gather_epi64(src, base, ni, mask, Scale);
						else if constexpr (is_i32<I> && any_i32<S>) return _mm256_mask_i32gather_epi32(src, base, ni, mask, Scale);
					}
					else if constexpr (is_xmm_size(MaxSize))
					{
						__m128i ni = ind;
						if constexpr (is_i64<I> && is_f64<S>) return _mm_mask_i64gather_pd(src, base, ni, mask, Scale);
						else if constexpr (is_i64<I> && is_f32<S>) return _mm_mask_i64gather_ps(src, base, ni, mask, Scale);
						else if constexpr (is_i64<I> && any_i64<S>) return _mm_mask_i64gather_epi64(src, base, ni, mask, Scale);
						else if constexpr (is_i64<I> && any_i32<S>) return _mm_mask_i64gather_epi32(src, base, ni, mask, Scale);

						else if constexpr (is_i32<I> && is_f64<S>) return _mm_mask_i32gather_pd(src, base, ni, mask, Scale);
						else if constexpr (is_i32<I> && is_f32<S>) return _mm_mask_i32gather_ps(src, base, ni, mask, Scale);
						else if constexpr (is_i32<I> && any_i64<S>) return _mm_mask_i32gather_epi64(src, base, ni, mask, Scale);
						else if constexpr (is_i32<I> && any_i32<S>) return _mm_mask_i32gather_epi32(src, base, ni, mask, Scale);
					}
				}

				//scream();
#endif
				SIMD_Vector<S, N> ret;
				size_t addr = size_t(base);
				for (size_t i = 0; i < N; ++i) ret[i] = mask[i] ? *(const S*)(addr + Scale * ind[i]) : src[i];
				return ret;
	}
}