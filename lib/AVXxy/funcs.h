#pragma once
#include "namespace.h"
#include "SIMD_Mask.h"
#include "SIMD_Vector.h"
#include <immintrin.h>

namespace AVXXY_NAMESPACE
{
	template<typename S, size_t N> SIMD_Vector<S, N> add(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);
	//template<typename S, size_t N> SIMD_Vector<S, N> add(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b, const typename SIMD_Vector<S,N>::MaskType& mask);
	//template<typename S, size_t N> SIMD_Vector<S, N> add(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b, const typename SIMD_Vector<S,N>::MaskType& mask, const SIMD_Vector<S,N>& src);
	template<typename S, size_t N> SIMD_Vector<S, N> sub(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);
	//template<typename S, size_t N> SIMD_Vector<S, N> sub(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b, const typename SIMD_Vector<S,N>::MaskType& mask);
	//template<typename S, size_t N> SIMD_Vector<S, N> sub(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b, const typename SIMD_Vector<S,N>::MaskType& mask, const SIMD_Vector<S,N>& src);
	template<typename S, size_t N> SIMD_Vector<S, N> mul(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);
	//template<typename S, size_t N> SIMD_Vector<S, N> mul(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b, const typename SIMD_Vector<S,N>::MaskType& mask);
	//template<typename S, size_t N> SIMD_Vector<S, N> mul(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b, const typename SIMD_Vector<S,N>::MaskType& mask, const SIMD_Vector<S,N>& src);

	//Performs divison of two vectors and returns the result
	//For integer types, the division is emulated by floating point divison of size large enough to guarantee the same result
	//For integer types smaller that 32-bits wide, an extra conversion is made to 32-bit integers before the division and back to input type after the division
	//Thus, the small integer division has much lower performance than that for other types
	template<typename S, size_t N> SIMD_Vector<S, N> div(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);

	template<typename S, size_t N> SIMD_Vector<S, N> logic_and(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);
	template<typename S, size_t N> SIMD_Vector<S, N> logic_or(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);
	template<typename S, size_t N> SIMD_Vector<S, N> logic_xor(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);
	template<typename S, size_t N> SIMD_Vector<S, N> logic_not(const SIMD_Vector<S, N>& a);
	template<typename S, size_t N, typename I> SIMD_Vector<S, N> shift_left(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& amount);
	template<typename S, size_t N, typename I> SIMD_Vector<S, N> shift_right(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& amount);

	template<typename S, size_t N, typename I> SIMD_Vector<S, N> permx(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& ind);
	template<typename S, size_t N, typename I> SIMD_Vector<S, N> permx2(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& ind, const SIMD_Vector<S, N>& b);

	template<typename S, size_t N> SIMD_Vector<S, N / 2> upper_half(const SIMD_Vector<S, N>& a);
	template<typename S, size_t N> SIMD_Vector<S, N / 2> lower_half(const SIMD_Vector<S, N>& a);

	template<typename S, size_t N> SIMD_Vector<float, N> sqrtf(const SIMD_Vector<S, N>& a);
	template<typename S, size_t N> SIMD_Vector<double, N> sqrtd(const SIMD_Vector<S, N>& a);

	//Converts the vector of one type to vector of another and returns the result
	//For floating point to integer conversions, the input vector is truncated
	//For integer to bigger integer conversions, the input vector is sign or zero extended, depending on input signedness
	//For integer to smaller integer conversions, the input vector is wrapped around small integer's max value (TODO: is it true?)
	template<typename To, size_t N, typename From> SIMD_Vector<To, N> cvt(const SIMD_Vector<From, N>& value);
	template<typename S, size_t N> SIMD_Vector<S, N*2> concat(const SIMD_Vector<S, N>& to, const SIMD_Vector<S, N>& what);
	template<size_t N> SIMD_Mask<N * 2> concat_masks(const SIMD_Mask<N>& to, const SIMD_Mask<N>& what);
	
	//Reinterprets value as vector of other type and returns the result.
	//If returned vector's size is smaller than input, input's upper bits are discarded
	//If returned vector's size is bigger than input, upper bits of returned value are undefined.
	template<typename T, typename S, size_t N> T reinterpret(const SIMD_Vector<S, N>& value);

	//Extracts Part'th part of size (vector size)/PartCount from input vector and returns the result.
	//For example, to extract third quarter of a vector, call extract<2,4> (2, because indices are starting from 0)
	//To extract upper half, call extract<1,2>, to extract 6th part out of 8 call extract<5,8>, etc.
	template<size_t Part, size_t PartCount, typename S, size_t N> SIMD_Vector<S, N / PartCount> extract(const SIMD_Vector<S, N>& value);

	//Inserts the vector `what` into vector `to` at Part'th division of sizeof(what). 
	//I.e. treats target vector as multiple contigious vectors of the same size as `what`, and replaces Part'th one with `what`.
	//For example, to replace upper half of target vector, call insert<1>(to, what), where `what` is half of `to`'s size.
	//To override 5'th 16'th part, call insert<4>(to, what), where what is 1/16th the size of to.
	//Sizes are checked and deduced automatically on compile time and raise static_assert errors on fail
	template<size_t Part, size_t N2, typename S, size_t N> SIMD_Vector<S, N> insert(const SIMD_Vector<S, N>& to, const SIMD_Vector<S, N2>& what);

	template <typename S, size_t N> SIMD_Vector<S, N> mask_mov(const SIMD_Vector<S, N>& ifBitClear, const typename SIMD_Vector<S, N>::MaskType& mask, const SIMD_Vector<S, N>& ifBitSet);
	template <typename S, size_t N> SIMD_Vector<S, N> maskz_mov(const typename SIMD_Vector<S, N>::MaskType& mask, const SIMD_Vector<S, N>& ifBitSet);
	template <typename S, size_t N> SIMD_Vector<S, N> blend(const typename SIMD_Vector<S, N>::MaskType& mask, const SIMD_Vector<S, N>& ifBitClear, const SIMD_Vector<S, N>& ifBitSet);

	template<typename S, size_t N> SIMD_Vector<S, N> load(const void* p, const typename SIMD_Vector<S, N>::MaskType& mask = SIMD_Vector<S, N>::MaskType::AllOnes, const SIMD_Vector<S, N>& src = 0);
	template<typename S, size_t N> void store(SIMD_Vector<S, N>& v, const void* p, const typename SIMD_Vector<S, N>::MaskType& mask = SIMD_Vector<S, N>::MaskType::AllOnes);
	template<typename S, size_t N, size_t Scale = sizeof(S), typename I> requires (std::is_integral_v<I> && sizeof(I) <= 8)
	SIMD_Vector<S, N> gather(const void* base, const SIMD_Vector<I, N>& ind, const typename SIMD_Vector<S, N>::MaskType& mask = SIMD_Vector<S, N>::MaskType::AllOnes, const SIMD_Vector<S, N>& src = 0);
	template<typename S, size_t N, size_t Scale, typename I> void scatter(const SIMD_Vector<S, N>& vec, void* base, const SIMD_Vector<I, N>& ind, const typename SIMD_Vector<S, N>::MaskType& mask = SIMD_Vector<S, N>::MaskType::AllOnes);

	template<typename S, size_t N> typename SIMD_Vector<S, N>::MaskType cmp_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);
	template<typename S, size_t N> typename SIMD_Vector<S, N>::MaskType cmp_not_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);
	template<typename S, size_t N> typename SIMD_Vector<S, N>::MaskType cmp_less(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);
	template<typename S, size_t N> typename SIMD_Vector<S, N>::MaskType cmp_less_or_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);
	template<typename S, size_t N> typename SIMD_Vector<S, N>::MaskType cmp_greater(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);
	template<typename S, size_t N> typename SIMD_Vector<S, N>::MaskType cmp_greater_or_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);

	//Converts mask to mask register and returns the result. Each vector's element is set to bitwise all ones for elements correcsponding to set mask bits, or bitwize zero otherwise.
	template<typename S, size_t N> SIMD_Vector<S, N> mask2vec(const SIMD_Mask<N>& mask);

	template<typename S, size_t N> SIMD_Vector<S, N> abs(const SIMD_Vector<S, N>& a);
	template<typename S, size_t N> SIMD_Vector<S, N> min(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);
	template<typename S, size_t N> SIMD_Vector<S, N> max(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);
	template<typename S, size_t N> SIMD_Vector<S, N> clamp(const SIMD_Vector<S, N>& val, const SIMD_Vector<S, N>& min, const SIMD_Vector<S, N>& max);
}