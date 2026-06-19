#pragma once
#include "SIMD_Vector.h"
#include "SIMD_BitMask.h"
namespace AVXXY_NAMESPACE
{
	//Performs element-wise addition of vectors and returns the result
	template<typename S, size_t N> SIMD_Vector<S, N> add(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);
	//Performs element-wise subtraction of elements of vector b from elements of vector a and returns the result
	template<typename S, size_t N> SIMD_Vector<S, N> sub(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);
	//Multiplies elements of input vectors together and returns the result. For integer vectors, only lower half of the 2x-sized intermediate result is returned
	template<typename S, size_t N> SIMD_Vector<S, N> mul(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);

	//Performs divison of two vectors and returns the result
	//For integer types, the division is emulated by floating point divison of size large enough to guarantee the same result
	//For integer types smaller that 32-bits wide, an extra conversion is made to 32-bit integers before the division and back to input type after the division
	//Thus, the small integer division has much lower performance than that for other types
	//64-bit integer division is scalar
	template<typename S, size_t N> SIMD_Vector<S, N> div(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);

	//Returns bitwise logical and of the two vectors. Floating point vectors are also legibile for this operation.
	template<typename S, size_t N> SIMD_Vector<S, N> logic_and(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);
	//Returns bitwise logical or of the two vectors. Floating point vectors are also legibile for this operation.
	template<typename S, size_t N> SIMD_Vector<S, N> logic_or(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);
	//Returns bitwise logical exclusive or of the two vectors. Floating point vectors are also legibile for this operation.
	template<typename S, size_t N> SIMD_Vector<S, N> logic_xor(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);
	//Returns bitwise logical negation of the input vector. Floating point vectors are also legibile for this operation.
	template<typename S, size_t N> SIMD_Vector<S, N> logic_not(const SIMD_Vector<S, N>& a);
	//Shift packed integers in `a` left by the amount specified by the corresponding element of `amount` while shifting in zeros, and returns the result
	template<typename S, size_t N, typename I> SIMD_Vector<S, N> shift_left(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& amount);
	//Shift packed integers in `a` right by the amount specified by the corresponding element of `amount` while shifting in zeros, and returns the result
	template<typename S, size_t N, typename I> SIMD_Vector<S, N> shift_right(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& amount);

	//Performs permutation on the elements from vector `a`. Elements of the returned vector are gathered from vector `a` by indices passed in `ind`.
	//Indices outside the range [0, N-1] wrap around N (-1 maps to N-1, N maps to 0).
	//ret[i] = a[ind[i] & (N-1)]
	template<typename S, size_t N, typename I> SIMD_Vector<S, N> permx(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& ind);
	//Appends vector `b` to vector `a`, then performs permutation on the elements from this temporary value. 
	//Elements of the returned vector are gathered from temporary vector by indices passed in `ind` 
	//Indices outside the range [0, 2*N-1] wrap around 2*N (-1 maps to 2*N-1, 2*N maps to 0).
	//t = ind[i] & (2*N - 1)
	//ret[i] = t < N ? a[t] : b[t-N]
	template<typename S, size_t N, typename I> SIMD_Vector<S, N> permx2(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b, const SIMD_Vector<I, N>& ind);

	//Converts the input to single-precision floating point numbers, then returns the square root of this value
	template<typename S, size_t N> SIMD_Vector<float, N> sqrtf(const SIMD_Vector<S, N>& a);
	//Converts the input to double-precision floating point numbers, then returns the square root of this value
	template<typename S, size_t N> SIMD_Vector<double, N> sqrtd(const SIMD_Vector<S, N>& a);

	//Converts the vector of one type to vector of another and returns the result
	//For floating point to integer conversions, the input vector is truncated
	//For integer to bigger integer conversions, the input vector is sign or zero extended, depending on input signedness
	//For integer to smaller integer conversions, the input vector is wrapped around small integer's max value (TODO: is it true?)
	template<typename To, size_t N, typename From> SIMD_Vector<To, N> vcvt(const SIMD_Vector<From, N>& value);
	//Appends vector `what` to vector `to` and returns the result
	template<typename S, size_t N> SIMD_Vector<S, N * 2> concat(const SIMD_Vector<S, N>& to, const SIMD_Vector<S, N>& what);
	//template<size_t N> SIMD_BitMask<N * 2> concat_masks(const SIMD_BitMask<SIMD_Vector<S, N>::LaneCount>& to, const SIMD_BitMask<SIMD_Vector<S, N>::LaneCount>& what);

	//Reinterprets value as vector of other type and returns the result.
	//If returned vector's size is smaller than input, input's upper bits are discarded
	//If returned vector's size is bigger than input, upper bits of returned value are undefined.
	template<typename T, typename S, size_t N> requires (T::IsSimdVector) T vcast(const SIMD_Vector<S, N>& value);

	//Reinterprets value as any other type and returns the result.
	//If returned value's size is smaller than input, input's upper bits are discarded
	//If returned value's size is bigger than input, upper bits of returned value are undefined.
	template<typename T, typename S, size_t N> T vreinterpret(const SIMD_Vector<S, N>& value);

	//Selects elements from two input vectors by corresponding mask bits and returns the result.
	//If the mask bit is 0, the corresponding element of `ifBitClear` is chosen
	//If the mask bit is 1, the corresponding element of `ifBitSet` is chosen
	//This function differs from blend only by the order of it's arguments
	//ret[i] = mask[i] ? ifBitSet[i] : ifBitClear[i]
	template <typename S, size_t N> SIMD_Vector<S, N> mask_mov(const SIMD_Vector<S, N>& ifBitClear, const SIMD_BitMask<SIMD_Vector<S, N>::LaneCount>& mask, const SIMD_Vector<S, N>& ifBitSet);
	//Selects elements from input vector by corresponding mask bits and returns the result.
	//If the mask bit is 0, the corresponding element of the returned vector is set to zero
	//If the mask bit is 1, the corresponding element of `ifBitSet` is chosen
	//ret[i] = mask[i] ? ifBitSet[i] : S(0)
	template <typename S, size_t N> SIMD_Vector<S, N> maskz_mov(const SIMD_BitMask<SIMD_Vector<S, N>::LaneCount>& mask, const SIMD_Vector<S, N>& ifBitSet);
	//Selects elements from two input vectors by corresponding mask bits and returns the result.
	//If the mask bit is 0, the corresponding element of `ifBitClear` is chosen
	//If the mask bit is 1, the corresponding element of `ifBitSet` is chosen
	//This function differs from mask_mov only by the order of it's arguments
	//ret[i] = mask[i] ? ifBitSet[i] : ifBitClear[i]
	template <typename S, size_t N> SIMD_Vector<S, N> blend(const SIMD_BitMask<SIMD_Vector<S, N>::LaneCount>& mask, const SIMD_Vector<S, N>& ifBitClear, const SIMD_Vector<S, N>& ifBitSet);

	//Loads the vector from memory location pointed to by `p` and returns the result.
	//If the corresponding mask bit is set, the corresponding element in memory is read and stored into the returned vector
	//If the corresponding mask bit is cleared, the corresponding element in memory is not read and the corresponding element from src is stored into the retuned vector
	//Masked out elements are guaranteed to not cause memory-related faults
	//ret[i] = mask[i] ? reinterpret_cast<const S*>(p)[i] : src[i]
	template<typename S, size_t N> SIMD_Vector<S, N> load(const void* p, const SIMD_BitMask<SIMD_Vector<S, N>::LaneCount>& mask = SIMD_BitMask<SIMD_Vector<S, N>::LaneCount>::AllOnes, const SIMD_Vector<S, N>& src = 0);
	//Loads the vector from memory location pointed to by `p` and returns the result.
	//If the corresponding mask bit is set, the corresponding element in memory is read and stored into the returned vector
	//If the corresponding mask bit is cleared, the corresponding element in memory is not read and the corresponding element from src is stored into the retuned vector
	//Masked out elements are guaranteed to not cause memory-related faults
	//ret[i] = mask[i] ? reinterpret_cast<const S*>(p)[i] : src[i]
	template<typename T> requires (T::IsSimdVector)
		__forceinline T load(const void* p, const SIMD_BitMask<T::LaneCount>& mask = SIMD_BitMask<T::LaneCount>::AllOnes, const T& src = 0)
	{
		return load<typename T::ScalarType, T::LaneCount>(p, mask, src);
	}

	//Conditionally stores vector `v` to memory location pointed by `p` using mask `mask`.
	//If the corresponding mask bit is set, the corresponding element of `v` is stored into the memory
	//Else, no action is performed
	//Masked out elements are guaranteed to not cause memory-related faults
	//if (mask[i]) reinterpret_cast<S*>(p)[i] = v[i]
	template<typename S, size_t N> void store(const SIMD_Vector<S, N>& v, void* p, const SIMD_BitMask<SIMD_Vector<S, N>::LaneCount>& mask = SIMD_BitMask<SIMD_Vector<S, N>::LaneCount>::AllOnes);

	//Conditionally gathers elements from memory, stores them into a vector and returns the result.
	//If the corresponding mask bit is set, the corresponding element in memory is read and stored into the returned vector
	//If the corresponding mask bit is cleared, the corresponding element in memory is not read and the corresponding element from src is stored into the retuned vector
	//Masked out elements are guaranteed to not cause memory-related faults
	//By default, scale is set to the size of vector's scalar type
	//ret[i] = mask[i] ? *reinterpret_cast<const S*>(size_t(base) + Scale*ind[i]) : src[i]
	template<typename S, size_t N, size_t Scale = sizeof(S), typename I> requires (std::is_integral_v<I> && sizeof(I) <= 8 && concepts::IsScalarType<S>)
	__forceinline SIMD_Vector<S, N> gather(const void* base, const SIMD_Vector<I, N>& ind, const SIMD_BitMask<SIMD_Vector<S, N>::LaneCount>& mask = SIMD_BitMask<SIMD_Vector<S, N>::LaneCount>::AllOnes, const SIMD_Vector<S, N>& src = 0)
	{
		return __gather_impl<S, N, Scale>(base, ind, mask, src);
	}

	//Conditionally gathers elements from memory, stores them into a vector and returns the result.
	//If the corresponding mask bit is set, the corresponding element in memory is read and stored into the returned vector
	//If the corresponding mask bit is cleared, the corresponding element in memory is not read and the corresponding element from src is stored into the retuned vector
	//Masked out elements are guaranteed to not cause memory-related faults
	//By default, scale is set to the size of vector's scalar type
	//ret[i] = mask[i] ? *reinterpret_cast<const S*>(size_t(base) + Scale*ind[i]) : src[i]
	template <typename T, size_t Scale = sizeof(typename T::ScalarType), typename I>
		requires (T::IsSimdVector)
	__forceinline T gather(const void* base, const SIMD_Vector<I, T::LaneCount>& ind, const SIMD_BitMask<T::LaneCount>& mask = SIMD_BitMask<T::LaneCount>::AllOnes, const T& src = 0)
	{
		return __gather_impl<typename T::ScalarType, T::LaneCount, Scale>(base, ind, mask, src);
	}

	//Conditionally scatters vector `v` to memory location pointed by `base` using mask `mask`.
	//If the corresponding mask bit is set, the corresponding element of `v` is stored into the memory
	//Else, no action is performed
	//Masked out elements are guaranteed to not cause memory-related faults
	//By default, scale is set to the size of vector's scalar type
	//if (mask[i]) *reinterpret_cast<S*>(size_t(base) + Scale*ind[i]) = v[i]
	template<typename S, size_t N, size_t Scale = sizeof(S), typename I> void scatter(const SIMD_Vector<S, N>& vec, void* base, const SIMD_Vector<I, N>& ind, const SIMD_BitMask<SIMD_Vector<S, N>::LaneCount>& mask = SIMD_BitMask<SIMD_Vector<S, N>::LaneCount>::AllOnes);

	//Performs the element-wise comparsion and returns the resultant mask.
	//If elements are equal, the corresponding mask bit is set to 1
	//Otherwise, the corresponding mask bit is set to 0
	//ret[i] = a[i] == b[i]
	template<typename S, size_t N> SIMD_BitMask<SIMD_Vector<S, N>::LaneCount> cmp_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);
	//Performs the element-wise comparsion and returns the resultant mask.
	//If elements are not equal, the corresponding mask bit is set to 1
	//Otherwise, the corresponding mask bit is set to 0
	//ret[i] = a[i] != b[i]
	template<typename S, size_t N> SIMD_BitMask<SIMD_Vector<S, N>::LaneCount> cmp_not_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);
	//Performs the element-wise comparsion and returns the resultant mask.
	//If element of vector `a` is less than element of vector `b`, the corresponding mask bit is set to 1
	//Otherwise, the corresponding mask bit is set to 0
	//ret[i] = a[i] < b[i]
	template<typename S, size_t N> SIMD_BitMask<SIMD_Vector<S, N>::LaneCount> cmp_less(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);
	//Performs the element-wise comparsion and returns the resultant mask.
	//If element of vector `a` is less than or equal to element of vector `b`, the corresponding mask bit is set to 1
	//Otherwise, the corresponding mask bit is set to 0
	//ret[i] = a[i] <= b[i]
	template<typename S, size_t N> SIMD_BitMask<SIMD_Vector<S, N>::LaneCount> cmp_less_or_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);
	//Performs the element-wise comparsion and returns the resultant mask.
	//If element of vector `a` is greater than element of vector `b`, the corresponding mask bit is set to 1
	//Otherwise, the corresponding mask bit is set to 0
	//ret[i] = a[i] > b[i]
	template<typename S, size_t N> SIMD_BitMask<SIMD_Vector<S, N>::LaneCount> cmp_greater(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);
	//Performs the element-wise comparsion and returns the resultant mask.
	//If element of vector `a` is greater than or equal to element of vector `b`, the corresponding mask bit is set to 1
	//Otherwise, the corresponding mask bit is set to 0
	//ret[i] = a[i] >= b[i]
	template<typename S, size_t N> SIMD_BitMask<SIMD_Vector<S, N>::LaneCount> cmp_greater_or_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);

	//Returns absolute value of the input vector
	//The returned values are undefined for signed elements equal to their minimum value
	template<typename S, size_t N> SIMD_Vector<S, N> abs(const SIMD_Vector<S, N>& a);
	//Rounds each element of input vector towards negative infinity (floor) and returns the result.
	template<typename S, size_t N> requires (std::is_floating_point_v<S>) SIMD_Vector<S, N> floor(const SIMD_Vector<S, N>& a);
	//Rounds each element of input vector towards positive infinity (ceil) and returns the result.
	template<typename S, size_t N> requires (std::is_floating_point_v<S>) SIMD_Vector<S, N> ceil(const SIMD_Vector<S, N>& a);
	//Compares two vectors together element-wise and returns the lower ones.
	//ret[i] = a[i] < b[i] ? a[i] : b[i]
	template<typename S, size_t N> SIMD_Vector<S, N> min(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);
	//Compares two vectors together element-wise and returns the higher ones.
	//ret[i] = a[i] > b[i] ? a[i] : b[i]
	template<typename S, size_t N> SIMD_Vector<S, N> max(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);
	//Compares vector `val` to vectors `min` and `max`
	//If corresponding value in `val` is outside the range [min..max], it will be forced to min or max.
	//tmp = val[i] < min[i] ? min[i] : val[i]
	//ret[i] = tmp > max[i] ? max[i] : tmp
	template<typename S, size_t N> SIMD_Vector<S, N> clamp(const SIMD_Vector<S, N>& val, const SIMD_Vector<S, N>& min, const SIMD_Vector<S, N>& max);

	//Split input vectors into 128-bit chunks. Upper half of each chunk are discarded.
	//For each result chunk, even elements are picked from a, while odd elements are picked from b.
	//Chunks are merged back into the resultant vector in the same order they appear in input vectors
	//chunk_ret[i] = i % 2 == 0 ? chunk_a[i/2] : chunk_b[i/2]
	template<typename S, size_t N> SIMD_Vector<S, N> unpacklo(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);
	//Split input vectors into 128-bit chunks. Lower half of each chunk are discarded.
	//For each result chunk, even elements are picked from a, while odd elements are picked from b.
	//Chunks are merged back into the resultant vector in the same order they appear in input vectors
	//x = 8 bytes / sizeof(S)
	//chunk_ret[i] = i % 2 == 0 ? chunk_a[x+i/2] : chunk_b[x+i/2]
	template<typename S, size_t N> SIMD_Vector<S, N> unpackhi(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);

	//Converts vector of half-precision floating point numbers (FP16) to single precision (FP32)
	template <size_t N> SIMD_Vector<float, N> vcvt_fp16_fp32(const SIMD_Vector<uint16_t, N>& a);
	//Converts vector of single precision floating point numbers (FP32) to half-precision (FP16)
	template <size_t N> SIMD_Vector<uint16_t, N> vcvt_fp32_fp16(const SIMD_Vector<float, N>& a);

	//Copies vector `src` and conditionally overwrites it with elements of vector `a`
	//Mask is iterated from lower bits to higher ones. 
	//If the mask bit is set, the corresponding element is read from `a` and is written to return vector at pivot point, 
	//advancing pivot point is by one element. Otherwise, no action is performed.
	//ret = src; pivot = 0
	//if (mask[i]) ret[pivot++] = a[i];
	template <typename S, size_t N> SIMD_Vector<S, N> compress(const SIMD_BitMask<SIMD_Vector<S, N>::LaneCount>& mask, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& src = 0);

	//Extracts sign bits of each element and returns them as mask. 
	//The mask bits are set to 1 if sign bits are 1 (negative), or 0 otherwise.
	template <typename S, size_t N> SIMD_BitMask<N> vec2mask(const SIMD_Vector<S, N>& v);
	//Sets all bits of each element to 0 if corresponding mask bit is 0, or 1 otherwise
	template <typename S, size_t N> SIMD_Vector<S,N> mask2vec(const SIMD_BitMask<N>& mask);
	/**
	@brief performs a lookup from lookup table using indices. The indices wrap around (-1 becomes LutElementCount-1, LutElementCount becomes 0, etc)
	//TODO: update the tooltip when implementing it
	Requires a power of two LutElementCount.
	@tparam S scalar type of the lookup elements
	@tparam LutElementCount count of elements inside the LUT
	@tparam GatherThresholdBytes if table size is greater than this parameter (in bytes), gather will be used instead of a permute network.
	@tparam N count of elements to look up 
	@tparam I type of index vector. If this type is larger than scalar type to be looked up (S), a gather is used unconditionally
	*/
	//template<typename S, size_t LutElementCount, size_t GatherThresholdBytes = 256, size_t N, typename I> SIMD_Vector<S,N> lookup(const S* lut, const SIMD_Vector<I, N>& ind);
}