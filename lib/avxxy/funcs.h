#pragma once
#include "SIMD_Vector.h"
#include "SIMD_Mask.h"

//template <typename S, size_t N> class SIMD_Mask;
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
	template<meta::any_int S, size_t N, meta::any_int I>
	SIMD_Vector<S, N> shift_left(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& amount);

	//Shift packed integers in `a` right by the amount specified by the corresponding element of `amount` while shifting in zeros, and returns the result
	template<meta::any_int S, size_t N, meta::any_int I>
	SIMD_Vector<S, N> shift_right(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& amount);



	//Performs permutation on the elements from vector `a`. Elements of the returned vector are gathered from vector `a` by indices passed in `ind`.
	//Indices outside the range [0, N-1] wrap around N (-1 maps to N-1, N maps to 0).
	//ret[i] = a[ind[i] & (N-1)]
	template<typename S, size_t N, meta::any_int I> SIMD_Vector<S, N> permx(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& ind);

	//Appends vector `b` to vector `a`, then performs permutation on the elements from this temporary value. 
	//Elements of the returned vector are gathered from temporary vector by indices passed in `ind` 
	//Indices outside the range [0, 2*N-1] wrap around 2*N (-1 maps to 2*N-1, 2*N maps to 0).
	//t = ind[i] & (2*N - 1)
	//ret[i] = t < N ? a[t] : b[t-N]
	template<typename S, size_t N, meta::any_int I> SIMD_Vector<S, N> permx2(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b, const SIMD_Vector<I, N>& ind);



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

	//vrzext - vector reinterpret and zero-extend
	//Reinterprets input vector as raw memory, zero-extends each element vector to size of S2 and returns the resultant vector
	//requires output scalar type to be larger or equal in size to input scalar type
	//i.e. vrzext<double>(SIMD_Vector<int8_t, 8>> will put the input element into lowest byte of 8 byte lane of output, upper 7 bytes will be zeroes, then reinterpreted as doubles and returned
	//If input and output scalar sizes match, the input vector is only reinterpreted as output vector
	template<typename S2, typename S, size_t N>
	requires (sizeof(S2) >= sizeof(S))
	SIMD_Vector<S2, N> vrzext(const SIMD_Vector<S, N>& a);

	//vrtrunc - vector reinterpret and truncate
	//Reinterprets input vector as raw memory and returns only the lowest sizeof(S) bytes in each elemnt, reinterpreted back to output type
	//requires output scalar type to be less or equal in size to input scalar type
	//i.e. vrtrunc<int16_t>(SIMD_Vector<double, 8>> will discard upper 6 bytes each input double and return the low 2 bytes of each element reinterpreted as int16_t
	//If input and output scalar sizes match, the input vector is only reinterpreted as output vector
	template<typename S2, typename S, size_t N>
	requires (sizeof(S2) <= sizeof(S))
	SIMD_Vector<S2, N> vrtrunc(const SIMD_Vector<S, N>& a);


	//Reinterprets input vector as vector of different scalar type
	//Lane count of output vector is computed automatically to match input's total size
	//This function is only available if size of input is divisible by size of output's scalar type
	//@tparam S2 scalar type of output vector
	//@tparam S scalar size of input vector
	//@tparam N lane count of input vector
	template<meta::IsScalarType S2, typename S, size_t N>
	requires (sizeof(SIMD_Vector<S,N>) % sizeof(S2) == 0)
	SIMD_Vector<S2, sizeof(SIMD_Vector<S, N>) / sizeof(S2)> vcast(const SIMD_Vector<S, N>& a);

	//Reinterprets input vector as vector of different scalar type
	//This function is only available if vector sizes match and size of input is divisible by size of output's scalar type
	//@tparam T vector type to be casted to (return type)
	//@tparam S scalar type of input vector
	//@tparam N input's lane count
	template<meta::IsSimdVector T, typename S, size_t N>
		requires ((sizeof(SIMD_Vector<S, N>) % sizeof(typename T::ScalarT) == 0) && sizeof(SIMD_Vector<S,N>) == sizeof(T))
	T vcast(const SIMD_Vector<S, N>& value);

	//Reinterprets value of any other same sized type and returns the result
	template<typename T, typename S, size_t N> 
	requires (sizeof(T) == sizeof(SIMD_Vector<S,N>))
	T vreinterpret(const SIMD_Vector<S, N>& value);

	//vreinterpret_us - vector reinterpret unsafe
	//Reinterprets value as any other type and returns the result.
	//If returned value's size is smaller than input, input's upper bits are discarded
	//If returned value's size is bigger than input, upper bits of returned value are undefined.
	//Unlike vcast, this function allows to reinterpret to any type of any size
	//Unlike vreinterpret, this functions doesn't check input sizes
	//@tparam T type to be casted to (return type)
	//@tparam S scalar type of input vector
	//@tparam N lane count of input vector
	template<typename T, typename S, size_t N> T vreinterpret_us(const SIMD_Vector<S, N>& value);

	//Selects elements from two input vectors by corresponding mask bits and returns the result.
	//If the mask bit is 0, the corresponding element of `ifBitClear` is chosen
	//If the mask bit is 1, the corresponding element of `ifBitSet` is chosen
	//This function differs from blend only by the order of it's arguments
	//ret[i] = mask[i] ? ifBitSet[i] : ifBitClear[i]
	template <typename S, size_t N> SIMD_Vector<S, N> mask_mov(const SIMD_Vector<S, N>& ifBitClear, const mask_t<S, N>& mask, const SIMD_Vector<S, N>& ifBitSet);
	//Selects elements from input vector by corresponding mask bits and returns the result.
	//If the mask bit is 0, the corresponding element of the returned vector is set to zero
	//If the mask bit is 1, the corresponding element of `ifBitSet` is chosen
	//ret[i] = mask[i] ? ifBitSet[i] : S(0)
	template <typename S, size_t N> SIMD_Vector<S, N> maskz_mov(const mask_t<S, N>& mask, const SIMD_Vector<S, N>& ifBitSet);
	//Selects elements from two input vectors by corresponding mask bits and returns the result.
	//If the mask bit is 0, the corresponding element of `ifBitClear` is chosen
	//If the mask bit is 1, the corresponding element of `ifBitSet` is chosen
	//This function differs from mask_mov only by the order of it's arguments
	//ret[i] = mask[i] ? ifBitSet[i] : ifBitClear[i]
	template <typename S, size_t N> SIMD_Vector<S, N> blend(const mask_t<S, N>& mask, const SIMD_Vector<S, N>& ifBitClear, const SIMD_Vector<S, N>& ifBitSet);

	//Loads vector from memory p and returns the result. The memory does not have to be aligned
	template<typename S, size_t N> SIMD_Vector<S, N> load(const void* p);

	//Loads vector from memory p and returns the result. The memory does not have to be aligned
	template<meta::IsSimdVector T> T __forceinline load(const void* p)
	{
		return load<typename T::ScalarT, T::LaneCount>(p);
	}

	//Loads vector from aligned memory p and returns the result. The pointer p must be aligned to boundary depending on output size:
	//16 bytes for vectors less than or equal to 16 bytes
	//32 bytes for vectors sized between 17 and 32 bytes inclusive
	//64 bytes for vectors larger than 32 bytes
	template<typename S, size_t N> SIMD_Vector<S, N> load_a(const void* p);

	//Loads vector from aligned memory p and returns the result. The pointer p must be aligned to boundary depending on output size:
	//16 bytes for vectors less than or equal to 16 bytes
	//32 bytes for vectors sized between 17 and 32 bytes inclusive
	//64 bytes for vectors larger than 32 bytes
	template<meta::IsSimdVector T> T __forceinline load_a(const void* p)
	{
		return load_a<typename T::ScalarT, T::LaneCount>(p);
	}

	//Loads the vector from unaligned memory location pointed to by `p` and returns the result.
	//If the corresponding mask bit is set, the corresponding element in memory is read and stored into the returned vector
	//If the corresponding mask bit is cleared, the corresponding element in memory is not read and the destination element is zeroed out
	//Masked out elements are guaranteed to not cause memory-related faults
	//ret[i] = mask[i] ? reinterpret_cast<const S*>(p)[i] : std::bit_cast<S>(0);
	template<typename S, size_t N> SIMD_Vector<S, N> load(const void* p, const mask_t<S, N>& mask);
	template<meta::IsSimdVector T> T __forceinline load(const void* p, const typename T::MaskT& mask)
	{
		return load<typename T::ScalarT, T::LaneCount>(p, mask);
	}

	//Loads the vector from unaligned memory location pointed to by `p` and returns the result.
	//If the corresponding mask bit is set, the corresponding element in memory is read and stored into the returned vector
	//If the corresponding mask bit is cleared, the corresponding element in memory is not read and the corresponding element from src is stored into the retuned vector
	//Masked out elements are guaranteed to not cause memory-related faults
	//ret[i] = mask[i] ? reinterpret_cast<const S*>(p)[i] : src[i]
	template<typename S, size_t N> SIMD_Vector<S, N> load(const void* p, const mask_t<S, N>& mask, const SIMD_Vector<S, N>& src);
	//Loads the vector from unaligned memory location pointed to by `p` and returns the result.
	//If the corresponding mask bit is set, the corresponding element in memory is read and stored into the returned vector
	//If the corresponding mask bit is cleared, the corresponding element in memory is not read and the corresponding element from src is stored into the retuned vector
	//Masked out elements are guaranteed to not cause memory-related faults
	//ret[i] = mask[i] ? reinterpret_cast<const S*>(p)[i] : src[i]
	template<meta::IsSimdVector T>
	__forceinline T load(const void* p, const typename T::MaskT& mask, const T& src)
	{
		return load<typename T::ScalarT, T::LaneCount>(p, mask, src);
	}

	//Conditionally stores vector `v` to memory location pointed by `p` using mask `mask`.
	//If the corresponding mask bit is set, the corresponding element of `v` is stored into the memory
	//Else, no action is performed
	//Masked out elements are guaranteed to not cause memory-related faults
	//if (mask[i]) reinterpret_cast<S*>(p)[i] = v[i]
	template<typename S, size_t N> void store(const SIMD_Vector<S, N>& v, void* p, const mask_t<S, N>& mask = mask_t<S, N>::AllOnesUint);

	//Conditionally gathers elements from memory, stores them into a vector and returns the result.
	//If the corresponding mask bit is set, the corresponding element in memory is read and stored into the returned vector
	//If the corresponding mask bit is cleared, the corresponding element in memory is not read and the corresponding element from src is stored into the retuned vector
	//Masked out elements are guaranteed to not cause memory-related faults
	//By default, scale is set to the size of vector's scalar type
	//ret[i] = mask[i] ? *reinterpret_cast<const S*>(size_t(base) + Scale*ind[i]) : src[i]
	template<typename S, size_t N, size_t Scale = sizeof(S), meta::any_int I>
		__forceinline SIMD_Vector<S, N> gather(const void* base, const SIMD_Vector<I, N>& ind, const mask_t<S, N>& mask = mask_t<S, N>::AllOnesUint, const SIMD_Vector<S, N>& src = 0)
	{
		return __gather_impl<S, N, Scale>(base, ind, mask, src);
	}

	//Conditionally gathers elements from memory, stores them into a vector and returns the result.
	//If the corresponding mask bit is set, the corresponding element in memory is read and stored into the returned vector
	//If the corresponding mask bit is cleared, the corresponding element in memory is not read and the corresponding element from src is stored into the retuned vector
	//Masked out elements are guaranteed to not cause memory-related faults
	//By default, scale is set to the size of vector's scalar type
	//ret[i] = mask[i] ? *reinterpret_cast<const S*>(size_t(base) + Scale*ind[i]) : src[i]
	template <meta::IsSimdVector T, size_t Scale = sizeof(typename T::ScalarT), meta::any_int I>
	__forceinline T gather(const void* base, const SIMD_Vector<I, T::LaneCount>& ind, const typename T::MaskT& mask = T::MaskT::AllOnesUint, const T& src = 0)
	{
		return __gather_impl<typename T::ScalarT, T::LaneCount, Scale>(base, ind, mask, src);
	}

	//Conditionally scatters vector `v` to memory location pointed by `base` using mask `mask`.
	//If the corresponding mask bit is set, the corresponding element of `v` is stored into the memory
	//Else, no action is performed
	//Masked out elements are guaranteed to not cause memory-related faults
	//By default, scale is set to the size of vector's scalar type
	//if (mask[i]) *reinterpret_cast<S*>(size_t(base) + Scale*ind[i]) = v[i]
	template<typename S, size_t N, size_t Scale = sizeof(S), meta::any_int I>
	void scatter(const SIMD_Vector<S, N>& vec, void* base, const SIMD_Vector<I, N>& ind, const mask_t<S, N>& mask = mask_t<S, N>::AllOnesUint);

	//Performs the element-wise comparsion and returns the resultant mask.
	//If elements are equal, the corresponding mask bit is set to 1
	//Otherwise, the corresponding mask bit is set to 0
	//ret[i] = a[i] == b[i]
	template<typename S, size_t N> mask_t<S, N> cmp_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);
	//Performs the element-wise comparsion and returns the resultant mask.
	//If elements are not equal, the corresponding mask bit is set to 1
	//Otherwise, the corresponding mask bit is set to 0
	//ret[i] = a[i] != b[i]
	template<typename S, size_t N> mask_t<S, N> cmp_not_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);
	//Performs the element-wise comparsion and returns the resultant mask.
	//If element of vector `a` is less than element of vector `b`, the corresponding mask bit is set to 1
	//Otherwise, the corresponding mask bit is set to 0
	//ret[i] = a[i] < b[i]
	template<typename S, size_t N> mask_t<S, N> cmp_less(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);
	//Performs the element-wise comparsion and returns the resultant mask.
	//If element of vector `a` is less than or equal to element of vector `b`, the corresponding mask bit is set to 1
	//Otherwise, the corresponding mask bit is set to 0
	//ret[i] = a[i] <= b[i]
	template<typename S, size_t N> mask_t<S, N> cmp_less_or_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);
	//Performs the element-wise comparsion and returns the resultant mask.
	//If element of vector `a` is greater than element of vector `b`, the corresponding mask bit is set to 1
	//Otherwise, the corresponding mask bit is set to 0
	//ret[i] = a[i] > b[i]
	template<typename S, size_t N> mask_t<S, N> cmp_greater(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);
	//Performs the element-wise comparsion and returns the resultant mask.
	//If element of vector `a` is greater than or equal to element of vector `b`, the corresponding mask bit is set to 1
	//Otherwise, the corresponding mask bit is set to 0
	//ret[i] = a[i] >= b[i]
	template<typename S, size_t N> mask_t<S, N> cmp_greater_or_equal(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);

	//Returns absolute value of the input vector
	//The returned values are undefined for signed elements equal to their minimum value
	template<typename S, size_t N> SIMD_Vector<S, N> abs(const SIMD_Vector<S, N>& a);
	//Rounds each element of input vector towards negative infinity (floor) and returns the result.
	template<meta::any_float S, size_t N> SIMD_Vector<S, N> floor(const SIMD_Vector<S, N>& a);
	//Rounds each element of input vector towards positive infinity (ceil) and returns the result.
	template<meta::any_float S, size_t N> SIMD_Vector<S, N> ceil(const SIMD_Vector<S, N>& a);
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
	//template <size_t N> SIMD_Vector<float, N> vcvt_fp16_fp32(const SIMD_Vector<uint16_t, N>& a);
	//Converts vector of single precision floating point numbers (FP32) to half-precision (FP16)
	//template <size_t N> SIMD_Vector<uint16_t, N> vcvt_fp32_fp16(const SIMD_Vector<float, N>& a);

	//Copies vector `src` and conditionally overwrites it with elements of vector `a`
	//Mask is iterated from lower bits to higher ones. 
	//If the mask bit is set, the corresponding element is read from `a` and is written to return vector at pivot point, 
	//advancing pivot point is by one element. Otherwise, no action is performed.
	//ret = src; pivot = 0
	//for (size_t i = 0; i < N; ++i)
	//    if (mask[i]) ret[pivot++] = a[i];
	template <typename S, size_t N> SIMD_Vector<S, N> compress(const mask_t<S, N>& mask, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& src = 0);

	//Iterates input vector from lowest elements to highest
	//For each element, checks elements below it, and sets corresponding output element's bit to 1 if it's equal to the tested element, or 0 otherwise
	//This function is only available for vectors in which elements have number of bits greater or equal to vector's lane count
	//ret = 0;
	//for (size_t i = 0; i < N; ++i)
	//    for (size_t j = 0; j < i; ++j)
	//        if (a[i] == a[j]) ret[i] |= 1 << j; 
	template <typename S, size_t N> requires (sizeof(S) * 8 >= N)
	SIMD_Vector<typename meta::ScalarTraits<S>::UintT, N> conflict(const SIMD_Vector<S, N>& a);

	//For each element in `a`, computes the number of set bits and stores the computed value into corresponding element of returned vector
	//for (size_t i = 0; i < N; ++i) ret[i] = popcnt(a[i])
	template<meta::vpopcnt_allowed S, size_t N> SIMD_Vector<typename meta::ScalarTraits<S>::UintT, N> vpopcnt(const SIMD_Vector<S, N>& a);

	//Extracts uppermost bit of each element and returns them as mask.
	template <typename S, size_t N> mask_t<S,N> movemask(const SIMD_Vector<S, N>& v);
	//Sets all bits of each element to 0 if corresponding mask bit is 0, or 1 otherwise.
	//@tparam S scalar type of the returned vector
	//@tparam N number of lanes in returned vector, same as bit count of input mask
	//@tparam C size class of the input mask
	template <typename S, meta::ScalarSizeClassEnum C, size_t N> SIMD_Vector<S, N> movm(const SIMD_Mask<C, N>& mask);

	//Reinterprets `a` as vector of bytes, then shuffles these bytes within 128-bit lanes by indices `b`.
	//After the shuffle is done, reinterprets the shuffled vector back to input type and returns it.
	//Only uppermost bit and 4 lowest bits of each index byte are used for the shuffle.
	//If uppermost bit of the index in `b` is set, then corresponding output lane is zeroed out.
	//Otherwise, the byte is taken from the same 128-bit lane `a` by index b[i] & 15.
	//for (size_t start = 0; start < sizeof(a); start += 16)
	//    for (size_t i = 0; i < 16; ++i)
	//        ret[start + i] = b[start + i] > 127 ? 0 : a[start + (b[i] & 15)]
	template<typename S, size_t N>
	SIMD_Vector<S, N> byte_shuffle(const SIMD_Vector<S, N>& a, const SIMD_Vector<uint8_t, N * sizeof(S)>& b);


	//Performs a block permutation of input vector by compile-time-known indices.
	//Requires size of input to be divisible by size of block.
	//Requires number of indices and number of blocks in input to match.
	//Requires all indices to be in range 0 to C-1 inclusive, where C in number of blocks in the input vector.
	//@tparam Block This type's size is used as permutation granularity. Only scalar and vector types are accepted
	//@tparam Idx zero-indexed source block indices. Output block i is copied from input block Idx[i]
	template<typename Block, size_t... Idx, typename S, size_t N>
	SIMD_Vector<S, N> permute(const SIMD_Vector<S, N>& a);

	//Performs a block permutation of 2 input vectors by compile-time-known indices.
	//Requires size of inputs to be divisible by size of block.
	//Requires number of indices and number of blocks in input to match.
	//Requires all indices to be in range 0 to 2*C-1 inclusive, where C in number of blocks in the input vector.
	
	//@tparam Block This type's size is used as permutation granularity. Only scalar and vector types are accepted
	//@tparam Idx zero-indexed source block indices. Output block i is copied from a's block Idx[i] if index is less than C or from b's block Idx[i] otherwise 
	template<typename Block, size_t... Idx, typename S, size_t N>
	SIMD_Vector<S, N> permute2(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b);

	//Loads a vector of same type as input from p using mask, then blends the loaded value with input vector, and stores the result back to p
	//Pointer p does not have to be aligned.
	//This operation is very similar to masked store, but umasked lanes may still cause memory-related faults.
	//Due to not requiring masking, it is preferred to use this function if caller guarantees that the load will not touch invalid memory
	//template<typename S, size_t N> blend_store(const SIMD_Vector<S, N>& v, const mask_t<S, N>& mask, void* p);
}