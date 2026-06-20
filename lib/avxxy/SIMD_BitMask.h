#pragma once
#include "namespace.h"
#include "concepts.h"

namespace AVXXY_NAMESPACE
{
#if 0
	//Represents a compacted mask with N bits.
	template<size_t N>
	struct SIMD_BitMask
	{
	public:
		static_assert(N >= 2);
		static_assert(N <= 64);
		static_assert(utils::isPowerOf2(N));

		static inline constexpr size_t BitCount = N;
		using UintT = typename concepts::bits_to_uint_t<N>::type;
		using IntT = typename concepts::bits_to_int_t<N>::type;
		static inline constexpr UintT AllOnes = (N == sizeof(UintT) * 8) ? ~UintT(0) : ((UintT(1) << N) - 1);

		SIMD_BitMask() {};
		SIMD_BitMask(UintT value);
		//Builds mask from two halves
		SIMD_BitMask(const SIMD_BitMask<N / 2>& lo, const SIMD_BitMask<N / 2>& hi);
		//Constructs mask from instrinsic vector of floating point type 
		/*template<typename T>
			requires (concepts::is_any_of_v<T, __m128, __m256, __m128d, __m256d>)
		SIMD_BitMask(const T& intrinsicVec);

		//Constructs mask from instrinsic vector of integral type. The type cannot be deduced automatically and should be provided by user in template argument S
		template<typename S, typename T>
			requires (concepts::is_any_of_v<T, __m128i, __m256i>&& std::is_integral_v<S>)
		SIMD_BitMask(const T& intrinsicVec);*/

		operator UintT() const;

		//Returns true if bit i is set, false otherwise. Cannot be used to modify mask bits, for that use setBit
		bool operator[](size_t i) const;

		//Sets the bit i of the mask to 1 if value is true, or 0 otherwise
		void setBit(size_t i, bool value);

		//Returns this mask represented as unsigned integer type of smallest size that can hold it
		UintT as_uint() const;

		//Returns this mask represented as signed integer type of smallest size that can hold it
		IntT as_int() const;

		SIMD_BitMask<N / 2> lo() const;
		SIMD_BitMask<N / 2> hi() const;

		SIMD_Mask<S,N>& operator&=(const SIMD_Mask<S,N>& other);
		SIMD_Mask<S,N>& operator|=(const SIMD_Mask<S,N>& other);
		SIMD_Mask<S,N>& operator^=(const SIMD_Mask<S,N>& other);
	private:
		UintT underlying;
	};
#endif
}