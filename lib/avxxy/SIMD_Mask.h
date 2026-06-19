#pragma once
#include "namespace.h"
#include "concepts.h"
#include "FeatureSet.h"

template<typename S, size_t N>
class SIMD_Vector;

namespace AVXXY_NAMESPACE
{
	//Represents a compacted mask with N bits.
	template<typename S, size_t N>
	struct SIMD_Mask
	{
	public:
		static_assert(N >= 2);
		static_assert(N <= 64);
		static_assert(utils::isPowerOf2(N));

		static inline constexpr size_t BitCount = N;
		using UintT = typename concepts::bits_to_uint_t<N>::type;
		using IntT = typename concepts::bits_to_int_t<N>::type;
		using VecT = SIMD_Vector<S, N>;
		//static inline constexpr UintT AllOnes = (N == sizeof(UintT) * 8) ? ~UintT(0) : ((UintT(1) << N) - 1);
		static inline constexpr bool IsVectorMask = !internals::FS_current.has(internals::Feature::AVX512_F);
		static inline constexpr bool IsBitMask = !IsVectorMask;
		static constexpr SIMD_Mask<S, N> AllOnes();


		SIMD_Mask(UintT bits);

		//Returns the vector type, where each lane is filled with 1 bits if corresponding mask bits are set, or 0 otherwise.
		//Thus, a SIMD_Mask<float, 4> with bits 0100 will return {0, std::bit_cast<float>(0xFFFFFFFF), 0, 0}
		VecT as_vector() const;

		//Returns this mask converted to smallest signed integer type that can hold it
		IntT as_int() const;

		//Returns this mask converted to smallest unsigned integer type that can hold it
		UintT as_uint() const;
		//explicit operator UintT() const;
		//explicit operator IntT() const;
		//explicit operator VecT() const;


	private:
		std::conditional_t<IsVectorMask, VecT, UintT> underlying;
	};
}