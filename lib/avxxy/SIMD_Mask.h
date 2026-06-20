#pragma once
#include "namespace.h"
#include "concepts.h"
#include "FeatureSet.h"

template<typename S, size_t N>
class SIMD_Vector;

namespace AVXXY_NAMESPACE
{
	template<typename S, size_t N>
	class SIMD_Mask
	{
	public:
		static_assert(N >= 2);
		static_assert(N <= 64);
		static_assert(utils::isPowerOf2(N));
		template <typename FriendS, size_t FriendN>
		friend class SIMD_Mask;

		static inline constexpr size_t BitCount = N;
		using UintT = typename concepts::bits_to_uint_t<N>::type;
		using IntT = typename concepts::bits_to_int_t<N>::type;
		using VecT = SIMD_Vector<S, N>;
		static inline constexpr UintT AllOnesUint = (N == sizeof(UintT) * 8) ? ~UintT(0) : ((UintT(1) << N) - 1);
		static inline constexpr bool IsVectorMask = !internals::FS_current.has(internals::Feature::AVX512_F);
		static inline constexpr bool IsBitMask = !IsVectorMask;
		static SIMD_Mask<S, N> AllOnes();

		SIMD_Mask() {};
		SIMD_Mask(UintT bits);
		SIMD_Mask(const SIMD_Mask<S, N / 2>& lo, const SIMD_Mask<S, N / 2>& hi);
		SIMD_Mask(const SIMD_Vector<S, N>& v);

		template <typename S2>
		SIMD_Mask(const SIMD_Mask<S2, N>& other);

		operator UintT() const;

		//bool operator!

		
		template<typename T>
		//requires (concepts::IsIntrinsicVector<T> && ((concepts::xmm_sized<VecT> && concepts::xmm_sized<T>) || (concepts::ymm_sized<VecT> && concepts::ymm_sized<T>) || (concepts::zmm_sized<VecT> && concepts::zmm_sized<T>)))
		requires (std::is_convertible_v<T, SIMD_Vector<S,N>> && concepts::IsIntrinsicVector<T>)
		SIMD_Mask(const T& intrVec);

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
		//Returns true if bit i is set, false otherwise. Cannot be used to modify mask bits, for that use setBit
		bool operator[](size_t i) const;

		//Sets the bit i of the mask to 1 if value is true, or 0 otherwise
		void setBit(size_t i, bool value);

		SIMD_Mask<S, N / 2> lo() const;
		SIMD_Mask<S, N / 2> hi() const;

		SIMD_Mask<S, N> operator&(const SIMD_Mask<S, N>& other) const;
		SIMD_Mask<S, N> operator|(const SIMD_Mask<S, N>& other) const;
		SIMD_Mask<S, N> operator^(const SIMD_Mask<S, N>& other) const;
		SIMD_Mask<S, N> operator~() const;
		SIMD_Mask<S, N>& operator&=(const SIMD_Mask<S, N>& other);
		SIMD_Mask<S, N>& operator|=(const SIMD_Mask<S, N>& other);
		SIMD_Mask<S, N>& operator^=(const SIMD_Mask<S, N>& other);
	private:
		std::conditional_t<IsVectorMask, VecT, UintT> underlying;
	};
}