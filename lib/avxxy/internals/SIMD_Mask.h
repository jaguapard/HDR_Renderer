#pragma once
#include "../namespace.h"
#include "../meta/meta.h"
#include "FeatureSet.h"

namespace AVXXY_NAMESPACE
{
	template<typename S, size_t N>
		requires meta::IsValid_SIMD_Vector<S, N>
	class SIMD_Vector;

	namespace internals
	{
		template<typename S, size_t N>
		SIMD_Vector<S, N> _movm_raw(uint64_t mask);

		template<typename S, size_t N>
		uint64_t _movemask_raw(const SIMD_Vector<S, N>& v);
	}

	template<size_t N>
	concept IsValid_SIMD_Mask = N == 1 || (N >= 2 && N <= 64 && meta::isPowerOf2(N));

	template<meta::ScalarSizeClassEnum LS, size_t N>
	requires IsValid_SIMD_Mask<N>
	class SIMD_Mask;


	namespace internals
	{
		//SIMD_Mask is a semantic type that can be either vector mask or bit mask underneath, depending on current feature set.
		//AVX512F or absent SSE makes this a bit mask. Otherwise, it's a vector mask 
		//@tparam LS size class of the masks' elements. Ignored in case of bit mask
		//@tparam N logical bit count of the mask. The actual storage holds at least this much bits
		template<meta::ScalarSizeClassEnum LS, size_t N>
			requires IsValid_SIMD_Mask<N>
		class SIMD_Mask
		{
		public:
			template <meta::ScalarSizeClassEnum FriendLS, size_t FriendN>
				requires IsValid_SIMD_Mask<FriendN>
			friend class SIMD_Mask;

			//Smallest unsigned integer type is able to hold all of this mask's bits
			using BitsUintT = meta::bits_to_uint_t<N>;
			//Smallest signed integer type is able to hold all of this mask's bits
			using VecIntT = meta::ScalarSizeTraits<LS>::IntT;
			//Vector type that has lane count equal to this mask's bit count, and whose lane size is the same as size class of this mask
			using VecT = SIMD_Vector<VecIntT, N>;

			static inline constexpr bool IsBitMask = internals::FS_current.has(internals::AVX512_F) || !internals::FS_current.has(internals::SSE);
			static inline constexpr bool IsVectorMask = !IsBitMask;
			static inline constexpr BitsUintT AllOnesUint = (N == sizeof(BitsUintT) * 8 ? ~BitsUintT(0) : ((BitsUintT(1) << N) - 1));

			SIMD_Mask() {};

			//Constructs this mask by copying bits to it's internal storage
			SIMD_Mask(BitsUintT bits);

			//Constructs this mask from intrinsic vector of same size class by extracting uppermost bits of each of it's elements
			//Input's elements are assumed to be the same size as this masks's scalar size class
			//I.e. for byte-class masks, bits 7, 15, 23, ... will be extracted from input vector
			//for word-class: 15, 31, 47, etc.
			template<typename T>
				requires (meta::IsIntrinsicVector<T>&& meta::SameSizeClasses<(sizeof(typename SIMD_Mask<LS, N>::VecT)), (sizeof(T))>)
			SIMD_Mask(const T& intrinsicVec);

			//Construct this mask by extracting uppermost bits of each lane and storing them the mask.
			//T is not required to be same size as this mask's scalar size class
			template<typename T> SIMD_Mask(const SIMD_Vector<T, N>& vec);

			//Constructs this mask by concatenating two masks of half it's size.
			//@param lo Lower half for the constructed mask
			//@param hi Upper half for the constructed mask
			template<size_t N2>
				requires (N2 * 2 == N)
			SIMD_Mask(const SIMD_Mask<LS, N2>& lo, const SIMD_Mask<LS, N2>& hi);

			//Constructs this mask from other mask type. Logical bits are preserved.
			//If constructed mask has more bits that the input mask, the upper bits of the constructed mask are set to zero
			template <meta::ScalarSizeClassEnum LS2, size_t N2> requires (N >= N2)
				SIMD_Mask(const SIMD_Mask<LS2, N2>& other);

			//Returns true if bit i is set, false otherwise. Cannot be used to modify mask bits, for that use setBit
			bool operator[](size_t i) const;

			//Sets the bit i of the mask to 1 if value is true, or 0 otherwise
			void setBit(size_t i, bool value);

			//Returns lower half of this mask
			auto lo() const requires (N >= 2);
			//Return upper half of this mask
			auto hi() const requires (N >= 2);

			operator BitsUintT() const;

			//Converts this mask to intrinsic vector, whose elements have the same size class as this mask's scalar size class
			//The returned vector is guaranteed to have all bits set to one for lanes corresponding to this mask's set bits, and all zeros for cleared bits
			template<typename T>
				requires (meta::IsIntrinsicVector<T> && (meta::ScalarSizeTraits<LS>::ByteSize* N == sizeof(T)))
			operator T() const;

			SIMD_Mask<LS, N> operator&(const SIMD_Mask<LS, N>& other) const;
			SIMD_Mask<LS, N> operator|(const SIMD_Mask<LS, N>& other) const;
			SIMD_Mask<LS, N> operator^(const SIMD_Mask<LS, N>& other) const;
			SIMD_Mask<LS, N> operator~() const;
			SIMD_Mask<LS, N>& operator&=(const SIMD_Mask<LS, N>& other);
			SIMD_Mask<LS, N>& operator|=(const SIMD_Mask<LS, N>& other);
			SIMD_Mask<LS, N>& operator^=(const SIMD_Mask<LS, N>& other);
		private:
			using SizeTraits = meta::ScalarSizeTraits<LS>;
			std::conditional_t<IsBitMask, BitsUintT, VecT> underlying;
		};

		//this magic detour allows mask_t to work, don't touch it
		template<typename S, size_t N>
		struct simd_mask_helper
		{
			using type = SIMD_Mask<meta::ScalarTraits<S>::size_class, N>;
		};
	}
	//mask type that has size class equal to that of S, and that can hold at least N logical bits. Use this instead of manual instantiation of SIMD_Mask 
	template<typename S, size_t N>
	using mask_t = typename internals::simd_mask_helper<S, N>::type;
}