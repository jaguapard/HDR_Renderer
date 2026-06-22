#pragma once
#include "namespace.h"
#include "meta/meta.h"
#include "FeatureSet.h"

namespace AVXXY_NAMESPACE
{
	template<typename S, size_t N>
		requires meta::IsValid_SIMD_Vector<S, N>
	class SIMD_Vector;

	template<size_t N>
	concept IsValid_SIMD_Mask = N >= 2 && N <= 64 && meta::isPowerOf2(N);

	template<meta::ScalarSizeClassEnum LS, size_t N>
	requires IsValid_SIMD_Mask<N>
	class SIMD_Mask;


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

		//Smallest unsigned integer type is able to hold of this mask's bits
		using BitsUintT = meta::bits_to_uint_t<N>;
		//Smallest signed integer type is able to hold of this mask's bits
		using VecIntT = meta::ScalarSizeTraits<LS>::IntT;
		//Vector type that has lane count equal to this mask's bit count, and whose lane size is the same as size class of this mask
		using VecT = SIMD_Vector<VecIntT, N>;

		static inline constexpr bool IsBitMask = internals::FS_current.has(internals::AVX512_F) || !internals::FS_current.has(internals::SSE);
		static inline constexpr bool IsVectorMask = !IsBitMask;
		static inline constexpr BitsUintT AllOnesUint = (N == sizeof(BitsUintT) * 8 ? ~BitsUintT(0) : ((BitsUintT(1) << N) - 1));
		
		SIMD_Mask() {};
		SIMD_Mask(BitsUintT bits);
		//Construct this mask by extracting uppermost bits of each lane and storing them the mask
		template<typename T> SIMD_Mask(const SIMD_Vector<T, N>& vec);
		//Constructs this mask by concatenating two masks of half it's size.
		//@param lo Lower half for the constructed mask
		//@param hi Upper half for the constructed mask
		SIMD_Mask(const SIMD_Mask<LS, N / 2>& lo, const SIMD_Mask<LS, N / 2>& hi);

		//Constructs this mask from other mask type. Logical bits are preserved.
		//If constructed mask has more bits that the input mask, the upper bits of the constructed mask are set to zero
		template <meta::ScalarSizeClassEnum LS2, size_t N2> requires (N >= N2)
		SIMD_Mask(const SIMD_Mask<LS2, N2>& other);

		//Returns true if bit i is set, false otherwise. Cannot be used to modify mask bits, for that use setBit
		bool operator[](size_t i) const;

		//Sets the bit i of the mask to 1 if value is true, or 0 otherwise
		void setBit(size_t i, bool value);

		//Returns lower half of this mask
		SIMD_Mask<LS, N / 2> lo() const;
		//Return upper half of this mask
		SIMD_Mask<LS, N / 2> hi() const;

		operator BitsUintT() const;

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

		//deposits uint bits to each lane of the vector.
		static VecT _movm(BitsUintT value);

		//extracts uppermost bits out of each lane of this mask and puts them into returned bits
		BitsUintT _movemask() const;
	};
}