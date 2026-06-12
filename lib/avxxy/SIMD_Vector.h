#pragma once
#include "concepts.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <cstring>

namespace AVXXY_NAMESPACE
{
	template<typename _S, size_t _N> concept IsValid_SIMD_Vector = _N >= 2 && _N <= 64 && utils::isPowerOf2(_N) && concepts::IsScalarType<_S>; //for now, bigger than 64 lanes vectors are not supported (mainly due to mask type not being ready for it)

	template<typename Vec, typename IntrinVec>
	concept ConversionToNativeVectorLegal =
		(concepts::xmm_sized<Vec> && std::is_same_v<IntrinVec, typename concepts::reg128<typename Vec::ScalarType>::type>) ||
		(concepts::ymm_sized<Vec> && std::is_same_v<IntrinVec, typename concepts::reg256<typename Vec::ScalarType>::type>) ||
		(concepts::zmm_sized<Vec> && std::is_same_v<IntrinVec, typename concepts::reg512<typename Vec::ScalarType>::type>);

	template<typename _S, size_t _N>
		requires IsValid_SIMD_Vector<_S, _N>
	struct alignas(std::min<uint32_t>(64, sizeof(_S)* _N)) SIMD_Vector
	{
		template<typename FriendS, size_t FriendN> requires IsValid_SIMD_Vector<FriendS, FriendN>
		friend struct SIMD_Vector;

		using ScalarType = _S;
		using Self = SIMD_Vector<_S, _N>;

		static inline constexpr size_t LaneCount = _N;
		//Size of vector's active elements. The size of actual struct (sizeof(SIMD_Vector)) may differ from it due to padding and unused elements
		static inline constexpr size_t ActiveByteSize = sizeof(ScalarType) * LaneCount;
		static inline constexpr bool IsSimdVector = true;

		SIMD_Vector() {};
		const ScalarType& operator[](size_t i) const { return arr[i]; }
		ScalarType& operator[](size_t i) { return arr[i]; }

		static SIMD_Vector<_S, _N> iota()
		{
			SIMD_Vector<_S, _N> ret;
			for (size_t i = 0; i < _N; ++i) ret[i] = i;
		}
		//Copies and returns lower half of this vector
		SIMD_Vector<ScalarType, LaneCount / 2> lo() const
			requires (LaneCount >= 4)
		{
			SIMD_Vector<ScalarType, LaneCount / 2> ret;
			memcpy(ret.arr.data(), arr.data(), sizeof(ret));
			return ret;
		}
		//Copies and returns upper half of this vector
		SIMD_Vector<ScalarType, LaneCount / 2> hi() const
			requires (LaneCount >= 4)
		{
			SIMD_Vector<ScalarType, LaneCount / 2> ret;
			memcpy(ret.arr.data(), arr.data() + LaneCount / 2, sizeof(ret));
			return ret;
		}
		//Copies and returns lower half of this vector
		ScalarType lo() const
			requires (LaneCount == 2)
		{
			return arr[0];
		}
		//Copies and returns upper half of this vector
		ScalarType hi() const
			requires (LaneCount == 2)
		{
			return arr[1];
		}

		//Constructs this vector from other vector. If vector scalar types mismatch, the input is converted to this vector's scalar type before assignment
		template<typename T>
		SIMD_Vector(const SIMD_Vector<T, LaneCount>& other) { *this = vcvt<ScalarType>(other); }
		//Broadcasts a scalar value to all lanes of a vector. The input value is converted to vector's scalar type before broadcasting
		template<typename T> requires concepts::IsScalarType<T>
		__forceinline SIMD_Vector(const T& s) { for (size_t i = 0; i < LaneCount; ++i) (*this)[i] = s; }

		//Constructs vector from it's intrinsic type. The intrinsic vector type must be of the same size class as constructed vector:
		//Vectors less than 17 bytes can be constructed from 128 bit intrinsic types.
		//Vectors between 17 and 32 bytes can be constructed from 256 bit intrinsic types.
		//Vectors between 33 and 64 bytes can be constructed from 512 bit intrinsic types.
		//Integral intrinsic vectors can be used to construct any integral SIMD_Vector of same size class
		//Floating point vectors require the scalar type of intrinsic vector and SIMD_Vector to match
		//If SIMD_Vector and intrinsic vector sizes mismatch, only the lowest bits of intrinsic vector are copied to the constructed SIMD_Vector
		template <typename T>
		__forceinline SIMD_Vector(const T& intrinsicVec) requires(concepts::IsIntrinsicVector<T>&& ConversionToNativeVectorLegal<Self, T>)
		{
			memcpy(arr.data(), &intrinsicVec, std::min(sizeof(Self), sizeof(T)));
		}

		//Constructs vector from halves
		template<typename S, size_t N>
			requires (N * 2 == LaneCount)
		__forceinline SIMD_Vector(const SIMD_Vector<S, N>& lo, const SIMD_Vector<S, N>& hi)
		{
			memcpy(arr.data(), lo.arr.data(), sizeof(arr) / 2);
			memcpy(arr.data() + N, hi.arr.data(), sizeof(arr) / 2);
		}

		//Represents vector as it's intrinsic type.  The intrinsic vector type is in the same size class as this vector:
		//Vectors less than 17 bytes: 128 bit intrinsic types.
		//Vectors between 17 and 32 bytes: 256 bit intrinsic types.
		//Vectors between 33 and 64 bytes: 512 bit intrinsic types.
		//Integral SIMD_Vectors convert to integral intrinsic vectors
		//Floating point SIMD_Vectors convert to intrinsic vectors of same scalar type
		//If SIMD_Vector and intrinsic vector sizes mismatch, upper bits of the returned vector are undefined
		template <typename T>
		__forceinline operator T() const requires(concepts::IsIntrinsicVector<T>&& ConversionToNativeVectorLegal<Self, T>)
		{
			T ret;
			memcpy(&ret, arr.data(), std::min(sizeof(T), sizeof(Self)));
			return ret;
		}

		//Reinterprets this vector as another SIMD_Vector.
		// If returned vector's size is smaller than this vector, only the lower bits are used. 
		// If returned vector is larger than this vector, then upper bits of the returned vector are undefined.
		/*
		template<typename T>
		requires (T::IsSimdVector)
		T vcast() const
		{
			T ret;
			memcpy(ret.arr.data(), arr.data(), std::min(sizeof(ret), sizeof(*this)));
			return ret;
		}*/

		//do NOT change these to variadic templates. We want users to see that the type has ctor from N scalars, not guess while typing. So something similar to _mm*_setr_* instead of printf
		// Constructs this vector from 2 scalar values. 
		// Each value is converted to vector's scalar type before assignment. 
		// This constructor is only available for 2-element vectors
		template <typename T0, typename T1>
			requires (LaneCount == 2 && concepts::AllAreScalarTypes<T0, T1>)
		__forceinline SIMD_Vector(T0 s0, T1 s1)
		{
			(*this)[0] = s0; (*this)[1] = s1;
		}

		// Constructs this vector from 4 scalar values. 
		// Each value is converted to vector's scalar type before assignment. 
		// This constructor is only available for 4-element vectors
		template <typename T0, typename T1, typename T2, typename T3>
			requires (LaneCount == 4 && concepts::AllAreScalarTypes<T0, T1, T2, T3>)
		__forceinline SIMD_Vector(T0 s0, T1 s1, T2 s2, T3 s3)
		{
			(*this)[0] = s0; (*this)[1] = s1; (*this)[2] = s2; (*this)[3] = s3;
		}

		// Constructs this vector from 8 scalar values. 
		// Each value is converted to vector's scalar type before assignment. 
		// This constructor is only available for 8-element vectors.
		template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
			requires (LaneCount == 8 && concepts::AllAreScalarTypes<T0, T1, T2, T3, T4, T5, T6, T7>)
		__forceinline SIMD_Vector(T0 s0, T1 s1, T2 s2, T3 s3, T4 s4, T5 s5, T6 s6, T7 s7)
		{
			(*this)[0] = s0; (*this)[1] = s1; (*this)[2] = s2; (*this)[3] = s3;
			(*this)[4] = s4; (*this)[5] = s5; (*this)[6] = s6; (*this)[7] = s7;
		}

		// Constructs this vector from 16 scalar values. 
		// Each value is converted to vector's scalar type before assignment. 
		// This constructor is only available for vectors with LaneCount == 16.
		template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8,
			typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15>
			requires (LaneCount == 16 && concepts::AllAreScalarTypes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15>)
		__forceinline SIMD_Vector(T0 s0, T1 s1, T2 s2, T3 s3, T4 s4, T5 s5, T6 s6, T7 s7, T8 s8, T9 s9, T10 s10, T11 s11, T12 s12, T13 s13, T14 s14, T15 s15)
		{
			(*this)[0] = s0; (*this)[1] = s1; (*this)[2] = s2; (*this)[3] = s3;
			(*this)[4] = s4; (*this)[5] = s5; (*this)[6] = s6; (*this)[7] = s7;
			(*this)[8] = s8; (*this)[9] = s9; (*this)[10] = s10; (*this)[11] = s11;
			(*this)[12] = s12; (*this)[13] = s13; (*this)[14] = s14; (*this)[15] = s15;
		}

		// Constructs this vector from 32 scalar values. 
		// Each value is converted to vector's scalar type before assignment. 
		// This constructor is only available for vectors with LaneCount == 32.
		template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15, typename T16, typename T17, typename T18, typename T19, typename T20, typename T21, typename T22, typename T23, typename T24, typename T25, typename T26, typename T27, typename T28, typename T29, typename T30, typename T31>
			requires (LaneCount == 32 && concepts::AllAreScalarTypes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31>)
		__forceinline SIMD_Vector(T0 s0, T1 s1, T2 s2, T3 s3, T4 s4, T5 s5, T6 s6, T7 s7, T8 s8, T9 s9, T10 s10, T11 s11, T12 s12, T13 s13, T14 s14, T15 s15, T16 s16, T17 s17, T18 s18, T19 s19, T20 s20, T21 s21, T22 s22, T23 s23, T24 s24, T25 s25, T26 s26, T27 s27, T28 s28, T29 s29, T30 s30, T31 s31)
		{
			(*this)[0] = s0; (*this)[1] = s1; (*this)[2] = s2; (*this)[3] = s3;
			(*this)[4] = s4; (*this)[5] = s5; (*this)[6] = s6; (*this)[7] = s7;
			(*this)[8] = s8; (*this)[9] = s9; (*this)[10] = s10; (*this)[11] = s11;
			(*this)[12] = s12; (*this)[13] = s13; (*this)[14] = s14; (*this)[15] = s15;
			(*this)[16] = s16; (*this)[17] = s17; (*this)[18] = s18; (*this)[19] = s19;
			(*this)[20] = s20; (*this)[21] = s21; (*this)[22] = s22; (*this)[23] = s23;
			(*this)[24] = s24; (*this)[25] = s25; (*this)[26] = s26; (*this)[27] = s27;
			(*this)[28] = s28; (*this)[29] = s29; (*this)[30] = s30; (*this)[31] = s31;
		}

		// Constructs this vector from 64 scalar values. 
		// Each value is converted to vector's scalar type before assignment. 
		// This constructor is only available for vectors with LaneCount == 64.
		template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15, typename T16, typename T17, typename T18, typename T19, typename T20, typename T21, typename T22, typename T23, typename T24, typename T25, typename T26, typename T27, typename T28, typename T29, typename T30, typename T31, typename T32, typename T33, typename T34, typename T35, typename T36, typename T37, typename T38, typename T39, typename T40, typename T41, typename T42, typename T43, typename T44, typename T45, typename T46, typename T47, typename T48, typename T49, typename T50, typename T51, typename T52, typename T53, typename T54, typename T55, typename T56, typename T57, typename T58, typename T59, typename T60, typename T61, typename T62, typename T63>
			requires (LaneCount == 64 && concepts::AllAreScalarTypes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63>)
		__forceinline SIMD_Vector(T0 s0, T1 s1, T2 s2, T3 s3, T4 s4, T5 s5, T6 s6, T7 s7, T8 s8, T9 s9, T10 s10, T11 s11, T12 s12, T13 s13, T14 s14, T15 s15, T16 s16, T17 s17, T18 s18, T19 s19, T20 s20, T21 s21, T22 s22, T23 s23, T24 s24, T25 s25, T26 s26, T27 s27, T28 s28, T29 s29, T30 s30, T31 s31, T32 s32, T33 s33, T34 s34, T35 s35, T36 s36, T37 s37, T38 s38, T39 s39, T40 s40, T41 s41, T42 s42, T43 s43, T44 s44, T45 s45, T46 s46, T47 s47, T48 s48, T49 s49, T50 s50, T51 s51, T52 s52, T53 s53, T54 s54, T55 s55, T56 s56, T57 s57, T58 s58, T59 s59, T60 s60, T61 s61, T62 s62, T63 s63)
		{
			(*this)[0] = s0; (*this)[1] = s1; (*this)[2] = s2; (*this)[3] = s3; (*this)[4] = s4; (*this)[5] = s5; (*this)[6] = s6; (*this)[7] = s7; (*this)[8] = s8;
			(*this)[9] = s9; (*this)[10] = s10; (*this)[11] = s11; (*this)[12] = s12; (*this)[13] = s13; (*this)[14] = s14; (*this)[15] = s15;
			(*this)[16] = s16; (*this)[17] = s17; (*this)[18] = s18; (*this)[19] = s19; (*this)[20] = s20; (*this)[21] = s21; (*this)[22] = s22; (*this)[23] = s23;
			(*this)[24] = s24; (*this)[25] = s25; (*this)[26] = s26; (*this)[27] = s27; (*this)[28] = s28; (*this)[29] = s29; (*this)[30] = s30; (*this)[31] = s31;
			(*this)[32] = s32; (*this)[33] = s33; (*this)[34] = s34; (*this)[35] = s35; (*this)[36] = s36; (*this)[37] = s37; (*this)[38] = s38; (*this)[39] = s39;
			(*this)[40] = s40; (*this)[41] = s41; (*this)[42] = s42; (*this)[43] = s43; (*this)[44] = s44; (*this)[45] = s45; (*this)[46] = s46; (*this)[47] = s47;
			(*this)[48] = s48; (*this)[49] = s49; (*this)[50] = s50; (*this)[51] = s51; (*this)[52] = s52; (*this)[53] = s53; (*this)[54] = s54; (*this)[55] = s55;
			(*this)[56] = s56; (*this)[57] = s57; (*this)[58] = s58; (*this)[59] = s59; (*this)[60] = s60; (*this)[61] = s61; (*this)[62] = s62; (*this)[63] = s63;
		}
	private:
		std::array<ScalarType, LaneCount> arr;
	};

	template<typename S, size_t N>
	std::ostream& operator<<(std::ostream& os, const SIMD_Vector<S, N>& a)
	{
		for (size_t i = 0; i < N; ++i)
		{
			if constexpr (concepts::any_i8<S>) os << int(a[i]);
			else os << a[i];
			if (i < N - 1) os << " ";
		}
		return os;
	}
}