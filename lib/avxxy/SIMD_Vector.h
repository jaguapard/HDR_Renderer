#pragma once

#include <algorithm>
#include <array>
#include <iostream>
#include <cstring>
#include "meta/meta.h"
#include "meta/type_factories.h"
#include "SIMD_Mask.h"

namespace AVXXY_NAMESPACE
{
	/**
	* Vector of N values of type S.
	* Although vectors don't have a hard size cap, the limitations grow more severe as vector size grows:
	* masked operations are unavailable for N > 64, 
	* other operation-specific limitations apply for large vectors.
	* Stack space can become a concern with large vectors
	* Performance scales non-linearly with vector size for some operations
	* @tparam S element type of this vector. Only these types are supported: signed and unsigned integers: 8, 16, 32 and 64-bit wide, float, double, FP16 (via fp16_t) and BF16 (via bf16_t)
	* @tparam N Lane count of this vector. Must be 1 or any power of 2.
	*/
	template <typename S, size_t N>
		requires meta::IsValid_SIMD_Vector<S, N>
	class alignas(std::min<uint32_t>(64, sizeof(S)* N)) SIMD_Vector
	{
	private:
		union {
			std::array<S, N> arr;
			//struct { std::array<S, N / 2> half_lo, half_hi; };
		};
	public:
		template<typename FriendS, size_t FriendN> requires meta::IsValid_SIMD_Vector<FriendS, FriendN>
		friend class SIMD_Vector;

		static inline constexpr bool IsSimdVector = true;
		static inline constexpr size_t LaneCount = N;

		using IntrinsicT = meta::typed_intrinsic_storage_t<S, N>;
		using ScalarT = S;

		SIMD_Vector() {};

		//Returns a constant reference to i'th scalar element (lane) of this vector. Does not perform bounds checks.
		const S& operator[](size_t i) const { return arr[i]; }

		//Returns a non-constant reference to i'th scalar element (lane) of this vector. Does not perform bounds checks.
		//The returned reference can be used to modify vector's values
		S& operator[](size_t i) { return arr[i]; }

		//Constructs this vector from other vector. If vector scalar types mismatch, the input is converted to this vector's scalar type before assignment
		template<typename T>
		SIMD_Vector(const SIMD_Vector<T, N>& other) { *this = vcvt<S>(other); }

		//Constructs this vector by copying input data into it's own storage
		SIMD_Vector(const std::array<S, N>& data) { arr = data; }

		//Constructs vector from it's intrinsic type. The intrinsic vector type must be of the same size class as constructed vector:
		//Vectors less than 17 bytes can be constructed from 128 bit intrinsic types.
		//Vectors between 17 and 32 bytes can be constructed from 256 bit intrinsic types.
		//Vectors between 33 and 64 bytes can be constructed from 512 bit intrinsic types.
		//Integral intrinsic vectors can be used to construct any integral SIMD_Vector of same size class
		//Floating point vectors require the scalar type of intrinsic vector and SIMD_Vector to match
		//If SIMD_Vector and intrinsic vector sizes mismatch, only the lower sizeof(SIMD_Vector) bytes from intrinsic vector are copied to the constructed SIMD_Vector
		SIMD_Vector(const IntrinsicT& intrinsicVec)
		{
			if constexpr (sizeof(arr) == sizeof(intrinsicVec)) arr = std::bit_cast<decltype(arr)>(intrinsicVec);
			else memcpy(arr.data(), &intrinsicVec, std::min(sizeof(arr), sizeof(intrinsicVec)));
		}

		//Converts the SIMD_Vector to it's intrinsic vector type.
		//If intrinsic vector is larger than SIMD_Vector, upper bytes of returned value are undefined
		operator IntrinsicT() const
		{
			IntrinsicT ret;
			if constexpr (sizeof(IntrinsicT) == sizeof(arr)) return std::bit_cast<IntrinsicT>(arr);
			else
			{
				memcpy(&ret, arr.data(), std::min(sizeof(arr), sizeof(ret)));
				return ret;
			}
		}

		//Constructs vector from halves
		template<size_t N2>
			requires (N2 * 2 == N)
		SIMD_Vector(const SIMD_Vector<S, N2>& lo, const SIMD_Vector<S, N2>& hi)
		{
			static_assert(N % 2 == 0);
			memcpy(arr.data(), lo.arr.data(), sizeof(lo.arr));
			memcpy(arr.data() + N / 2, hi.arr.data(), sizeof(hi.arr));
		}


		//Broadcasts a scalar value to all lanes of a vector. The input value is converted to vector's scalar type before broadcasting
		template<typename T> requires meta::IsScalarType<T>
		SIMD_Vector(T s) { for (size_t i = 0; i < N; ++i) (*this)[i] = s; }

		//Returns vector filled with sequential values (value == lane index, like 0, 1, 2, ..., N-1)
		static SIMD_Vector<S, N> iota()
		{
			SIMD_Vector<S, N> ret;
			for (size_t i = 0; i < N; ++i) ret[i] = i;
			return ret;
		}
		//Copies and returns lower half of this vector
		auto lo() const requires (N >= 2)
		{
			SIMD_Vector<S, N / 2> ret;
			memcpy(ret.arr.data(), arr.data(), sizeof(ret.arr));
			return ret;
		}
		//Copies and returns upper half of this vector
		auto hi() const requires (N >= 2)
		{
			SIMD_Vector<S, N / 2> ret;
			memcpy(ret.arr.data(), arr.data() + N / 2, sizeof(ret.arr));
			return ret;
		}

		//Constructs vector by reinterperting the value of inp as vector of wanted type
		//If input value is larger than returned vector, the input's upper bits are discarded
		//If input value is smaller than returned vector, upper bits of returned vector values are underfined
		template<typename T>
		static SIMD_Vector<S, N> fromBits(const T& inp)
		{
			SIMD_Vector<S, N> ret;
			static_assert(sizeof(ret.arr) == sizeof(ret));
			if constexpr (sizeof(ret.arr) == sizeof(inp)) ret.arr = std::bit_cast<decltype(ret.arr)>(inp);
			else memcpy(ret.arr.data(), &inp, std::min(sizeof(inp), sizeof(ret)));
			return ret;
		}







		//@note do NOT change these to variadic templates. We want users to see that the type has ctor from N scalars, not guess while typing. So something similar to _mm*_setr_* instead of printf
		// Constructs this vector from 2 scalar values. 
		// Each value is converted to vector's scalar type before assignment. 
		// This constructor is only available for 2-element vectors
		template <typename T0, typename T1>
			requires (N == 2 && meta::AllAreScalarTypes<T0, T1>)
		__forceinline SIMD_Vector(T0 s0, T1 s1)
		{
			(*this)[0] = s0; (*this)[1] = s1;
		}

		// Constructs this vector from 4 scalar values. 
		// Each value is converted to vector's scalar type before assignment. 
		// This constructor is only available for 4-element vectors
		template <typename T0, typename T1, typename T2, typename T3>
			requires (N == 4 && meta::AllAreScalarTypes<T0, T1, T2, T3>)
		__forceinline SIMD_Vector(T0 s0, T1 s1, T2 s2, T3 s3)
		{
			(*this)[0] = s0; (*this)[1] = s1; (*this)[2] = s2; (*this)[3] = s3;
		}

		// Constructs this vector from 8 scalar values. 
		// Each value is converted to vector's scalar type before assignment. 
		// This constructor is only available for 8-element vectors.
		template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
			requires (N == 8 && meta::AllAreScalarTypes<T0, T1, T2, T3, T4, T5, T6, T7>)
		__forceinline SIMD_Vector(T0 s0, T1 s1, T2 s2, T3 s3, T4 s4, T5 s5, T6 s6, T7 s7)
		{
			(*this)[0] = s0; (*this)[1] = s1; (*this)[2] = s2; (*this)[3] = s3;
			(*this)[4] = s4; (*this)[5] = s5; (*this)[6] = s6; (*this)[7] = s7;
		}

		// Constructs this vector from 16 scalar values. 
		// Each value is converted to vector's scalar type before assignment. 
		// This constructor is only available for vectors with N == 16.
		template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8,
			typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15>
			requires (N == 16 && meta::AllAreScalarTypes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15>)
		__forceinline SIMD_Vector(T0 s0, T1 s1, T2 s2, T3 s3, T4 s4, T5 s5, T6 s6, T7 s7, T8 s8, T9 s9, T10 s10, T11 s11, T12 s12, T13 s13, T14 s14, T15 s15)
		{
			(*this)[0] = s0; (*this)[1] = s1; (*this)[2] = s2; (*this)[3] = s3;
			(*this)[4] = s4; (*this)[5] = s5; (*this)[6] = s6; (*this)[7] = s7;
			(*this)[8] = s8; (*this)[9] = s9; (*this)[10] = s10; (*this)[11] = s11;
			(*this)[12] = s12; (*this)[13] = s13; (*this)[14] = s14; (*this)[15] = s15;
		}

		// Constructs this vector from 32 scalar values. 
		// Each value is converted to vector's scalar type before assignment. 
		// This constructor is only available for vectors with N == 32.
		template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15, typename T16, typename T17, typename T18, typename T19, typename T20, typename T21, typename T22, typename T23, typename T24, typename T25, typename T26, typename T27, typename T28, typename T29, typename T30, typename T31>
			requires (N == 32 && meta::AllAreScalarTypes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31>)
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
		// This constructor is only available for vectors with N == 64.
		template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15, typename T16, typename T17, typename T18, typename T19, typename T20, typename T21, typename T22, typename T23, typename T24, typename T25, typename T26, typename T27, typename T28, typename T29, typename T30, typename T31, typename T32, typename T33, typename T34, typename T35, typename T36, typename T37, typename T38, typename T39, typename T40, typename T41, typename T42, typename T43, typename T44, typename T45, typename T46, typename T47, typename T48, typename T49, typename T50, typename T51, typename T52, typename T53, typename T54, typename T55, typename T56, typename T57, typename T58, typename T59, typename T60, typename T61, typename T62, typename T63>
			requires (N == 64 && meta::AllAreScalarTypes<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59, T60, T61, T62, T63>)
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

		//Generic constructor. Converts all input arguments to vector's scalar type and assigns them in left-to-right order
		template<typename... Ts> requires (N != 1 && N > 64 && meta::AllAreScalarTypes<Ts...> && sizeof...(Ts) == N)
			SIMD_Vector(Ts... s)
		{
			size_t i = 0;
			auto append = [&](auto x) {
				(*this)[i++] = x;
				};
			(append(s), ...);
		}
	};

	template<typename S, size_t N>
	std::ostream& operator<<(std::ostream& os, const SIMD_Vector<S, N>& a)
	{
		for (size_t i = 0; i < N; ++i)
		{
			if constexpr (meta::any_i8<S>) os << int(a[i]);
			else os << a[i];
			if (i < N - 1) os << " ";
		}
		return os;
	}
}