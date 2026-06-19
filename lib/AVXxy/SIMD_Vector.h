#pragma once
#include "namespace.h"
#include "SIMD_Mask.h"
#include "infer.h"
#include "utils.h"

namespace AVXXY_NAMESPACE
{
	template <typename T, typename... Ts>
	inline constexpr bool is_any_of_v =
		(std::is_same_v<T, Ts> || ...);

	using namespace utils;
	template<typename T> concept IsScalarType = is_any_of_v<T, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, float, double>;
	template<typename T> concept IsIntrinsicVector = is_any_of_v<T, __m128i, __m128, __m128d, __m256i, __m256, __m256d, __m512i, __m512, __m512d>;
	template<typename _S, size_t _N> concept IsValid_SIMD_Vector = _N <= 64 && isPowerOf2(_N) && IsScalarType<_S>; //for now, bigger than 64 lanes vectors are not supported (mainly due to mask type not being ready for it)

	template<typename Vec, typename IntrinVec>
	concept ConversionToNativeVectorLegal =
		(inRange(sizeof(Vec), 0, 16) && std::is_same_v<IntrinVec, typename reg128<typename Vec::ScalarType>::type>) ||
		(inRange(sizeof(Vec), 17, 32) && std::is_same_v<IntrinVec, typename reg256<typename Vec::ScalarType>::type>) ||
		(inRange(sizeof(Vec), 33, 64) && std::is_same_v<IntrinVec, typename reg512<typename Vec::ScalarType>::type>);


	template<typename _S, size_t _N>
		requires IsValid_SIMD_Vector<_S, _N>
	struct alignas(std::min(64ull, sizeof(_S)* _N)) SIMD_Vector
	{
		using ScalarType = _S;
		using IntScalarType = typename bits_to_int_t<sizeof(_S) * 8>::type;
		static inline constexpr size_t LaneCount = _N;
		using MaskType = SIMD_Mask<LaneCount>;
		using Self = SIMD_Vector<ScalarType, LaneCount>;

		static inline constexpr size_t ByteSize = sizeof(ScalarType) * LaneCount;
		static inline constexpr bool IsSimdVector = true;

		union
		{
			std::array<_S, _N> arr;
			struct { typename std::conditional_t<_N <= 2, _S, SIMD_Vector<_S, _N / 2>> lo, hi; };
		};
		__forceinline SIMD_Vector() {};

		//Broadcasts a scalar value to all lanes of vector. The input value is converted to vector's intrinsic type before broadcasting
		template<typename T> requires IsScalarType<T>
		__forceinline SIMD_Vector(const T& s) { for (size_t i = 0; i < LaneCount; ++i) (*this)[i] = s; }

		//Constructs vector from it's intrinsic type. The intrinsic vector type is in the same size class as this vector:
		//Vectors <= 16 bytes: 128 bit intrinsic types.
		//Vectors between 16 and 32 bytes: 256 bit intrinsic types.
		//Vectors between 32 and 64 bytes: 512 bit intrinsic types.
		//Integral intrinsic vectors can be used to construct any integral SIMD_Vector of same size class
		//Floating point vectors require the scalar type of intrinsic vector and SIMD_Vector to be the same
		//If SIMD_Vector and intrinsic vector sizes mismatch, only the lowest bits of intrinsic vector are copied to the constructed SIMD_Vector
		template <typename T>
		__forceinline SIMD_Vector(const T& intrinsicVec) requires(IsIntrinsicVector<T>&& ConversionToNativeVectorLegal<Self, T>)
		{
			memcpy(this, &intrinsicVec, std::min(sizeof(Self), sizeof(T)));
		}

		//Represents vector as it's intrinsic type.  The intrinsic vector type is in the same size class as this vector:
		//Vectors <= 16 bytes: 128 bit intrinsic types.
		//Vectors between 16 and 32 bytes: 256 bit intrinsic types.
		//Vectors between 32 and 64 bytes: 512 bit intrinsic types.
		//Integral SIMD_Vectors convert to integral intrinsic vectors
		//Floating point SIMD_Vectors convert to intrinsic vectors of same scalar type
		//If SIMD_Vector and intrinsic vector sizes mismatch, the upper bits of returned vector are undefined
		template <typename T>
		__forceinline operator T() const requires(IsIntrinsicVector<T>&& ConversionToNativeVectorLegal<Self, T>)
		{
			T ret;
			memcpy(&ret, this, std::min(sizeof(T), sizeof(Self)));
			return ret;
		}

		template<typename T>
		__forceinline SIMD_Vector(const SIMD_Vector<T, LaneCount>& other)
		{
			*this = vec_cvt<ScalarType, LaneCount>(other);
		}
		static __forceinline Self sequence()
		{
			Self ret;
			for (size_t i = 0; i < LaneCount; ++i) ret[i] = i;
			return ret;
		}

		//Returns true if there's at least one element with it's most significant bit set, or false otherwise.
		/*operator bool() const
		{
			return reinterpret<IntScalarType>(*this) < 0;
		}*/

		//Returns a reference to scalar element in lane i
		__forceinline const ScalarType& operator[](size_t i) const { return this->arr[i]; };
		//Returns a reference to scalar element in lane i
		__forceinline ScalarType& operator[](size_t i) { return this->arr[i]; };

		/*
		__forceinline static Self load(const void* p, const MaskType& mask = MaskType::AllOnes, const Self& src = 0) { return AVXXY_NAMESPACE::load<ScalarType, LaneCount>(p, mask, src); }

		template<size_t Scale = sizeof(ScalarType), typename I>
		__forceinline static Self gather(const void* p, const SIMD_Vector<I, LaneCount>& ind, const MaskType& mask = MaskType::AllOnes, const Self& src = 0) {
			return AVXXY_NAMESPACE::gather<Scale>(p, ind, mask, src);
		}

		template<size_t Scale = sizeof(ScalarType), typename I>
		__forceinline void scatter(const void* p, const SIMD_Vector<I, LaneCount>& ind, const MaskType& mask = MaskType::AllOnes)
		{
			return AVXXY_NAMESPACE::scatter<Scale>(*this, p, ind, mask);
		}*/

		//do NOT change these to variadic templates. We want users to see that the type has ctor from N scalars, not guess while typing. So something similar to _mm*_setr_* instead of printf
		template <typename T0, typename T1>
			requires (LaneCount == 2 && IsScalarType<T0> && IsScalarType<T1>)
		__forceinline SIMD_Vector(T0 s0, T1 s1)
		{
			(*this)[0] = s0; (*this)[1] = s1;
		}
		template <typename T0, typename T1, typename T2, typename T3>
			requires (LaneCount == 4 && IsScalarType<T0> && IsScalarType<T1> && IsScalarType<T2> && IsScalarType<T3>)
		__forceinline SIMD_Vector(T0 s0, T1 s1, T2 s2, T3 s3)
		{
			(*this)[0] = s0; (*this)[1] = s1; (*this)[2] = s2; (*this)[3] = s3;
		}
		template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
			requires (LaneCount == 8 && IsScalarType<T0> && IsScalarType<T1> && IsScalarType<T2> && IsScalarType<T3> && IsScalarType<T4> && IsScalarType<T5> && IsScalarType<T6> && IsScalarType<T7>)
		__forceinline SIMD_Vector(T0 s0, T1 s1, T2 s2, T3 s3, T4 s4, T5 s5, T6 s6, T7 s7)
		{
			(*this)[0] = s0; (*this)[1] = s1; (*this)[2] = s2; (*this)[3] = s3;
			(*this)[4] = s4; (*this)[5] = s5; (*this)[6] = s6; (*this)[7] = s7;
		}
		template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8,
			typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15>
			requires (LaneCount == 16 && IsScalarType<T0> && IsScalarType<T1> && IsScalarType<T2> && IsScalarType<T3> && IsScalarType<T4> && IsScalarType<T5> && IsScalarType<T6> && IsScalarType<T7> && IsScalarType<T8> && IsScalarType<T9> && IsScalarType<T10> && IsScalarType<T11> && IsScalarType<T12> && IsScalarType<T13> && IsScalarType<T14> && IsScalarType<T15>)
		__forceinline SIMD_Vector(T0 s0, T1 s1, T2 s2, T3 s3, T4 s4, T5 s5, T6 s6, T7 s7, T8 s8, T9 s9, T10 s10, T11 s11, T12 s12, T13 s13, T14 s14, T15 s15)
		{
			(*this)[0] = s0; (*this)[1] = s1; (*this)[2] = s2; (*this)[3] = s3;
			(*this)[4] = s4; (*this)[5] = s5; (*this)[6] = s6; (*this)[7] = s7;
			(*this)[8] = s8; (*this)[9] = s9; (*this)[10] = s10; (*this)[11] = s11;
			(*this)[12] = s12; (*this)[13] = s13; (*this)[14] = s14; (*this)[15] = s15;
		}
		template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15, typename T16, typename T17, typename T18, typename T19, typename T20, typename T21, typename T22, typename T23, typename T24, typename T25, typename T26, typename T27, typename T28, typename T29, typename T30, typename T31>
			requires (LaneCount == 32 && IsScalarType<T0> && IsScalarType<T1> && IsScalarType<T2> && IsScalarType<T3> && IsScalarType<T4> && IsScalarType<T5> && IsScalarType<T6> && IsScalarType<T7> && IsScalarType<T8> && IsScalarType<T9> && IsScalarType<T10> && IsScalarType<T11> && IsScalarType<T12> && IsScalarType<T13> && IsScalarType<T14> && IsScalarType<T15> && IsScalarType<T16> && IsScalarType<T17> && IsScalarType<T18> && IsScalarType<T19> && IsScalarType<T20> && IsScalarType<T21> && IsScalarType<T22> && IsScalarType<T23> && IsScalarType<T24> && IsScalarType<T25> && IsScalarType<T26> && IsScalarType<T27> && IsScalarType<T28> && IsScalarType<T29> && IsScalarType<T30> && IsScalarType<T31>)
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
		template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15, typename T16, typename T17, typename T18, typename T19, typename T20, typename T21, typename T22, typename T23, typename T24, typename T25, typename T26, typename T27, typename T28, typename T29, typename T30, typename T31, typename T32, typename T33, typename T34, typename T35, typename T36, typename T37, typename T38, typename T39, typename T40, typename T41, typename T42, typename T43, typename T44, typename T45, typename T46, typename T47, typename T48, typename T49, typename T50, typename T51, typename T52, typename T53, typename T54, typename T55, typename T56, typename T57, typename T58, typename T59, typename T60, typename T61, typename T62, typename T63>
			requires (LaneCount == 64 && IsScalarType<T0> && IsScalarType<T1> && IsScalarType<T2> && IsScalarType<T3> && IsScalarType<T4> && IsScalarType<T5> && IsScalarType<T6> && IsScalarType<T7> && IsScalarType<T8> && IsScalarType<T9> && IsScalarType<T10> && IsScalarType<T11> && IsScalarType<T12> && IsScalarType<T13> && IsScalarType<T14> && IsScalarType<T15> && IsScalarType<T16> && IsScalarType<T17> && IsScalarType<T18> && IsScalarType<T19> && IsScalarType<T20> && IsScalarType<T21> && IsScalarType<T22> && IsScalarType<T23> && IsScalarType<T24> && IsScalarType<T25> && IsScalarType<T26> && IsScalarType<T27> && IsScalarType<T28> && IsScalarType<T29> && IsScalarType<T30> && IsScalarType<T31> && IsScalarType<T32> && IsScalarType<T33> && IsScalarType<T34> && IsScalarType<T35> && IsScalarType<T36> && IsScalarType<T37> && IsScalarType<T38> && IsScalarType<T39> && IsScalarType<T40> && IsScalarType<T41> && IsScalarType<T42> && IsScalarType<T43> && IsScalarType<T44> && IsScalarType<T45> && IsScalarType<T46> && IsScalarType<T47> && IsScalarType<T48> && IsScalarType<T49> && IsScalarType<T50> && IsScalarType<T51> && IsScalarType<T52> && IsScalarType<T53> && IsScalarType<T54> && IsScalarType<T55> && IsScalarType<T56> && IsScalarType<T57> && IsScalarType<T58> && IsScalarType<T59> && IsScalarType<T60> && IsScalarType<T61> && IsScalarType<T62> && IsScalarType<T63>)
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
	};
}