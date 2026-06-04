#pragma once
#include "namespace.h"
#include "infer.h"
#include <array>

namespace AVXXY_NAMESPACE
{
	template<size_t N>
	struct SIMD_Mask;

	template<size_t N>
	requires (N > 0 && N <= 64)
	struct SIMD_Mask<N>
	{
		static inline constexpr size_t BitCount = N;
		using UintType = bits_to_uint_t<N>::type;
		static inline constexpr UintType AllOnes = (BitCount == sizeof(UintType) * 8) ? ~UintType(0) : ((UintType(1) << BitCount) - 1);
		static inline constexpr size_t ByteSize = sizeof(UintType);

		SIMD_Mask() {};
		SIMD_Mask(UintType x) { bits = x & AllOnes; };
		__forceinline operator UintType() const { return bits & AllOnes; //clear garbage bits. Else, it could fail for BitCount == 1 for instance, since naive usage of it will touch out of bounds data for tiny vectors
		}

		template <typename T> __forceinline std::array<T, BitCount> explode(const T& trueValue = T(1), const T& falseValue = T(0)) const
		{
			std::array<T, N> ret;
			this->explode(ret.data(), trueValue, falseValue);
			return ret;
		}
		template <typename T> __forceinline void explode(T* ret, const T& trueValue = T(1), const T& falseValue = T(0)) const
		{
			for (int i = 0; i < N; ++i) ret[i] = (*this)[i] ? trueValue : falseValue;
		}

		__forceinline SIMD_Mask<N / 2> lo() const
		{
			return bits;
		}
		__forceinline SIMD_Mask<N / 2> hi() const
		{
			return bits >> (N / 2);
		}
		bool operator[](size_t i) const;

		SIMD_Mask<N>& operator&=(const SIMD_Mask<N>& other)
		{
			*this = *this & other;
			return *this;
		}
		SIMD_Mask<N>& operator|=(const SIMD_Mask<N>& other)
		{
			*this = *this | other;
			return *this;
		}
		SIMD_Mask<N>& operator^=(const SIMD_Mask<N>& other)
		{
			*this = *this ^ other;
			return *this;
		}

		void setBit(size_t i, bool bit = true)
		{
			bits &= ~(UintType(1) << i);
			bits |= UintType(bit) << i;
			bits &= AllOnes; //avoid putting garbage int upper bits
		}
	private:
		UintType bits;
	};

	//TODO: add clearing of upper garbage, lower_half, upper_half
	template<size_t N>
	requires (N > 64)
	struct SIMD_Mask<N>
	{
		static inline constexpr size_t BitCount = N;
		using UintType = std::array<uint64_t, aligned_size(N, 64)/64>;
		static inline constexpr size_t ByteSize = sizeof(UintType);
		static inline constexpr UintType AllOnes = []() {
			UintType ret;
			for (size_t i = 0; i < N; i += 64) ret[i / 64] = ~uint64_t(0);
			return ret;
			}();

		bool operator[](size_t i) const
		{
			return bits[i / 64] & (1ull << (i % 64));
		}

		template <typename T> __forceinline void explode(T* ret, const T& trueValue = T(1), const T& falseValue = T(0)) const
		{
			for (int i = 0; i < N; ++i) ret[i] = (*this)[i] ? trueValue : falseValue;
		}
		template <typename T> __forceinline std::array<T, BitCount> explode(const T& trueValue = T(1), const T& falseValue = T(0)) const
		{
			std::array<T, N> ret;
			this->explode(ret.data(), trueValue, falseValue);
			return ret;
		}
	private:
		UintType bits;
	};
}