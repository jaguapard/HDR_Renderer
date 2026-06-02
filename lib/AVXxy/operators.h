#pragma once
#include "SIMD_Mask.h"
#include "SIMD_Vector.h"
#include "funcs.h"
namespace AVXXY_NAMESPACE
{
	template <typename S, size_t N> __forceinline SIMD_Vector<S, N> operator+(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b) { return add(a, b); };
	template <typename S, size_t N> __forceinline SIMD_Vector<S, N> operator-(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b) { return sub(a, b); };
	template <typename S, size_t N> __forceinline SIMD_Vector<S, N> operator*(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b) { return mul(a, b); };
	template <typename S, size_t N> __forceinline SIMD_Vector<S, N> operator/(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b) { return div(a, b); };
	template <typename S, size_t N> __forceinline SIMD_Vector<S, N> operator&(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b) { return logic_and(a, b); };
	template <typename S, size_t N> __forceinline SIMD_Vector<S, N> operator|(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b) { return logic_or(a, b); };
	template <typename S, size_t N> __forceinline SIMD_Vector<S, N> operator^(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b) { return logic_xor(a, b); };
	template <typename S, size_t N> __forceinline SIMD_Vector<S, N> operator~(const SIMD_Vector<S, N>& a) { return logic_not(a); };
	template <typename S, size_t N, typename I> __forceinline SIMD_Vector<S, N> operator<<(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& b) { return shift_left(a, b); };
	template <typename S, size_t N, typename I> __forceinline SIMD_Vector<S, N> operator>>(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& b) { return shift_right(a, b); };

	template <typename S, size_t N> __forceinline SIMD_Vector<S, N>& operator+=(SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b) { a = a + b; return a; }
	template <typename S, size_t N> __forceinline SIMD_Vector<S, N>& operator-=(SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b) { a = a - b; return a; }
	template <typename S, size_t N> __forceinline SIMD_Vector<S, N>& operator*=(SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b) { a = a * b; return a; }
	template <typename S, size_t N> __forceinline SIMD_Vector<S, N>& operator/=(SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b) { a = a / b; return a; }
	template <typename S, size_t N> __forceinline SIMD_Vector<S, N>& operator&=(SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b) { a = a & b; return a; }
	template <typename S, size_t N> __forceinline SIMD_Vector<S, N>& operator|=(SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b) { a = a | b; return a; }
	template <typename S, size_t N> __forceinline SIMD_Vector<S, N>& operator^=(SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b) { a = a ^ b; return a; }
	template <typename S, size_t N, typename I> __forceinline SIMD_Vector<S, N>& operator<<=(SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& b) { a = a << b; return a; }
	template <typename S, size_t N, typename I> __forceinline SIMD_Vector<S, N>& operator>>=(SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& b) { a = a >> b; return a; }

	template <typename S, size_t N> __forceinline typename SIMD_Vector<S, N>::MaskType operator==(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b) { return cmp_equal(a, b); }
	template <typename S, size_t N> __forceinline typename SIMD_Vector<S, N>::MaskType operator!=(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b) { return cmp_not_equal(a, b); }
	template <typename S, size_t N> __forceinline typename SIMD_Vector<S, N>::MaskType operator>(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b) { return cmp_greater(a, b); }
	template <typename S, size_t N> __forceinline typename SIMD_Vector<S, N>::MaskType operator<(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b) { return cmp_less(a, b); }
	template <typename S, size_t N> __forceinline typename SIMD_Vector<S, N>::MaskType operator<=(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b) { return cmp_less_or_equal(a, b); }
	template <typename S, size_t N> __forceinline typename SIMD_Vector<S, N>::MaskType operator>=(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b) { return cmp_greater_or_equal(a, b); }
}