#pragma once
#include "SIMD_Mask.h"
#include "SIMD_Vector.h"
#include "funcs.h"
namespace AVXXY_NAMESPACE
{
	template <typename T, typename S, size_t N> __forceinline SIMD_Vector<S, N> operator+(const SIMD_Vector<S, N>& a, const T& b) { return add(a, SIMD_Vector<S, N>(b)); };
	template <typename T, typename S, size_t N> __forceinline SIMD_Vector<S, N> operator-(const SIMD_Vector<S, N>& a, const T& b) { return sub(a, SIMD_Vector<S, N>(b)); };
	template <typename T, typename S, size_t N> __forceinline SIMD_Vector<S, N> operator*(const SIMD_Vector<S, N>& a, const T& b) { return mul(a, SIMD_Vector<S, N>(b)); };
	template <typename T, typename S, size_t N> __forceinline SIMD_Vector<S, N> operator/(const SIMD_Vector<S, N>& a, const T& b) { return div(a, SIMD_Vector<S, N>(b)); };
	template <typename T, typename S, size_t N> __forceinline SIMD_Vector<S, N> operator&(const SIMD_Vector<S, N>& a, const T& b) { return logic_and(a, SIMD_Vector<S, N>(b)); };
	template <typename T, typename S, size_t N> __forceinline SIMD_Vector<S, N> operator|(const SIMD_Vector<S, N>& a, const T& b) { return logic_or(a, SIMD_Vector<S, N>(b)); };
	template <typename T, typename S, size_t N> __forceinline SIMD_Vector<S, N> operator^(const SIMD_Vector<S, N>& a, const T& b) { return logic_xor(a, SIMD_Vector<S, N>(b)); };
	template <typename T, typename S, size_t N> __forceinline SIMD_Vector<S, N> operator~(const SIMD_Vector<S, N>& a) { return logic_not(a); };
	template <typename S, size_t N, typename T> __forceinline SIMD_Vector<S, N> operator<<(const SIMD_Vector<S, N>& a, const T& b) { return shift_left(a, SIMD_Vector<S, N>(b)); };
	template <typename S, size_t N, typename T> __forceinline SIMD_Vector<S, N> operator>>(const SIMD_Vector<S, N>& a, const T& b) { return shift_right(a, SIMD_Vector<S, N>(b)); };

	template <typename T, typename S, size_t N> __forceinline SIMD_Vector<S, N>& operator+=(SIMD_Vector<S, N>& a, const T& b) { a = a + b; return a; }
	template <typename T, typename S, size_t N> __forceinline SIMD_Vector<S, N>& operator-=(SIMD_Vector<S, N>& a, const T& b) { a = a - b; return a; }
	template <typename T, typename S, size_t N> __forceinline SIMD_Vector<S, N>& operator*=(SIMD_Vector<S, N>& a, const T& b) { a = a * b; return a; }
	template <typename T, typename S, size_t N> __forceinline SIMD_Vector<S, N>& operator/=(SIMD_Vector<S, N>& a, const T& b) { a = a / b; return a; }
	template <typename T, typename S, size_t N> __forceinline SIMD_Vector<S, N>& operator&=(SIMD_Vector<S, N>& a, const T& b) { a = a & b; return a; }
	template <typename T, typename S, size_t N> __forceinline SIMD_Vector<S, N>& operator|=(SIMD_Vector<S, N>& a, const T& b) { a = a | b; return a; }
	template <typename T, typename S, size_t N> __forceinline SIMD_Vector<S, N>& operator^=(SIMD_Vector<S, N>& a, const T& b) { a = a ^ b; return a; }
	template <typename S, size_t N, typename T> __forceinline SIMD_Vector<S, N>& operator<<=(SIMD_Vector<S, N>& a, const T& b) { a = a << b; return a; }
	template <typename S, size_t N, typename T> __forceinline SIMD_Vector<S, N>& operator>>=(SIMD_Vector<S, N>& a, const T& b) { a = a >> b; return a; }

	template <typename T, typename S, size_t N> __forceinline mask_t<S, N> operator==(const SIMD_Vector<S, N>& a, const T& b) { return cmp_equal(a, SIMD_Vector<S, N>(b)); }
	template <typename T, typename S, size_t N> __forceinline mask_t<S, N> operator!=(const SIMD_Vector<S, N>& a, const T& b) { return cmp_not_equal(a, SIMD_Vector<S, N>(b)); }
	template <typename T, typename S, size_t N> __forceinline mask_t<S, N> operator>(const SIMD_Vector<S, N>& a, const T& b) { return cmp_greater(a, SIMD_Vector<S, N>(b)); }
	template <typename T, typename S, size_t N> __forceinline mask_t<S, N> operator<(const SIMD_Vector<S, N>& a, const T& b) { return cmp_less(a, SIMD_Vector<S, N>(b)); }
	template <typename T, typename S, size_t N> __forceinline mask_t<S, N> operator<=(const SIMD_Vector<S, N>& a, const T& b) { return cmp_less_or_equal(a, SIMD_Vector<S, N>(b)); }
	template <typename T, typename S, size_t N> __forceinline mask_t<S, N> operator>=(const SIMD_Vector<S, N>& a, const T& b) { return cmp_greater_or_equal(a, SIMD_Vector<S, N>(b)); }

	template<typename S, size_t N> __forceinline SIMD_Vector<S, N> operator-(const SIMD_Vector<S, N>& a) { return sub(SIMD_Vector<S, N>(S(0)), a); }
}