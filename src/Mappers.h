#pragma once
#include <stdint.h>
#include "Vec.h"

class WrappingMapper
{
public:
	WrappingMapper() = default;
	WrappingMapper(uint32_t w, uint32_t h);

	struct Params
	{
		uint32_t w, h;
		float fw, fh;
		uint64_t u64_rcpW, u64_rcpH;
		double f64_rcpW, f64_rcpH, f64_w, f64_h;
	};
	__forceinline const Params& getParams() const
	{
		return params;
	}
	//Wraps x around w and y around h in-place using this instance's values.
	//It is guaranteed than returned x >= 0 && x < w && y >= 0 && y < h
	//Negative values are not mirrored, x == -1 wraps to w-1, not to 1, x == -w wraps to 0, x == -w-1 wraps to 1, etc.
	template<size_t N>
	void wrapInts(SIMD_Vector<int32_t, N>& x, SIMD_Vector<int32_t, N>& y) const
	{
		x = this->wrapIntWithRcp(x, this->params.u64_rcpW, this->params.w);
		y = this->wrapIntWithRcp(y, this->params.u64_rcpH, this->params.h);
	}

	template<typename T>
	__forceinline std::pair<T, T> UV_to_XY(const T& u, const T& v) const
	{
		auto [wu, wv] = wrapUV(u, v);
		return { wu * params.fw, wv * params.fh };
	}
	/**
	@param x: value to be wrapped around
	@param rcp: 1<<32 / v
	@param v: value that x will be wrapped around
	*/
	template<size_t N>
	static SIMD_Vector<int32_t, N> wrapIntWithRcp(const SIMD_Vector<int32_t, N>& x, uint64_t rcp, uint32_t v)
	{
		auto ax = abs(x);
		SIMD_Vector<int32_t, N> rem = wrapPositiveIntWithRcp(vcvt<uint32_t>(ax), rcp, v);
		//TODO: some of these steps may be too cautious and can be removed?
		SIMD_Vector<int32_t, N> vbcst = v;
		rem = mask_mov(vbcst - rem, x >= 0, rem);
		rem = mask_mov(rem, rem >= v, rem - vbcst);
		rem = mask_mov(rem, rem < 0, rem + vbcst);
		return rem;
	}

	//wraps values to range 0 and 1, 0 <= u,v < 1
	template<typename T>
	static std::pair<T, T> wrapUV(T u, T v)
	{
		using namespace std; //cheeky, but I don't want to make another version for float :)
		u -= floor(u); //doing floor subtraction once sometimes returns 1. Doing it twice guarantees 0 <= u < 1 for all non-nan non-inf values
		u -= floor(u);
		v -= floor(v);
		v -= floor(v);
		return { u,v };
	}

private:
	Params params;
	template<size_t N>
	static SIMD_Vector<uint32_t, N> wrapPositiveIntWithRcp(const SIMD_Vector<uint32_t, N>& x, uint64_t rcp, uint32_t v)
	{
		//for (int i = 0; i < 16; ++i) assert(x[i] >= 0);
		SIMD_Vector<uint64_t, N> x64 = x;
		SIMD_Vector<uint64_t, N> div = (x64 * rcp) >> 32;
		SIMD_Vector<uint64_t, N> rem = x64 - (div * v);
		//return permx2(vcast<u32x16>(rem.lo()), vcast<u32x16>(rem.hi()), u32x16(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30));
		return rem;
	}
};