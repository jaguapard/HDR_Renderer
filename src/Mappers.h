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
	void wrapInts(int32x16& x, int32x16& y) const;

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
	static int32x16 wrapIntWithRcp(int32x16 x, uint64_t rcp, uint32_t v);

	//wraps values to range 0 and 1, 0 <= u,v < 1
	static std::pair<float, float> wrapUV(float u, float v);
	static std::pair<float32x8, float32x8> wrapUV(float32x8 u, float32x8 v);
	static std::pair<float32x16, float32x16> wrapUV(float32x16 u, float32x16 v);

private:
	Params params;
	void wrapIntsPositive(int32x16& x, int32x16& y) const;
	static int32x16 wrapPositiveIntWithRcp(int32x16 x, uint64_t rcp, uint32_t v);
};