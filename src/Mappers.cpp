#include "Mappers.h"

WrappingMapper::WrappingMapper(uint32_t w, uint32_t h)
{
	this->params.w = w;
	this->params.h = h;
	this->params.fw = w;
	this->params.fh = h;
	this->params.u64_rcpW = uint64_t(1ull << 32) / w;
	this->params.u64_rcpH = uint64_t(1ull << 32) / h;
	this->params.f64_rcpW = 1.0 / w;
	this->params.f64_rcpH = 1.0 / h;
	this->params.f64_w = w;
	this->params.f64_h = h;
}

void WrappingMapper::wrapInts(int32x16& x, int32x16& y) const
{
	x = this->wrapIntWithRcp(x, this->params.u64_rcpW, this->params.w);
	y = this->wrapIntWithRcp(y, this->params.u64_rcpH, this->params.h);
}
//[[gnu::target("avx512vbmi")]] //todo: change this later
int32x16 WrappingMapper::wrapIntWithRcp(int32x16 x, uint64_t rcp, uint32_t v)
{
	int32x16 ax = abs(x);
	int32x16 rem = wrapPositiveIntWithRcp(ax, rcp, v);
	//TODO: some of these steps may be too cautious and can be removed?
	rem = mask_mov(int32x16(v) - rem, x >= 0, rem);
	rem = mask_mov(rem, rem >= v, rem - int32x16(v));
	rem = mask_mov(rem, rem < 0, rem + int32x16(v));
	return rem;
}
//[[gnu::target("avx512vbmi")]] //todo: change this later
int32x16 WrappingMapper::wrapPositiveIntWithRcp(u32x16 x, uint64_t rcp, uint32_t v)
{
	for (int i = 0; i < 16; ++i) assert(x[i] >= 0);
	auto x64 = vec_cvt<uint64_t>(x);
	auto div = (x64 * rcp) >> 32;
	auto rem = x64 - (div * v);
	return permx2(reinterpret<u32x16>(rem.lo), u32x16(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30), reinterpret<u32x16>(rem.hi));
}

std::pair<float, float> WrappingMapper::wrapUV(float u, float v)
{
	u -= std::floor(u); //doing floor subtraction once sometimes returns 1. Doing it twice guarantees 0 <= u < 1 for all non-nan non-inf values
	u -= std::floor(u);
	v -= std::floor(v);
	v -= std::floor(v);
	return { u,v };
}

std::pair<float32x8, float32x8> WrappingMapper::wrapUV(float32x8 u, float32x8 v)
{
	u -= _mm256_floor_ps(u); //doing floor subtraction once sometimes returns 1. Doing it twice guarantees 0 <= u < 1 for all non-nan non-inf values
	u -= _mm256_floor_ps(u);
	v -= _mm256_floor_ps(v);
	v -= _mm256_floor_ps(v);
	return { u,v };
}

std::pair<float32x16, float32x16> WrappingMapper::wrapUV(float32x16 u, float32x16 v)
{
	u -= _mm512_floor_ps(u); //doing floor subtraction once sometimes returns 1. Doing it twice guarantees 0 <= u < 1 for all non-nan non-inf values
	u -= _mm512_floor_ps(u);
	v -= _mm512_floor_ps(v);
	v -= _mm512_floor_ps(v);
	return { u,v };
}