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

const WrappingMapper::Params& WrappingMapper::getParams() const
{
	return params;
}

void WrappingMapper::wrapInts(int32x16& x, int32x16& y) const
{
	x = this->wrapIntWithRcp(x, this->params.u64_rcpW, this->params.w);
	y = this->wrapIntWithRcp(y, this->params.u64_rcpH, this->params.h);
}

int32x16 WrappingMapper::wrapIntWithRcp(int32x16 x, uint64_t rcp, uint32_t v)
{
	int32x16 ax = _mm512_abs_epi32(x);
	int32x16 rem = wrapPositiveIntWithRcp(ax, rcp, v);
	//TODO: some of these steps may be too cautious and can be removed?
	rem = _mm512_mask_mov_epi32(int32x16(v) - rem, x >= 0, rem);
	rem = _mm512_mask_sub_epi32(rem, rem >= v, rem, int32x16(v));
	rem = _mm512_mask_add_epi32(rem, rem < 0, rem, int32x16(v));
	return rem;
}

void WrappingMapper::wrapIntsPositive(int32x16& x, int32x16& y) const
{
	for (int i = 0; i < 16; ++i) assert(x[i] >= 0 && y[i] >= 0);
}

int32x16 WrappingMapper::wrapPositiveIntWithRcp(int32x16 x, uint64_t rcp, uint32_t v)
{
	for (int i = 0; i < 16; ++i) assert(x[i] >= 0);
	__m512i lo64 = _mm512_cvtepu32_epi64(_mm512_extracti32x8_epi32(x, 0));
	__m512i hi64 = _mm512_cvtepu32_epi64(_mm512_extracti32x8_epi32(x, 1));

	__m512i divLo = _mm512_srli_epi64(_mm512_mullo_epi64(lo64, _mm512_set1_epi64(rcp)), 32);
	__m512i divHi = _mm512_srli_epi64(_mm512_mullo_epi64(hi64, _mm512_set1_epi64(rcp)), 32);
	__m512i remLo = _mm512_sub_epi64(lo64, _mm512_mullo_epi64(divLo, _mm512_set1_epi64(v)));
	__m512i remHi = _mm512_sub_epi64(hi64, _mm512_mullo_epi64(divHi, _mm512_set1_epi64(v)));
	__m256i reml = _mm512_cvtepi64_epi32(remLo);
	__m256i remh = _mm512_cvtepi64_epi32(remHi);
	return _mm512_inserti32x8(_mm512_castsi256_si512(reml), remh, 1);
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