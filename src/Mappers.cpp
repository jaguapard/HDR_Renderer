#include "Mappers.h"

WrappingMapper::WrappingMapper(uint32_t w, uint32_t h)
{
	this->w = w;
	this->h = h;
	this->fw = w;
	this->fh = h;
	this->u64_rcpW = uint64_t(1ull << 32) / w;
	this->u64_rcpH = uint64_t(1ull << 32) / h;
	this->f64_rcpW = 1.0 / w;
	this->f64_rcpH = 1.0 / h;
	this->f64_w = w;
	this->f64_h = h;
}

void WrappingMapper::wrapInts(int32x16& x, int32x16& y) const
{
	x = this->wrapIntWithRcp(x, u64_rcpW, w);
	y = this->wrapIntWithRcp(y, u64_rcpH, h);
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
