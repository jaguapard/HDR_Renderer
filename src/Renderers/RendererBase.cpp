#include "RendererBase.h"

void RendererBase::mask_store_vec4_f32x16_to_framebuffer(const Vec4_f32x16& pack, void* frameBuffer, int x, int y, int w, Mask16 mask)
{
	//we have px[0] == r0,r1,r2...,r15, px[1] == g0,..g15, ...
	//DX wants: r0,g0,b0,a0,r1,g1,b1,a1, etc
	//Meanings, that first 16-wide register to store should be r0,g0,b0,a0,...,r3,g3,b3,a3
	//Second - 4-7, third - 8-11, fourth - 12-15
	constexpr int DC = 0; //garbage value
	//duplicate each opaquePixelsMask bit 4 times, i.e: 0123 -> 0000111122223333, 16 bits -> 64
	__m512i expanded = _mm512_maskz_mov_epi32(mask, _mm512_set1_epi32(-1));
	__mmask64 duplicated = _mm512_cmpneq_epi8_mask(expanded, _mm512_set1_epi32(0));

	__m256i ph_r = _mm512_cvtps_ph(pack.r, _MM_FROUND_NO_EXC);
	__m256i ph_g = _mm512_cvtps_ph(pack.g, _MM_FROUND_NO_EXC);
	__m256i ph_b = _mm512_cvtps_ph(pack.b, _MM_FROUND_NO_EXC);
	__m256i ph_a = _mm512_cvtps_ph(pack.a, _MM_FROUND_NO_EXC);
	for (int i = 0; i < 16; i += 4)
	{
		__m256i rg_ind = _mm256_add_epi16(_mm256_set1_epi16(i), _mm256_setr_epi16(0, 16, DC, DC, 1, 17, DC, DC, 2, 18, DC, DC, 3, 19, DC, DC));
		__m256i ba_ind = _mm256_add_epi16(_mm256_set1_epi16(i), _mm256_setr_epi16(DC, DC, 0, 16, DC, DC, 1, 17, DC, DC, 2, 18, DC, DC, 3, 19));
		__m256i rgxx = _mm256_permutex2var_epi16(ph_r, rg_ind, ph_g);
		__m256i xxba = _mm256_permutex2var_epi16(ph_b, ba_ind, ph_a);
		__m256i rgba = _mm256_mask_mov_epi16(rgxx, 0b1100110011001100, xxba);
		int storeInd = (y * w + x + i) * 4;
		_mm256_mask_storeu_epi16((int16_t*)frameBuffer + storeInd, duplicated >> (i * 4), rgba);
	}
}
