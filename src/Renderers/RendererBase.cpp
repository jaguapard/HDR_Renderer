#include "RendererBase.h"

void RendererBase::mask_store_vec4_f32x16_to_framebuffer(const Vec4_f32x16& pack, void* frameBuffer, int x, int y, int w, Mask16 mask)
{
	//we have px[0] == r0,r1,r2...,r15, px[1] == g0,..g15, ...
	//DX wants: r0,g0,b0,a0,r1,g1,b1,a1, etc
	//Meanings, that first 16-wide register to store should be r0,g0,b0,a0,...,r3,g3,b3,a3
	//Second - 4-7, third - 8-11, fourth - 12-15
	__m256i ph_r = _mm512_cvtps_ph(pack.r, _MM_FROUND_TO_NEAREST_INT);
	__m256i ph_g = _mm512_cvtps_ph(pack.g, _MM_FROUND_TO_NEAREST_INT);
	__m256i ph_b = _mm512_cvtps_ph(pack.b, _MM_FROUND_TO_NEAREST_INT);
	__m256i ph_a = _mm512_cvtps_ph(pack.a, _MM_FROUND_TO_NEAREST_INT);
	for (int i = 0; i < 16; i += 4)
	{
		__m256i rg_ind = _mm256_add_epi16(_mm256_set1_epi16(i), _mm256_setr_epi16(0, 16, 0, 0, 1, 17, 0, 0, 2, 18, 0, 0, 3, 19, 0, 0));
		__m256i ba_ind = _mm256_add_epi16(_mm256_set1_epi16(i), _mm256_setr_epi16(0, 0, 0, 16, 0, 0, 1, 17, 0, 0, 2, 18, 0, 0, 3, 19));
		__m256i rgxx = _mm256_permutex2var_epi16(ph_r, rg_ind, ph_g);
		__m256i xxba = _mm256_permutex2var_epi16(ph_b, ba_ind, ph_a);
		__m256i rgba = _mm256_mask_mov_epi16(rgxx, 0b1100110011001100, xxba);
		_mm256_mask_storeu_epi64((int64_t*)frameBuffer + y * w + x + i, mask >> i, rgba);
	}
}

Vec4_f32x16 RendererBase::mask_load_vec4_f32x16_from_framebuffer(const void* frameBuffer, int x, int y, int w, Mask16 mask)
{
	int loadInd = y * w + x;
	const int64_t* p = (const int64_t*)frameBuffer;
	__m512i rgba0_7 = _mm512_maskz_loadu_epi64(mask, p + loadInd);
	__m512i rgba8_15 = _mm512_maskz_loadu_epi64(mask >> 8, p + loadInd + 8);

	__m512i r0_15_g0_15_ph = _mm512_permutex2var_epi16(rgba0_7, _mm512_setr_epi16(0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61), rgba8_15);
	__m512i b0_15_a0_15_ph = _mm512_permutex2var_epi16(rgba0_7, _mm512_setr_epi16(2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63), rgba8_15);

	Vec4_f32x16 ret;
	ret.r = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(r0_15_g0_15_ph, 0));
	ret.g = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(r0_15_g0_15_ph, 1));
	ret.b = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(b0_15_a0_15_ph, 0));
	ret.a = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(b0_15_a0_15_ph, 1));
	return ret;
}

void RendererBase::scatterToFrameBuffer(const Vec4_f32x16& colors, int32x16 x, int32x16 y, Mask16 mask, void* frameBuf, int framebufW)
{
	int32x16 scatterInd = y * framebufW + x;
	__m256i fp16_r = _mm512_cvtps_ph(colors.r, _MM_FROUND_TO_NEAREST_INT);
	__m256i fp16_g = _mm512_cvtps_ph(colors.g, _MM_FROUND_TO_NEAREST_INT);
	__m256i fp16_b = _mm512_cvtps_ph(colors.b, _MM_FROUND_TO_NEAREST_INT);
	__m256i fp16_a = _mm512_cvtps_ph(colors.a, _MM_FROUND_TO_NEAREST_INT); //TODO: can be forced to 1 and moved later

	int32x16 fp16_rg = _mm512_inserti32x8(_mm512_castsi256_si512(fp16_r), fp16_g, 1);
	int32x16 fp16_ba = _mm512_inserti32x8(_mm512_castsi256_si512(fp16_b), fp16_a, 1);
	int32x16 rgba0_7 = _mm512_permutex2var_epi16(fp16_rg, _mm512_setr_epi16(0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51, 4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55), fp16_ba);
	int32x16 rgba8_15 = _mm512_permutex2var_epi16(fp16_rg, _mm512_setr_epi16(8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59, 12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63), fp16_ba);
	_mm512_mask_i32scatter_epi64(frameBuf, mask, _mm512_extracti32x8_epi32(scatterInd, 0), rgba0_7, 8);
	_mm512_mask_i32scatter_epi64(frameBuf, mask >> 8, _mm512_extracti32x8_epi32(scatterInd, 1), rgba8_15, 8);
}
