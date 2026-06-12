#include "RendererBase.h"

void RendererBase::mask_store_vec4_f32x16_to_framebuffer(const Vec4_f32x16& pack, void* frameBuffer, int x, int y, int w, Mask16 mask)
{
	//we have px[0] == r0,r1,r2...,r15, px[1] == g0,..g15, ...
	//DX wants: r0,g0,b0,a0,r1,g1,b1,a1, etc
	//Meanings, that first 16-wide register to store should be r0,g0,b0,a0,...,r3,g3,b3,a3
	//Second - 4-7, third - 8-11, fourth - 12-15
	u16x16 ph_r = vec_cvt_ps2ph(pack.r);
	u16x16 ph_g = vec_cvt_ps2ph(pack.g);
	u16x16 ph_b = vec_cvt_ps2ph(pack.b);
	u16x16 ph_a = vec_cvt_ps2ph(pack.a);
	for (int i = 0; i < 16; i += 4)
	{
		u16x16 rg_ind = u16x16(0, 16, 0, 0, 1, 17, 0, 0, 2, 18, 0, 0, 3, 19, 0, 0) + i;
		u16x16 ba_ind = u16x16(0, 0, 0, 16, 0, 0, 1, 17, 0, 0, 2, 18, 0, 0, 3, 19) + i;
		u16x16 rgxx = permx2(ph_r, rg_ind, ph_g);
		u16x16 xxba = permx2(ph_b, ba_ind, ph_a);
		u16x16 rgba = mask_mov(rgxx, 0b1100110011001100, xxba);
		store(reinterpret<u64x4>(rgba), (int64_t*)frameBuffer + y * w + x + i, mask >> i);
	}
}

//TODO: verify that it works
Vec4_f32x16 RendererBase::mask_load_vec4_f32x16_from_framebuffer(const void* frameBuffer, int x, int y, int w, Mask16 mask)
{
	int loadInd = y * w + x;
	const int64_t* p = (const int64_t*)frameBuffer;
	u16x32 rgba0_7 = reinterpret<u16x32>(load<u64x8>(p + loadInd, mask.lo()));
	u16x32 rgba8_15 = reinterpret<u16x32>(load<u64x8>(p + loadInd + 8, mask.hi()));

	u16x32 r0_15_g0_15_ph = permx2(rgba0_7, u16x32(0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61), rgba8_15);
	u16x32 b0_15_a0_15_ph = permx2(rgba0_7, u16x32(2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63), rgba8_15);

	Vec4_f32x16 ret;
	ret.r = vcvt_fp16_fp32(r0_15_g0_15_ph.lo);
	ret.g = vcvt_fp16_fp32(r0_15_g0_15_ph.hi);
	ret.b = vcvt_fp16_fp32(b0_15_a0_15_ph.lo);
	ret.a = vcvt_fp16_fp32(b0_15_a0_15_ph.hi);
	return ret;
}

//TODO: verify that it works
void RendererBase::scatterToFrameBuffer(const Vec4_f32x16& colors, int32x16 x, int32x16 y, Mask16 mask, void* frameBuf, int framebufW)
{
	int32x16 scatterInd = y * framebufW + x;
	u16x16 fp16_r = vec_cvt_ps2ph(colors.r);
	u16x16 fp16_g = vec_cvt_ps2ph(colors.g);
	u16x16 fp16_b = vec_cvt_ps2ph(colors.b);
	u16x16 fp16_a = vec_cvt_ps2ph(colors.a); //TODO: can be forced to 1 and moved later

	u16x32 fp16_rg = concat(fp16_r, fp16_g);
	u16x32 fp16_ba = concat(fp16_b, fp16_a);

	u64x8 rgba0_7 = reinterpret<u64x8>(permx2(fp16_rg, u16x32(0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51, 4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55), fp16_ba));
	u64x8 rgba8_15 = reinterpret<u64x8>(permx2(fp16_rg, u16x32(8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59, 12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63), fp16_ba));
	scatter(rgba0_7, frameBuf, scatterInd.lo, mask.lo());
	scatter(rgba8_15, frameBuf, scatterInd.hi, mask.hi());
}
