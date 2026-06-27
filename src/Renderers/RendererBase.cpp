#include "RendererBase.h"

//TODO: verify that it works
Vec4_f32x16 RendererBase::mask_load_vec4_f32x16_from_framebuffer(const void* frameBuffer, int x, int y, int w, mask16d mask)
{
	int loadInd = y * w + x;
	const int64_t* p = (const int64_t*)frameBuffer;
	auto rgba0_7 = vcast<fp16x32>(load<u64x8>(p + loadInd, mask.lo()));
	auto rgba8_15 = vcast<fp16x32>(load<u64x8>(p + loadInd + 8, mask.hi()));

	u16x32 r0_15_g0_15_ph = permx2(rgba0_7, rgba8_15, u16x32(0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61));
	u16x32 b0_15_a0_15_ph = permx2(rgba0_7, rgba8_15, u16x32(2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63));

	Vec4_f32x16 ret;
	ret.r = r0_15_g0_15_ph.lo();
	ret.g = r0_15_g0_15_ph.hi();
	ret.b = b0_15_a0_15_ph.lo();
	ret.a = b0_15_a0_15_ph.hi();
	return ret;
}

//TODO: verify that it works
void RendererBase::scatterToFrameBuffer(const Vec4_f32x16& colors, int32x16 x, int32x16 y, mask16d mask, void* frameBuf, int framebufW)
{
	int32x16 scatterInd = y * framebufW + x;
	fp16x16 fp16_r = colors.r;
	fp16x16 fp16_g = colors.g;
	fp16x16 fp16_b = colors.b;
	fp16x16 fp16_a = colors.a; //TODO: can be forced to 1 and moved later

	fp16x32 fp16_rg = { fp16_r, fp16_g };
	fp16x32 fp16_ba = { fp16_b, fp16_a };

	u64x8 rgba0_7 = vcast<u64x8>(permx2(fp16_rg, fp16_ba, u16x32(0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51, 4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55)));
	u64x8 rgba8_15 = vcast<u64x8>(permx2(fp16_rg, fp16_ba, u16x32(8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59, 12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63)));
	scatter(rgba0_7, frameBuf, scatterInd.lo(), mask.lo());
	scatter(rgba8_15, frameBuf, scatterInd.hi(), mask.hi());
}
