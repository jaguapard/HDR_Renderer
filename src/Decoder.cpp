#include "Decoder.h"
#include "LUTMan.h"
#include "helpers.h"

Vec4_f32x16 Decoder::R10G11B10A1_gamma2_to_linear(int32x16 packed)
{
    u32x16 up = vec_cvt<uint32_t>(packed);
    u32x16 r = up & 1023;
    u32x16 g = (up >> 10) & 2047;
    u32x16 b = (up >> 21) & 1023;

    float32x16 fr = vec_cvt<float>(r);
    float32x16 fg = vec_cvt<float>(g);
    float32x16 fb = vec_cvt<float>(b);
    float32x16 fa = maskz_mov(packed < 0, float32x16(1)); //if uppermost bit is 1 (i.e. sign bit is 1, i.e negative), then alpha is 1
    fr *= 1.f / 1023;
    fg *= 1.f / 2047;
    fb *= 1.f / 1023;
    return { fr * fr, fg * fg, fb * fb, fa };
}

[[gnu::target("avx512vbmi")]]
Vec4_f32x16 Decoder::RGBA8888_to_linear_using_FP16_LUT(const u32x16& packed)
{
    std::array<u16x32, 8> lut;
    for (int i = 0; i < 8; ++i) lut[i] = load<u16x32>(&LUTMan::tables.rgbToLinear_fp16[i * 32]);
    Vec4_f32x16 ret;
    ret.a = f32x16(packed >> 24) / 255; //alpha channel is linear already, not gamma encoded
    //zero-extend and split channels into halves, i.e. rgba,rgba,rgba,rgba is now r_r_r_r_g_g_g_g_, b and a in other
    //using setr16 for convinience. Doesn't matter what we put in upper bytes of each 16 byte word, since that will be zeroed out by zero-masking
    //TODO: some fix for this massive reinterpreting
    zmm_u16 rg = reinterpret<zmm_u16>(permx(reinterpret<zmm_u8>(packed), reinterpret<zmm_u8>(zmm_u16(0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61))));
    zmm_u16 ba = reinterpret<zmm_u16>(permx(reinterpret<zmm_u8>(packed), reinterpret<zmm_u8>(zmm_u16(2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63))));
    rg = reinterpret<zmm_u16>(maskz_mov(0x5555555555555555, reinterpret<zmm_u8>(rg)));
    ba = reinterpret<zmm_u16>(maskz_mov(0x5555555555555555, reinterpret<zmm_u8>(ba)));

    zmm_u16 fp16_rg = 0, fp16_ba = 0;
    for (int j = 0; j < 4; ++j)
    {
        //no need to change index, permutex2var already ignores upper bits.
        zmm_u16 perm_rg = permx2(lut[2 * j], reinterpret<zmm_u16>(rg), lut[2 * j + 1]);
        zmm_u16 perm_ba = permx2(lut[2 * j], reinterpret<zmm_u16>(ba), lut[2 * j + 1]);

        //only update positions that are part of this LUT slice
        fp16_rg = mask_mov(fp16_rg, (rg > (j * 64)) & (rg < (j * 64 + 64)), perm_rg);
        fp16_ba = mask_mov(fp16_ba, (ba > (j * 64)) & (ba < (j * 64 + 64)), perm_ba);
    }

    ret.r = _mm512_cvtph_ps(fp16_rg.lo);
    ret.g = _mm512_cvtph_ps(fp16_rg.hi);
    ret.b = _mm512_cvtph_ps(fp16_ba.lo);
    return ret;
}