#include "Decoder.h"
#include "LUTMan.h"
#include "helpers.h"

Vec4_f32x16 Decoder::R10G11B10A1_gamma2_to_linear(int32x16 packed)
{
    u32x16 r = packed & 1023;
    u32x16 g = (packed >> 10) & 2047;
    u32x16 b = (packed >> 21) & 1023;

    float32x16 fr = r;
    float32x16 fg = g;
    float32x16 fb = b;
    float32x16 fa = maskz_mov(packed < 0, float32x16(1)); //if uppermost bit is 1 (i.e. sign bit is 1, i.e negative), then alpha is 1
    fr *= 1.f / 1023;
    fg *= 1.f / 2047;
    fb *= 1.f / 1023;
    return { fr * fr, fg * fg, fb * fb, fa };
}

//[[gnu::target("avx512vbmi")]]
Vec4_f32x16 Decoder::RGBA8888_to_linear_using_FP16_LUT(const u32x16& packed)
{
    std::array<u16x32, 8> lut;
    for (int i = 0; i < 8; ++i) lut[i] = load<u16x32>(&LUTMan::tables.rgbToLinear_fp16[i * 32]);
    Vec4_f32x16 ret;
    ret.a = f32x16(packed >> 24) / 255; //alpha channel is linear already, not gamma encoded
    //zero-extend and split channels into halves, i.e. rgba,rgba,rgba,rgba is now r_r_r_r_g_g_g_g_, b and a in other
    //using setr16 for convinience. Doesn't matter what we put in upper bytes of each 16 byte word, since that will be zeroed out by zero-masking
    //TODO: some fix for this massive reinterpreting
    u8x64 bytes = vcast<u8x64>(packed);
    u16x32 rg = vcast<u16x32>(maskz_mov(0x5555555555555555, permx(bytes, vcast<u8x64>(u16x32(0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61)))));
    u16x32 ba = vcast<u16x32>(maskz_mov(0x5555555555555555, permx(bytes, vcast<u8x64>(u16x32(2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63)))));

    u16x32 fp16_rg = 0, fp16_ba = 0;
    for (int j = 0; j < 4; ++j)
    {
        //no need to change index, permutex2var already ignores upper bits.
        u16x32 perm_rg = permx2(lut[2 * j], lut[2 * j + 1], rg);
        u16x32 perm_ba = permx2(lut[2 * j], lut[2 * j + 1], ba);

        //only update positions that are part of this LUT slice
        int lo = j * 64, hi = lo + 64;
        fp16_rg = mask_mov(fp16_rg, rg >= lo & rg < hi, perm_rg);
        fp16_ba = mask_mov(fp16_ba, ba >= lo & ba < hi, perm_ba);
    }

    ret.r = vcast<fp16x16>(fp16_rg.lo());
    ret.g = vcast<fp16x16>(fp16_rg.hi());
    ret.b = vcast<fp16x16>(fp16_ba.lo());
    return ret;
}