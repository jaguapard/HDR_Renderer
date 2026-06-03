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
    __m512i rg = _mm512_maskz_permutexvar_epi8(0x5555555555555555, _mm512_setr_epi16(0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61), packed);
    __m512i ba = _mm512_maskz_permutexvar_epi8(0x5555555555555555, _mm512_setr_epi16(2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63), packed);

    __m512i fp16_rg, fp16_ba;
    fp16_rg = fp16_ba = _mm512_setzero_si512();
    for (int j = 0; j < 4; ++j)
    {
        //no need to change index, permutex2var already ignores upper bits.
        __m512i perm_rg = _mm512_permutex2var_epi16(lut[2 * j], rg, lut[2 * j + 1]);
        __m512i perm_ba = _mm512_permutex2var_epi16(lut[2 * j], ba, lut[2 * j + 1]);

        //only update positions that are part of this LUT slice
        __mmask32 lb1 = _mm512_cmpge_epu16_mask(rg, _mm512_set1_epi16(j * 64));
        __mmask32 ub1 = _mm512_cmplt_epu16_mask(rg, _mm512_set1_epi16((j + 1) * 64));
        __mmask32 m1 = lb1 & ub1;
        __mmask32 lb2 = _mm512_cmpge_epu16_mask(ba, _mm512_set1_epi16(j * 64));
        __mmask32 ub2 = _mm512_cmplt_epu16_mask(ba, _mm512_set1_epi16((j + 1) * 64));
        __mmask32 m2 = lb2 & ub2;
        fp16_rg = _mm512_mask_mov_epi16(fp16_rg, m1, perm_rg);
        fp16_ba = _mm512_mask_mov_epi16(fp16_ba, m2, perm_ba);
    }

    ret.r = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(fp16_rg, 0));
    ret.g = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(fp16_rg, 1));
    ret.b = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(fp16_ba, 0));
    return ret;
}