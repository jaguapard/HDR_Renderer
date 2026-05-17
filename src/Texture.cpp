#include "Texture.h"
#include "helpers.h"
MipLevel::MipLevel(uint32_t w, uint32_t h)
{
	this->mapper = { w,h };
	this->colors.resize(w * h);
}

[[gnu::target("avx512vbmi")]]
Vec4_f32x16 MipLevel::gatherLinearIntensities(float32x16 u, float32x16 v, Mask16 mask) const
{
    auto [pixelsX, pixelsY] = this->mapper.UV_to_XY(u, v);
    float32x16 lerpT_x = pixelsX - float32x16(_mm512_floor_ps(pixelsX));
    float32x16 lerpT_y = pixelsY - float32x16(_mm512_floor_ps(pixelsY));
    int32x16 startX = pixelsX.trunc();
    int32x16 startY = pixelsY.trunc();
    const auto& p = this->mapper.getParams();

    std::array<Vec4_f32x16, 4> linear;
    for (int i = 0; i < 4; ++i)
    {
        int sx = i % 2;
        int sy = i / 2;
        int32x16 sampleX = startX + sx, sampleY = startY + sy;
        this->mapper.wrapInts(sampleX, sampleY);

        
        int32x16 samples = int32x16::gather(this->colors.data(), sampleY * p.w + sampleX, mask);
        linear[i] = Decoder::RGBA8888_to_linear_using_FP16_LUT(samples);
    }

    Vec4_f32x16 lerp1 = lerp(linear[0], linear[1], lerpT_x);
    Vec4_f32x16 lerp2 = lerp(linear[2], linear[3], lerpT_x);
    return lerp(lerp1, lerp2, lerpT_y);
}