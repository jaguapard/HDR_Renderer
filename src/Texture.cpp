#include "Texture.h"
#include "helpers.h"
#include "Decoder.h"
#include "Threadpool.h"
#include <stdexcept>

int32x16 morton_half(int32x16 v)
{
    v = (v | (v << 8)) & 0x00FF00FF;
    v = (v | (v << 4)) & 0x0F0F0F0F;
    v = (v | (v << 2)) & 0x33333333;
    v = (v | (v << 1)) & 0x55555555;
    return v;
}

//only works for 16 bit x and y!!!
int32x16 morton(int32x16 x, int32x16 y)
{
    return morton_half(x) | (morton_half(y) << 1);
}

MipLevel::MipLevel(uint32_t w, uint32_t h)
{
	this->mapper = { w,h };
	this->colors.resize(w * h);
    this->opacityMap.resize((w * h) / 32 + 1);

    this->colors.shrink_to_fit();
    this->opacityMap.shrink_to_fit();
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

float32x16 MipLevel::gatherA(float32x16 u, float32x16 v, Mask16 mask) const
{
    auto [pixelsX, pixelsY] = this->mapper.UV_to_XY(u, v);
    //TODO: same filtering for opacity maps as textures, else it creates disagreement between stages
    int32x16 sx = pixelsX.trunc();
    int32x16 sy = pixelsY.trunc();
    this->mapper.wrapInts(sx, sy);

    int32x16 ind = sy * this->mapper.getParams().w + sx;
    int32x16 gatherInd = ind >> 5;
    int32x16 shifts = ind & 31;
    int32x16 gathered = int32x16::gather(this->opacityMap.data(), gatherInd, mask);
    gathered &= int32x16(1) << shifts;
    return _mm512_mask_mov_ps(float32x16(0.f), gathered != 0, float32x16(1.f));
}


Texture::Texture(const SDL_Surface* s)
{
    int w = s->w;
    int h = s->h;

    if (s->format == SDL_PIXELFORMAT_RGBA32)
    {
        const uint32_t* srcPixels = std::bit_cast<uint32_t*>(s->pixels);
        this->mipLevels.emplace_back(w, h);

        std::vector<Threadpool::TaskHandle> tasks;
        int tCount = Threadpool::instance->getWorkerCount();
        for (int tIndex = 0; tIndex < tCount; ++tIndex)
        {
            //TODO: this kills everything. Threadpool is not ready for tasks within tasks yet
            //tasks.push_back(Threadpool::instance->addTask([&, tIndex, this] 
            {
                auto [low, high] = Threadpool::instance->getLimitsForThread(tIndex, 0, h, tCount);
                for (int y = low; y < high; ++y)
                {
                    const uint32_t* srcRow = std::bit_cast<const uint32_t*>(size_t(srcPixels) + s->pitch * y);
                    for (int x = 0; x < w; x += 16)
                    {
                        //alpha is binary, all values above 0 considered fully opaque. TODO: when implementing transparency, change this
                        Mask16 boundsMask = (int32x16::sequence() + x) < w;
                        int32x16 srcUint32 = _mm512_maskz_loadu_epi32(boundsMask, srcRow + x);

                        int32x16 dstR = srcUint32 & 0xFF;
                        int32x16 dstG = (srcUint32 >> 8) & 0xFF;
                        int32x16 dstB = (srcUint32 >> 16) & 0xFF;
                        int32x16 dstA = (srcUint32 >> 24) & 0xFF;
                        int32x16 dstFull = dstR | (dstG << 8) | (dstB << 16) | (dstA << 24);
                        _mm512_mask_storeu_epi32(&this->mipLevels[0].colors[y * w + x], boundsMask, dstFull.zmm);
                    }
                }
            }
            //));
        }
        //Threadpool::instance->waitForMultipleTasks(tasks);
        int totalPixels = w * h;
        for (int i = 0; i < totalPixels; i += 32)
        {
            Mask16 boundsMask1 = (int32x16::sequence() + i) < totalPixels;
            Mask16 boundsMask2 = (int32x16::sequence() + i + 16) < totalPixels;
            int32x16 packed1 = _mm512_maskz_loadu_epi32(boundsMask1, &this->mipLevels[0].colors[i]);
            int32x16 packed2 = _mm512_maskz_loadu_epi32(boundsMask2, &this->mipLevels[0].colors[i + 16]);
            Vec4_f32x16 p1, p2;
            p1 = Decoder::RGBA8888_to_linear_using_FP16_LUT(packed1);
            p2 = Decoder::RGBA8888_to_linear_using_FP16_LUT(packed2);
            Mask16 m1 = p1.a > 0.f;
            Mask16 m2 = p2.a > 0.f;
            uint32_t mt = (uint32_t(m2) << 16) | m1;
            this->mipLevels[0].opacityMap[i / 32] = mt;
            //if (~mt) this->mipLevels[0]isFullyOpaque = false;
        }
    }
    else
    {
        throw std::runtime_error("Unsupported pixel format for ColorPixelBuffer import: ");
    }
}

Vec4_f32x16 Texture::gatherLinearIntensities(float32x16 u, float32x16 v, Mask16 mask) const
{
    return this->mipLevels[0].gatherLinearIntensities(u, v, mask);
}

float32x16 Texture::gatherA(float32x16 u, float32x16 v, Mask16 mask) const
{
    return this->mipLevels[0].gatherA(u, v, mask);
}