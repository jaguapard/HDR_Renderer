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

uint32_t MipLevel::getPixelRGBA32(uint32_t x, uint32_t y) const
{
    return this->colors[y * mapper.getParams().w + x];
}

[[gnu::target("avx512vbmi")]]
Vec4_f32x16 MipLevel::gatherLinearIntensities(const float32x16& u, const float32x16& v, Mask16 mask) const
{
    auto [pixelsX, pixelsY] = this->mapper.UV_to_XY(u, v);
    float32x16 lerpT_x = pixelsX - floor(pixelsX);
    float32x16 lerpT_y = pixelsY - floor(pixelsY);
    int32x16 startX = vcvt<int>(pixelsX);
    int32x16 startY = vcvt<int>(pixelsY);
    const auto& p = this->mapper.getParams();

    std::array<Vec4_f32x16, 4> linear;
    for (int i = 0; i < 4; ++i)
    {
        int sx = i % 2;
        int sy = i / 2;
        int32x16 sampleX = startX + sx, sampleY = startY + sy;
        this->mapper.wrapInts(sampleX, sampleY);

        u32x16 samples = gather<u32x16>(this->colors.data(), sampleY * p.w + sampleX, mask);
        linear[i] = Decoder::RGBA8888_to_linear_using_FP16_LUT(samples);
    }

    Vec4_f32x16 lerp1 = lerp(linear[0], linear[1], lerpT_x);
    Vec4_f32x16 lerp2 = lerp(linear[2], linear[3], lerpT_x);
    return lerp(lerp1, lerp2, lerpT_y);
}

float32x16 MipLevel::gatherA(const float32x16& u, const float32x16& v, Mask16 mask) const
{
    auto [pixelsX, pixelsY] = this->mapper.UV_to_XY(u, v);
    //TODO: same filtering for opacity maps as textures, else it creates disagreement between stages
    int32x16 sx = vcvt<int>(pixelsX);
    int32x16 sy = vcvt<int>(pixelsY);
    this->mapper.wrapInts(sx, sy);

    int32x16 ind = sy * this->mapper.getParams().w + sx;
    int32x16 gatherInd = ind >> 5;
    int32x16 shifts = ind & 31;
    int32x16 gathered = gather<i32x16>(this->opacityMap.data(), gatherInd, mask);
    gathered &= int32x16(1) << shifts;
    return mask_mov(float32x16(0.f), gathered != 0, float32x16(1.f));
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
                        int32x16 srcUint32 = load<i32x16>(srcRow + x, boundsMask);

                        int32x16 dstR = srcUint32 & 0xFF;
                        int32x16 dstG = (srcUint32 >> 8) & 0xFF;
                        int32x16 dstB = (srcUint32 >> 16) & 0xFF;
                        int32x16 dstA = (srcUint32 >> 24) & 0xFF;
                        int32x16 dstFull = dstR | (dstG << 8) | (dstB << 16) | (dstA << 24);
                        store(dstFull, &this->mipLevels[0].colors[y * w + x], boundsMask);
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
            i32x16 packed1 = load<i32x16>(&this->mipLevels[0].colors[i], boundsMask1);
            i32x16 packed2 = load<i32x16>(&this->mipLevels[0].colors[i+16], boundsMask2);
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

Vec4_f32x16 Texture::gatherLinearIntensities(const float32x16& u, const float32x16& v, const Mask16& mask) const
{
    return this->mipLevels[0].gatherLinearIntensities(u, v, mask);
}

float32x16 Texture::gatherA(const float32x16& u, const float32x16& v, const Mask16& mask) const
{
    return this->mipLevels[0].gatherA(u, v, mask);
}

void Texture::QueryTexture(uint32_t* w, uint32_t* h, std::unique_ptr<uint32_t[]>* retRGBA32, int mipLevel) const
{
    const auto& mip = this->mipLevels[mipLevel];
    uint32_t mw = mip.mapper.getParams().w;
    uint32_t mh = mip.mapper.getParams().h;
    if (w) *w = mw;
    if (h) *h = mh;
    if (retRGBA32)
    {
        std::unique_ptr<uint32_t[]> rd = std::make_unique<uint32_t[]>(mip.mapper.getParams().w * mip.mapper.getParams().h);
        for (uint32_t y = 0; y < mh; ++y)
        {
            for (uint32_t x = 0; x < mw; ++x)
            {
                rd[y * mw + x] = mip.getPixelRGBA32(x, y);
            }
        }
        *retRGBA32 = std::move(rd);
    }
}
