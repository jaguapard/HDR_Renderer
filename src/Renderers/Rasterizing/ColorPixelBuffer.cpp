#include "ColorPixelBuffer.h"
#include <stdexcept>
#include "../../Threadpool.h"
#include "../../helpers.h"
using namespace Rasterizing;

Rasterizing::ColorPixelBuffer::ColorPixelBuffer(ColorPixelBuffer&& dying) :
    mipLevels(std::move(dying.mipLevels))
   // isFullyOpaque(dying.isFullyOpaque)
{
}

Rasterizing::ColorPixelBuffer::ColorPixelBuffer(uint32_t w, uint32_t h)
{
    this->init(w, h);
}


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

void Rasterizing::ColorPixelBuffer::init(uint32_t w, uint32_t h)
{
    if (!ColorPixelBuffer::toLinearLUT_fp16 || !ColorPixelBuffer::toLinearLUT_fp32)
    {
        ColorPixelBuffer::toLinearLUT_fp16 = std::make_unique<int16_t[]>(256);
        ColorPixelBuffer::toLinearLUT_fp32 = std::make_unique<float[]>(256);
        for (int i = 0; i < 256; ++i)
        {
            double normalized = i / 255.0;
            float linear = std::pow(normalized, 2.2);
            ColorPixelBuffer::toLinearLUT_fp32[i] = linear;
            ColorPixelBuffer::toLinearLUT_fp16[i] = _mm_extract_epi16(_mm_cvtps_ph(_mm_set1_ps(linear), 0), 0);
        }
    }
}

Rasterizing::ColorPixelBuffer::ColorPixelBuffer(const SDL_Surface* s)
{
    int w = s->w;
    int h = s->h;

    if (s->format == SDL_PIXELFORMAT_RGBA32)
    {
        const uint32_t* srcPixels = std::bit_cast<uint32_t*>(s->pixels);
        this->init(w, h);
        assert(this->mipLevels.size() == 0);

        int pixelCount = w * h;
        std::vector<float> initialLinearTexture(pixelCount * 4);
        MipLevel& fullTexture = this->mipLevels.emplace_back(w, h);
        for (int y = 0; y < h; ++y)
        {
            const uint32_t* srcRow = std::bit_cast<const uint32_t*>(size_t(srcPixels) + s->pitch * y);
            for (int x = 0; x < w; ++x)
            {
                uint32_t sourcePixel = srcRow[x];
                uint32_t r = sourcePixel & 0xFF;
                uint32_t g = (sourcePixel >> 8) & 0xFF;
                uint32_t b = (sourcePixel >> 16) & 0xFF;
                uint32_t a = (sourcePixel >> 24) & 0xFF;

                uint32_t dstPixelIndex = (y * w + x);
                uint32_t dstLinearIndex = dstPixelIndex * 4;
                fullTexture.packedColors[y * w + x] = sourcePixel;
                initialLinearTexture[dstLinearIndex] = ColorPixelBuffer::toLinearLUT_fp32[r];
                initialLinearTexture[dstLinearIndex + 1] = ColorPixelBuffer::toLinearLUT_fp32[g];
                initialLinearTexture[dstLinearIndex + 2] = ColorPixelBuffer::toLinearLUT_fp32[b];
                initialLinearTexture[dstLinearIndex + 3] = a / 255.f;
            }
        }
        
        std::vector<float> prevLevelLinear = std::move(initialLinearTexture);
        uint32_t prevW, prevH;
        while (w > 1 && h > 1)
        {
            prevW = w;
            prevH = h;
            w /= 2;
            h /= 2;
            std::vector<float> mipLinear(w * h * 4);
            MipLevel& currMipMap = this->mipLevels.emplace_back(w, h);
            for (int mipY = 0; mipY < h; ++mipY)
            {
                for (int mipX = 0; mipX < w; ++mipX)
                {
                    //uint32_t dstInd = y * w + x;
                    float linearMipR = 0, linearMipG = 0, linearMipB = 0, srcA;
                    for (int oy = 0; oy < 2; ++oy)
                    {
                        for (int ox = 0; ox < 2; ++ox)
                        {
                            uint32_t srcIndStart = ((mipY * 2 + oy) * prevW + (mipX * 2 + ox)) * 4;
                            linearMipR += prevLevelLinear[srcIndStart];
                            linearMipG += prevLevelLinear[srcIndStart + 1];
                            linearMipB += prevLevelLinear[srcIndStart + 2];
                            if (ox == 0 && oy == 0) srcA = prevLevelLinear[srcIndStart + 3]; //pass through parent alpha for now
                        }
                    }
                    uint32_t pixelStoreIndex = mipY * w + mipX;
                    uint32_t linearStoreIndexStart = pixelStoreIndex * 4;
                    linearMipR /= 4;
                    linearMipG /= 4;
                    linearMipB /= 4;
                    mipLinear[linearStoreIndexStart] = linearMipR;
                    mipLinear[linearStoreIndexStart + 1] = linearMipG;
                    mipLinear[linearStoreIndexStart + 2] = linearMipB;
                    mipLinear[linearStoreIndexStart + 3] = srcA;

                    uint32_t uR = std::pow(linearMipR, 1.f / 2.2) * 255;
                    uint32_t uG = std::pow(linearMipG, 1.f / 2.2) * 255;
                    uint32_t uB = std::pow(linearMipB, 1.f / 2.2) * 255;
                    uint32_t uA = srcA * 255;
                    uint32_t dstUint32 = uR | (uG << 8) | (uB << 16) | (uA << 24);
                    currMipMap.packedColors[pixelStoreIndex] = dstUint32;
                }
            }

            prevLevelLinear = std::move(mipLinear);
        }
       
        for (auto& mipLevel : this->mipLevels)
        {
            int totalPixels = mipLevel.sizes.w * mipLevel.sizes.h;
            for (int i = 0; i < totalPixels; i += 32)
            {
                Mask16 boundsMask1 = (int32x16::sequence() + i) < totalPixels;
                Mask16 boundsMask2 = (int32x16::sequence() + i + 16) < totalPixels;
                int32x16 packed1 = _mm512_maskz_loadu_epi32(boundsMask1, mipLevel.packedColors.get() + i);
                int32x16 packed2 = _mm512_maskz_loadu_epi32(boundsMask2, mipLevel.packedColors.get() + i + 16);
                Mask16 m1 = (packed1 & int32x16(0x80000000)) != 0;
                Mask16 m2 = (packed2 & int32x16(0x80000000)) != 0;
                uint32_t mt = (uint32_t(m2) << 16) | m1;
                mipLevel.opacityMap[i / 32] = mt;
                if (~mt) mipLevel.isFullyOpaque = false;
            }
        }
    }

    /*
    * This format supplies linear intensities. Alpha uncertain. Don't use for now
    if (s->format == SDL_PIXELFORMAT_RGBA128_FLOAT)
    {
        //if (s->pitch % sizeof(Vec4f) != 0) throw std::runtime_error("ABGR128 input pitch not divisible by sizeof float!");
        const Vec4f* srcPixels = (Vec4f*)(s->pixels);
        this->init(w, h);
        for (int y = 0; y < h; ++y)
        {
            const Vec4f* srcRow = std::bit_cast<const Vec4f*>(size_t(srcPixels) + s->pitch * y);
            for (int x = 0; x < w; ++x)
            {
                Vec4f linear = srcRow[x];
                Vec4f gammaEncoded = _mm_pow_ps(linear, _mm_set1_ps(1/2.2));
                gammaEncoded *= 255;
                uint32_t dstR = gammaEncoded.x, dstG = gammaEncoded.y, dstB = gammaEncoded.z, dstA = linear.w*255;
                
                //TODO: gamma 2 and expansion to R/G/B 11/11/10 bits and 1 bit alpha. For now, just save back
                //TODO: seems a bit bright?
                this->packedColors[y * w + x] = dstR | (dstG << 8) | (dstB << 16) | (dstA << 24);
            }
        }
    }*/
    else
    {
        throw std::runtime_error("Unsupported pixel format for ColorPixelBuffer import: ");
    }
}
/*
Vec4_f32x16 Rasterizing::ColorPixelBuffer::gatherLinearIntensities(float32x16 x, float32x16 y, Mask16 mask) const
{
    auto [pixelsX, pixelsY] = Mapper::UV_to_XY<MappingType::WRAP>(x, y, sizes.w, sizes.h);
    int32x16 intX = pixelsX.trunc();
    int32x16 intY = pixelsY.trunc();
    for (int i = 0; i < 16; ++i)
    {
        if (mask.mask & (1 << i))
        {
            assert(intX[i] >= 0 && intX[i] < sizes.w);
            assert(intY[i] >= 0 && intY[i] < sizes.h);
        }
    }
    int32x16 pixelsIndices = intY * sizes.w + intX;
    Mask16 gatherMask = mask;
    int32x16 gathered = _mm512_mask_i32gather_epi32(int32x16(0), gatherMask, pixelsIndices, this->packedColors.get(), 4);
    return Decoder::R10G11B10A1_gamma2_to_linear(gathered);
}
*/
Rasterizing::ColorPixelBufferGatherAccessor Rasterizing::ColorPixelBuffer::getGatherAccessor(float32x16 u, float32x16 v, Mask16 mask) const
{
    auto [pixelsX, pixelsY] = Mapper::UV_to_XY<MappingType::WRAP>(u, v, this->mipLevels[0].sizes.w, this->mipLevels[0].sizes.h);
    int32x16 intX = pixelsX.trunc();
    int32x16 intY = pixelsY.trunc();
    for (int i = 0; i < 16; ++i)
    {
        if (mask.mask & (1 << i))
        {
            assert(intX[i] >= 0 && intX[i] < sizes.w);
            assert(intY[i] >= 0 && intY[i] < sizes.h);
        }
    }

    ColorPixelBufferGatherAccessor accessor;
    accessor.gatherInd = intY * this->mipLevels[0].sizes.w + intX;
    accessor.gatherMask = mask;
    accessor.buf = this;
    return accessor;
}
/*
ColorPixelBufferGatherAccessor256 Rasterizing::ColorPixelBuffer::getGatherAccessor(float32x8 u, float32x8 v, float32x8 mask) const
{
    auto [pixelsX, pixelsY] = Mapper::UV_to_XY<MappingType::WRAP>(u, v, sizes.w, sizes.h);
    __m256i intX = _mm256_cvttps_epi32(pixelsX);
    __m256i intY = _mm256_cvttps_epi32(pixelsY);

    ColorPixelBufferGatherAccessor256 accessor;
    __m256i ind = _mm256_mullo_epi32(intY, _mm256_set1_epi32(sizes.w));
    ind = _mm256_add_epi32(intX, ind);
    accessor.gatherInd = ind;
    accessor.gatherMask = _mm256_castps_si256(mask);
    accessor.buf = this;
    return accessor;
}*/

Vec4f Rasterizing::ColorPixelBuffer::getLinearIntensity(float u, float v, float du_dx, float du_dy, float dv_dx, float dv_dy) const
{
    const MipLevel& full = this->mipLevels[0];
    float du_dxp = du_dx * full.sizes.fw;
    float du_dyp = du_dy * full.sizes.fh;
    float dv_dxp = dv_dx * full.sizes.fw;
    float dv_dyp = dv_dy * full.sizes.fh;
    float ro_x = std::sqrt(du_dxp * du_dxp + dv_dxp * dv_dxp);
    float ro_y = std::sqrt(du_dyp * du_dyp + dv_dyp * dv_dyp);
    float ro = std::max(ro_x, ro_y);
    float lod = std::log2(ro);
    int targetMipLevel = std::clamp<float>(lod, 0.f, this->mipLevels.size() - 1);

    const MipLevel& target = this->mipLevels[targetMipLevel];
    auto [fx, fy] = Mapper::UV_to_XY<MappingType::WRAP>(u, v, target.sizes.fw, target.sizes.fh);
    auto [x, y] = Mapper::wrapInts(fx, fy, target.sizes.w, target.sizes.h);
    uint32_t channels = target.packedColors[y * target.sizes.w + x];
    return _mm_castsi128_ps(_mm_setr_epi32(
        std::bit_cast<int>(this->toLinearLUT_fp32[channels & 0xFF]),
        std::bit_cast<int>(this->toLinearLUT_fp32[(channels >> 8) & 0xFF]),
        std::bit_cast<int>(this->toLinearLUT_fp32[(channels >> 16) & 0xFF]),
        (channels >> 24) ? 0x3F800000 : 0)); //Force alpha to 0 if it's 0, or 1 if it's not
    /*
    BilinearInterpolationContext<float, int, Vec4f> ctx(fx, fy);
    std::array<Vec4f, 4> linear;
    for (int i = 0; i < 4; ++i)
    {
        auto [x, y] = Mapper::wrapInts(ctx.ix[i], ctx.iy[i], this->sizes.w, this->sizes.h);
        uint32_t channels = this->packedColors[y * sizes.w + x];
        linear[i] = _mm_castsi128_ps(_mm_setr_epi32(
            std::bit_cast<int>(this->toLinearLUT_fp32[channels & 0xFF]),
            std::bit_cast<int>(this->toLinearLUT_fp32[(channels >> 8) & 0xFF]),
            std::bit_cast<int>(this->toLinearLUT_fp32[(channels >> 16) & 0xFF]),
            (channels >> 24) ? 0x3F800000 : 0)); //Force alpha to 0 if it's 0, or 1 if it's not
    }
    return ctx.interpolate(linear); //TODO: force alpha to first parent?*/
}

bool Rasterizing::ColorPixelBuffer::areAllPixelsOpaque() const
{
    return this->mipLevels[0].isFullyOpaque;
}

/*
void Rasterizing::ColorPixelBuffer::setPixelLinearIntensityUnsafe(int x, int y, float r, float g, float b, float a)
{
    //float 
}*/

Rasterizing::ColorPixelBuffer::MipLevel::MipLevel(uint32_t w, uint32_t h)
{
    this->init(w, h);
}

void Rasterizing::ColorPixelBuffer::MipLevel::init(uint32_t w, uint32_t h)
{
    uint32_t totalPixels = w * h;
    this->packedColors = std::make_unique<uint32_t[]>(totalPixels);
    this->opacityMap = std::make_unique<uint32_t[]>(totalPixels / 32 + 1);
    this->sizes = { w,h };
}
/*
void Rasterizing::ColorPixelBufferGatherAccessor::gatherLinearRGB(Vec4_f32x16& output) const
{
    int32x16 gathered = _mm512_mask_i32gather_epi32(int32x16(0).zmm, this->gatherMask, this->gatherInd.zmm, this->buf->packedColors.get(), 4);

    int32x16 r = gathered & 1023;
    int32x16 g = _mm512_srli_epi32(gathered.zmm, 10);
    g &= 2047;
    int32x16 b = _mm512_srli_epi32(gathered.zmm, 21);
    b &= 1023;

    float32x16 fr = _mm512_cvtepu32_ps(r.zmm);
    float32x16 fg = _mm512_cvtepu32_ps(g.zmm);
    float32x16 fb = _mm512_cvtepu32_ps(b.zmm);
    float32x16 fa = _mm512_maskz_mov_ps(gathered < 0, float32x16(1)); //if uppermost bit is 1 (i.e. sign bit is 1, i.e negative), then alpha is 1
    fr *= 1.f / 1023;
    fg *= 1.f / 2047;
    fb *= 1.f / 1023;
    
    output.r = fr * fr;
    output.g = fg * fg;
    output.b = fb * fb;
}
*/
float32x16 Rasterizing::ColorPixelBufferGatherAccessor::gatherA() const
{
    int32x16 gathered = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(0), this->gatherMask, (this->gatherInd >> 5).zmm, this->buf->mipLevels[0].opacityMap.get(), 4);
    int32x16 shifts = this->gatherInd & 31;
    int32x16 opacityMapValuesForPixels = gathered & (int32x16(1) << shifts);
    return _mm512_maskz_mov_ps(opacityMapValuesForPixels != 0, _mm512_set1_ps(1));
}
/*
float32x8 Rasterizing::ColorPixelBufferGatherAccessor256::gatherA() const
{
    __m256i gathered = _mm256_mask_i32gather_epi32(_mm256_set1_epi32(0), (const int*)(this->buf->opacityMap.get()), _mm256_srli_epi32(this->gatherInd, 5), this->gatherMask, 4);
    __m256i shifts = _mm256_and_si256(this->gatherInd, _mm256_set1_epi32(31));
    __m256i opacityMapValuesForPixels = _mm256_and_si256(gathered, _mm256_sllv_epi32(_mm256_set1_epi32(1), shifts));
    return _mm256_blendv_ps(_mm256_set1_ps(1), _mm256_set1_ps(0), _mm256_castsi256_ps(_mm256_cmpeq_epi32(opacityMapValuesForPixels, _mm256_set1_epi32(0))));
}

Vec4_f32x16 Rasterizing::Decoder::R10G11B10A1_gamma2_to_linear(int32x16 packed)
{
    int32x16 r = packed & 1023;
    int32x16 g = _mm512_srli_epi32(packed, 10);
    g &= 2047;
    int32x16 b = _mm512_srli_epi32(packed, 21);
    b &= 1023;

    float32x16 fr = _mm512_cvtepu32_ps(r);
    float32x16 fg = _mm512_cvtepu32_ps(g);
    float32x16 fb = _mm512_cvtepu32_ps(b);
    float32x16 fa = _mm512_maskz_mov_ps(packed < 0, float32x16(1)); //if uppermost bit is 1 (i.e. sign bit is 1, i.e negative), then alpha is 1
    fr *= 1.f / 1023;
    fg *= 1.f / 2047;
    fb *= 1.f / 1023;
    return { fr * fr, fg * fg, fb * fb, fa };
}
*/

std::pair<float, float> Rasterizing::Mapper::wrapUV(float u, float v)
{
    u -= std::floor(u); //doing floor subtraction once sometimes returns 1. Doing it twice guarantees 0 <= u < 1 for all non-nan non-inf values
    u -= std::floor(u);
    v -= std::floor(v);
    v -= std::floor(v);
    return { u,v };
}

std::pair<float32x8, float32x8> Rasterizing::Mapper::wrapUV(float32x8 u, float32x8 v)
{
    u -= _mm256_floor_ps(u); //doing floor subtraction once sometimes returns 1. Doing it twice guarantees 0 <= u < 1 for all non-nan non-inf values
    u -= _mm256_floor_ps(u);
    v -= _mm256_floor_ps(v);
    v -= _mm256_floor_ps(v);
    return { u,v };
}

std::pair<float32x16, float32x16> Rasterizing::Mapper::wrapUV(float32x16 u, float32x16 v)
{
    u -= _mm512_floor_ps(u); //doing floor subtraction once sometimes returns 1. Doing it twice guarantees 0 <= u < 1 for all non-nan non-inf values
    u -= _mm512_floor_ps(u);
    v -= _mm512_floor_ps(v);
    v -= _mm512_floor_ps(v);
    return { u,v };
}

std::pair<uint32_t, uint32_t> Rasterizing::Mapper::wrapIntsWithRcp(int a, int b, uint32_t amax, uint64_t rcp_aMax, uint32_t bmax, uint64_t rcp_bMax)
{
    return { wrapIntWithRcp(a,amax,rcp_aMax), wrapIntWithRcp(b,bmax,rcp_bMax) };
}

std::pair<uint32_t, uint32_t> Rasterizing::Mapper::wrapInts(int a, int b, uint32_t amax, uint32_t bmax)
{
    return { wrapInt(a,amax), wrapInt(b,bmax) };
}

uint32_t Rasterizing::Mapper::wrapInt(int a, uint32_t amax)
{
    int rem = a % amax;
    rem += rem >= 0 ? 0 : amax;
#ifdef REL_DBG
    if (rem < 0 || rem >= amax) throw std::runtime_error("wrapInt returned incorrect value!");
#endif
    return rem;
}

uint32_t Rasterizing::Mapper::wrapIntWithRcp(int a, uint32_t amax, uint64_t rcp_aMax)
{
    assert(rcp_aMax == (1ull << 32) / amax);
    //TODO: I believe this may return incorrect results for negative a, but it has ever thrown the sentry exception ever
    int64_t div = (int64_t(a) * rcp_aMax) >> 32;
    int rem = a - div * amax;
    rem += rem >= 0 ? 0 : amax;
#ifdef REL_DBG
    if (rem < 0 || rem >= amax) throw std::runtime_error("wrapIntWithRcp returned incorrect value!");
#endif
    assert(rem >= 0 && rem < amax);
    return rem;
}

Rasterizing::ColorPixelBuffer::Sizes::Sizes(uint32_t w, uint32_t h)
{
    this->fw = this->w = w;
    this->fh = this->h = h;
    this->rcpW = double(1) / w;
    this->rcpH = double(1) / h;
    this->float_maxSafeX = w - 1;
    this->float_maxSafeY = h - 1;
    this->rcp_maxSafeX = float(1) / this->float_maxSafeX;
    this->rcp_maxSafeY = float(1) / this->float_maxSafeY;
    this->intRcpW = (1ull << 32) / w;
    this->intRcpH = (1ull << 32) / h;
}
