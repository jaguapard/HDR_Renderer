#include "ColorPixelBuffer.h"
#include <stdexcept>
#include "../../Threadpool.h"
#include "../../helpers.h"
using namespace Rasterizing;

Rasterizing::ColorPixelBuffer::ColorPixelBuffer(ColorPixelBuffer&& dying) :
    packedColors(std::move(dying.packedColors)),
    opacityMap(std::move(dying.opacityMap)),
    sizes(dying.sizes),
    isFullyOpaque(dying.isFullyOpaque)
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

Rasterizing::ColorPixelBuffer::ColorPixelBuffer(const SDL_Surface* s)
{
    int w = s->w;
    int h = s->h;

    if (s->format == SDL_PIXELFORMAT_RGBA32)
    {
        const uint32_t* srcPixels = std::bit_cast<uint32_t*>(s->pixels);
        this->init(w, h);
        
        std::vector<task_id> tasks;
        int tCount = Threadpool::instance->getThreadCount();
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
                        //reencode gamma 2.2 texture into gamma 2 with expanded precision for internal use. Gamma 2 greatly simplifies texture->linear conversion (x*x instead of x^2.2),
                        //meaning you can just use multiplication instead of power, much faster, and error is >1% mostly, more on very dark shades.
                        //alpha is binary, all values above 0 considered fully opaque. TODO: when implementing transparency, change this
                        Mask16 boundsMask = (int32x16::sequence() + x) < w;
                        int32x16 srcUint32 = _mm512_maskz_loadu_epi32(boundsMask, srcRow + x);

                        int32x16 srcR = srcUint32 & 0xFF;
                        int32x16 srcG = (srcUint32 >> 8) & 0xFF;
                        int32x16 srcB = (srcUint32 >> 16) & 0xFF;
                        float32x16 floatR = _mm512_mul_ps(_mm512_cvtepu32_ps(srcR.zmm), float32x16(1.f / 255));
                        float32x16 floatG = _mm512_mul_ps(_mm512_cvtepu32_ps(srcG.zmm), float32x16(1.f / 255));
                        float32x16 floatB = _mm512_mul_ps(_mm512_cvtepu32_ps(srcB.zmm), float32x16(1.f / 255));
                        float32x16 encodedR, encodedG, encodedB;
                        for (int i = 0; i < 16; ++i)
                        {
                            encodedR[i] = powf(floatR[i], 1.1) * 1023;
                            encodedG[i] = powf(floatG[i], 1.1) * 2047;
                            encodedB[i] = powf(floatB[i], 1.1) * 1023;
                        }

                        encodedR = encodedR.clamp(0, 1023);
                        encodedG = encodedG.clamp(0, 2047);
                        encodedB = encodedB.clamp(0, 1023);

                        int32x16 dstR = _mm512_cvtps_epi32(encodedR);
                        int32x16 dstG = _mm512_cvtps_epi32(encodedG);
                        int32x16 dstB = _mm512_cvtps_epi32(encodedB);
                        int32x16 dstFull = dstR | (dstG << 10) | (dstB << 21);// | 0x80000000; //storeA = (srcUint32 >> 24) ? 1 : 0;
                        dstFull = _mm512_mask_or_epi32(dstFull.zmm, _mm512_cmpgt_epu32_mask(srcUint32.zmm, _mm512_set1_epi32(0x00FFFFFF)), dstFull.zmm, int32x16(0x80000000).zmm);

                        //R10G11B10A1
                        _mm512_mask_storeu_epi32(&this->packedColors[y * w + x], boundsMask, dstFull.zmm);
                        //this->packedColors[y * w + x] = storeUint;
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
            int32x16 packed1 = _mm512_maskz_loadu_epi32(boundsMask1, this->packedColors.get() + i);
            int32x16 packed2 = _mm512_maskz_loadu_epi32(boundsMask2, this->packedColors.get() + i + 16);
            Mask16 m1 = (packed1 & int32x16(0x80000000)) != 0;
            Mask16 m2 = (packed2 & int32x16(0x80000000)) != 0;
            uint32_t mt = (uint32_t(m2) << 16) | m1;
            this->opacityMap[i / 32] = mt;
            if (~mt) this->isFullyOpaque = false;
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

Rasterizing::ColorPixelBufferGatherAccessor Rasterizing::ColorPixelBuffer::getGatherAccessor(float32x16 u, float32x16 v, Mask16 mask) const
{
    auto [pixelsX, pixelsY] = Mapper::UV_to_XY<MappingType::WRAP>(u, v, sizes.w, sizes.h);
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
    accessor.gatherInd = intY * sizes.w + intX;
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
    accessor.gatherMask = mask;
    accessor.buf = this;
    return accessor;
}*/

Vec4f Rasterizing::ColorPixelBuffer::getLinearIntensity(float u, float v) const
{
    auto [fx, fy] = Mapper::UV_to_XY<MappingType::WRAP>(u, v, sizes.fw, sizes.fh);
    BilinearInterpolationContext<float, int, Vec4f> ctx(fx, fy);
    std::array<Vec4f, 4> linear;
    for (int i = 0; i < 4; ++i)
    {
        auto [x, y] = Mapper::wrapInts(ctx.ix[i], ctx.iy[i], this->sizes.w, this->sizes.h);
        __m128i channels = _mm_set1_epi32(this->packedColors[y * sizes.w + x]);
        channels = _mm_srlv_epi32(channels, _mm_setr_epi32(0, 10, 21, 31));
        channels = _mm_and_si128(channels, _mm_setr_epi32(1023, 2047, 1023, 1));
        Vec4f gammaEncodedChannels = _mm_cvtepu32_ps(channels);
        Vec4f normalized = gammaEncodedChannels * Vec4f(_mm_setr_ps(1.f / 1023, 1.f / 2047, 1.f / 1023, 1)); //TODO: alpha will get messed up if it's not 0 or 1!
        linear[i] = normalized * normalized;
    }
    return ctx.interpolate(linear);
}

bool Rasterizing::ColorPixelBuffer::areAllPixelsOpaque() const
{
    return this->isFullyOpaque;
}

/*
void Rasterizing::ColorPixelBuffer::setPixelLinearIntensityUnsafe(int x, int y, float r, float g, float b, float a)
{
    //float 
}*/

void Rasterizing::ColorPixelBuffer::init(uint32_t w, uint32_t h)
{
    uint32_t totalPixels = w * h;
    this->packedColors = std::make_unique<uint32_t[]>(totalPixels);
    this->opacityMap = std::make_unique<uint32_t[]>(totalPixels / 32 + 1);
    this->sizes = { w,h };
}

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

float32x16 Rasterizing::ColorPixelBufferGatherAccessor::gatherA() const
{
    int32x16 gathered = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(0), this->gatherMask, (this->gatherInd >> 5).zmm, this->buf->opacityMap.get(), 4);
    int32x16 shifts = this->gatherInd & 31;
    int32x16 opacityMapValuesForPixels = gathered & (int32x16(1) << shifts);
    return _mm512_maskz_mov_ps(opacityMapValuesForPixels != 0, _mm512_set1_ps(1));
}
/*
float32x8 Rasterizing::ColorPixelBufferGatherAccessor256::gatherA() const
{
    __m256i gathered = _mm256_mask_i32gather_epi32(_mm256_set1_epi32(0), this->buf->opacityMap.get(), _mm256_srli_epi32(this->gatherInd, 5), this->gatherMask, 4);
    __m256i shifts = _mm256_and_si256(this->gatherInd, _mm256_set1_epi32(31));
    __m256i opacityMapValuesForPixels = _mm256_and_si256(gathered, _mm256_sllv_epi32(_mm256_set1_epi32(1), shifts));
    return _mm256_blendv_ps(_mm256_set1_ps(1), _mm256_set1_ps(0), _mm256_cmpeq_epi32(opacityMapValuesForPixels, _mm256_set1_epi32(0)));
}*/

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
