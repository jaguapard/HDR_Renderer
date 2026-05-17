#include "ColorPixelBuffer.h"
#include <stdexcept>
#include "../Threadpool.h"
#include "../helpers.h"
#include "../LUTMan.h"

ColorPixelBuffer::ColorPixelBuffer(ColorPixelBuffer&& dying) :
    packedColors(std::move(dying.packedColors)),
    opacityMap(std::move(dying.opacityMap)),
    sizes(dying.sizes),
    isFullyOpaque(dying.isFullyOpaque),
    mapper(dying.mapper)
{
}

ColorPixelBuffer::ColorPixelBuffer(uint32_t w, uint32_t h)
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


ColorPixelBufferGatherAccessor ColorPixelBuffer::getGatherAccessor(float32x16 u, float32x16 v, Mask16 mask) const
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

ColorPixelBufferGatherAccessor256 ColorPixelBuffer::getGatherAccessor(float32x8 u, float32x8 v, float32x8 mask) const
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
}

Vec4f ColorPixelBuffer::getLinearIntensity(float u, float v) const
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

bool ColorPixelBuffer::areAllPixelsOpaque() const
{
    return this->isFullyOpaque;
}

/*
void ColorPixelBuffer::setPixelLinearIntensityUnsafe(int x, int y, float r, float g, float b, float a)
{
    //float
}*/

void ColorPixelBuffer::init(uint32_t w, uint32_t h)
{
    uint32_t totalPixels = w * h;
    this->packedColors = std::make_unique<uint32_t[]>(totalPixels);
    this->opacityMap = std::make_unique<uint32_t[]>(totalPixels / 32 + 1);
    this->sizes = { w,h };
    this->mapper = WrappingMapper(w, h);
}

void ColorPixelBufferGatherAccessor::gatherLinearRGB(Vec4_f32x16& output) const
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

float32x16 ColorPixelBufferGatherAccessor::gatherA() const
{
    int32x16 gathered = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(0), this->gatherMask, (this->gatherInd >> 5).zmm, this->buf->opacityMap.get(), 4);
    int32x16 shifts = this->gatherInd & 31;
    int32x16 opacityMapValuesForPixels = gathered & (int32x16(1) << shifts);
    return _mm512_maskz_mov_ps(opacityMapValuesForPixels != 0, _mm512_set1_ps(1));
}

float32x8 ColorPixelBufferGatherAccessor256::gatherA() const
{
    __m256i gathered = _mm256_mask_i32gather_epi32(_mm256_set1_epi32(0), (const int*)(this->buf->opacityMap.get()), _mm256_srli_epi32(this->gatherInd, 5), this->gatherMask, 4);
    __m256i shifts = _mm256_and_si256(this->gatherInd, _mm256_set1_epi32(31));
    __m256i opacityMapValuesForPixels = _mm256_and_si256(gathered, _mm256_sllv_epi32(_mm256_set1_epi32(1), shifts));
    return _mm256_blendv_ps(_mm256_set1_ps(1), _mm256_set1_ps(0), _mm256_castsi256_ps(_mm256_cmpeq_epi32(opacityMapValuesForPixels, _mm256_set1_epi32(0))));
}

std::pair<uint32_t, uint32_t> Mapper::wrapIntsWithRcp(int a, int b, uint32_t amax, uint64_t rcp_aMax, uint32_t bmax, uint64_t rcp_bMax)
{
    return { wrapIntWithRcp(a,amax,rcp_aMax), wrapIntWithRcp(b,bmax,rcp_bMax) };
}

std::pair<uint32_t, uint32_t> Mapper::wrapInts(int a, int b, uint32_t amax, uint32_t bmax)
{
    return { wrapInt(a,amax), wrapInt(b,bmax) };
}

uint32_t Mapper::wrapInt(int a, uint32_t amax)
{
    int rem = a % amax;
    rem += rem >= 0 ? 0 : amax;
#ifdef REL_DBG
    if (rem < 0 || rem >= amax) throw std::runtime_error("wrapInt returned incorrect value!");
#endif
    return rem;
}

uint32_t Mapper::wrapIntWithRcp(int a, uint32_t amax, uint64_t rcp_aMax)
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

ColorPixelBuffer::Sizes::Sizes(uint32_t w, uint32_t h)
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
