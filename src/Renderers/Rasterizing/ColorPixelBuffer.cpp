#include "ColorPixelBuffer.h"
#include <stdexcept>

Rasterizing::ColorPixelBuffer::ColorPixelBuffer(int w, int h)
{
    this->init(w, h);
}

Rasterizing::ColorPixelBuffer::ColorPixelBuffer(const SDL_Surface* s)
{
    int w = s->w;
    int h = s->h;

    if (s->format == SDL_PIXELFORMAT_RGBA32)
    {
        const uint32_t* srcPixels = std::bit_cast<uint32_t*>(s->pixels);
        this->init(w, h);
        for (int y = 0; y < h; ++y)
        {
            const uint32_t* srcRow = std::bit_cast<const uint32_t*>(size_t(srcPixels) + s->pitch * y);
            for (int x = 0; x < w; ++x)
            {
                //TODO: annoyingly slow, make startup very long. MT this this lmao?
                //reencode gamma 2.2 texture into gamma 2 with expanded precision for internal use. Gamma 2 greatly simplifies texture->linear conversion (x*x instead of x^2.2),
                //meaning you can just use multiplication instead of power, much faster, and error is >1% mostly, more on very dark shades.
                //alpha is binary, all values above 0 considered fully opaque. TODO: when implementing transparency, change this
                uint32_t srcUint32 = srcRow[x];
                double linearR = pow(double(srcUint32 & 0xFF) / 255, 2.2);
                double linearG = pow(double((srcUint32 >> 8) & 0xFF)/255, 2.2);
                double linearB = pow(double((srcUint32 >> 16) & 0xFF)/255, 2.2);
                //double a = (srcUint32 >> 24);// / 255;

                //TODO: Lift blacks? I.e. lifting blacks to not be black hole level (0% reflectivity), but reasonable IRL black (~1-3% reflectivity)
                double encodedR = round(sqrt(linearR) * 1023);
                double encodedG = round(sqrt(linearG) * 2047);
                double encodedB = round(sqrt(linearB) * 1023);

                encodedR = std::clamp<double>(encodedR, 0, 1023);
                encodedG = std::clamp<double>(encodedG, 0, 2047);
                encodedB = std::clamp<double>(encodedB, 0, 1023);

                //R10G11B10A1 format
                uint32_t storeR = encodedR, storeG = encodedG, storeB = encodedB, storeA = (srcUint32 >> 24) ? 1 : 0;
                uint32_t storeUint = (storeR) | (storeG << 10) | (storeB << 21) | (storeA << 31);
                this->packedColors[y * w + x] = storeUint;
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

Vec4_f32x16 Rasterizing::ColorPixelBuffer::gatherLinearIntensities(float32x16 x, float32x16 y, Mask16 mask) const
{
    /*
    __m512d xd_low = _mm512_cvtpslo_pd(x);
    __m512d xd_high = _mm512_cvtps_pd(_mm512_extractf32x8_ps(x, 1));
    __m512d yd_low = _mm512_cvtpslo_pd(y);
    __m512d yd_high = _mm512_cvtps_pd(_mm512_extractf32x8_ps(y, 1));*/
    //clamp out of bounds. These can sometimes still return one, but there's some margin, since maxSafeX/Y are 1 less than width and height
    x -= _mm512_floor_ps(x);
    y -= _mm512_floor_ps(y);
    //x = _mm512_fmod_ps(x, float32x16(1));
    //y = _mm512_fmod_ps(y, float32x16(1));
    float32x16 pixelsX = x * float_maxSafeX;
    float32x16 pixelsY = y * float_maxSafeY;
    //Mask16 inBoundsMask = x >= 0.f & x < 1.f & y >= 0.f & y < 1.f;
    int32x16 intX = pixelsX.trunc();
    int32x16 intY = pixelsY.trunc();
    for (int i = 0; i < 16; ++i)
    {
        assert(intX[i] >= 0 && intX[i] < w);
        assert(intY[i] >= 0 && intY[i] < h);
    }
    int32x16 pixelsIndices = intY * w + intX;
    Mask16 gatherMask = mask;
    int32x16 gathered = _mm512_mask_i32gather_epi32(int32x16(0), gatherMask, pixelsIndices, this->packedColors.get(), 4);

    int32x16 r = gathered & 1023;
    /*
    int32x16 g = _mm512_srli_epi32(gathered, 10);
    g &= 2047;
    int32x16 b = _mm512_srli_epi32(gathered, 21);
    b &= 1023;
    int32x16 a = _mm512_srli_epi32(gathered, 31);*/
    int32x16 g = (gathered >> 10) & 2047;
    int32x16 b = (gathered >> 21) & 1023;

    float32x16 fr = _mm512_cvtepu32_ps(r);
    float32x16 fg = _mm512_cvtepu32_ps(g);
    float32x16 fb = _mm512_cvtepu32_ps(b);
    float32x16 fa = _mm512_maskz_mov_ps(gathered < 0, float32x16(1)); //if uppermost bit is 1 (i.e. sign bit is 1, i.e negative), then alpha is 1
    fr *= 1.f/1023;
    fg *= 1.f/2047;
    fb *= 1.f/1023;
    return { fr * fr, fg * fg, fb * fb, fa };
    /*
    Vec4_f32x16 ret = { ,_mm512_cvtepu32_ps(g), _mm512_cvtepu32_ps(b), _mm512_cvtepu32_ps(a) };
    ret.x = _mm512_pow_ps(ret.x/255, float32x16(2.2));
    ret.g = _mm512_pow_ps(ret.y/255, float32x16(2.2));
    ret.b = _mm512_pow_ps(ret.z/255, float32x16(2.2));
    ret.a /= 255;
    return ret;*/
}

/*
void Rasterizing::ColorPixelBuffer::setPixelLinearIntensityUnsafe(int x, int y, float r, float g, float b, float a)
{
    //float 
}*/

void Rasterizing::ColorPixelBuffer::init(int w, int h)
{
    int totalPixels = w * h;
    this->packedColors = std::make_unique<uint32_t[]>(totalPixels);
    this->fw = this->w = w;
    this->fh = this->h = h;
    this->rcpW = double(1) / w;
    this->rcpH = double(1) / h;
    this->float_maxSafeX = w - 1;
    this->float_maxSafeY = h - 1;
    this->rcp_maxSafeX = float(1) / float_maxSafeX;
    this->rcp_maxSafeY = float(1) / float_maxSafeY;

}
