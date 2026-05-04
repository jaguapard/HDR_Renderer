#pragma once
#include <vector>
#include <memory>
#include "../Vec.h"
#include <SDL3/SDL.h>
#include <array>
#include "../helpers.h"

/*
enum class Channel
{
	RED, GREEN, BLUE, ALPHA;
};

template<>
struct Accessor
{
	float fw, fh;

};*/
class ColorPixelBuffer;
class TextureManager;

enum class MappingType
{
	WRAP, //UVs are forced into 0 <= U,V < 1 range
	PLAIN, //UVs are not affected, users are responsible for bounds checks
	CLAMP, //UVs are clamped. Values below 0 are forced to 0, values >= 1 are forced to last pixel
};

struct Mapper
{
	static std::pair<float, float> wrapUV(float u, float v);
	static std::pair<float32x8, float32x8> wrapUV(float32x8 u, float32x8 v);
	static std::pair<float32x16, float32x16> wrapUV(float32x16 u, float32x16 v);

	//wraps integers a and b into range 0 <= a < amax, 0 <= b < bmax
	static std::pair<uint32_t, uint32_t> wrapIntsWithRcp(int a, int b, uint32_t amax, uint64_t rcp_aMax, uint32_t bmax, uint64_t rcp_bMax);

	//wraps integers a and b into range 0 <= a < amax, 0 <= b < bmax
	static std::pair<uint32_t, uint32_t> wrapInts(int a, int b, uint32_t amax, uint32_t bmax);

	//wraps integers a into range 0 <= a < amax
	static uint32_t wrapInt(int a, uint32_t amax);

	//uses special value provided to replace division with multiplication and shift
	static uint32_t wrapIntWithRcp(int a, uint32_t amax, uint64_t rcp_aMax);

	template <MappingType M, typename T>
	static std::pair<T, T> UV_to_XY(T u, T v, float w, float h)
	{
		if constexpr (M == MappingType::WRAP)
		{
			auto [wu, wv] = wrapUV(u, v);
			return { wu * w, wv * h };
		}

		if constexpr (M == MappingType::CLAMP)
		{
			T x = u * w;
			T y = v * h;
			if constexpr (std::is_same_v<T, float32x16>)
			{
				return { x.clamp(0,w - 1), y.clamp(0,h - 1) };
			}
			if constexpr (std::is_same_v<T, float>)
			{
				return { std::clamp(x,0,w - 1), std::clamp(y,0,h - 1) };
			}
		}

		if constexpr (M == MappingType::PLAIN)
		{
			return { u * w, v * h };
		}
	}
};

struct Swizzler
{
	//static void 
};

template<typename FloatType, typename SignedIntType, typename InterpolandType>
struct BilinearInterpolationContext
{
	std::array<SignedIntType, 4> ix, iy;
	FloatType tx, ty;

	BilinearInterpolationContext(FloatType x, FloatType y)
	{
		/*
		static_assert(std::is_signed_v<SignedIntType>, "Bilinear interpolation context requires a signed int type.");
		static_assert(std::is_integral_v<SignedIntType>);
		static_assert(std::is_floating_point_v<FloatType>);*/
		if constexpr (std::is_same_v<FloatType, float>)
		{
			this->tx = std::fmod(x, 1.f);
			this->ty = std::fmod(y, 1.f);
			for (int oy = 0; oy < 2; ++oy)
			{
				for (int ox = 0; ox < 2; ++ox)
				{
					int i = oy * 2 + ox;
					this->ix[i] = int(x) + ox;
					this->iy[i] = int(y) + oy;
				}
			}
		}
		else if constexpr (std::is_same_v<FloatType, float32x16> && std::is_same_v<SignedIntType, int32x16>)
		{
			this->tx = x - _mm512_floor_ps(x);
			this->ty = y - _mm512_floor_ps(y);
			for (int oy = 0; oy < 2; ++oy)
			{
				for (int ox = 0; ox < 2; ++ox)
				{
					int i = oy * 2 + ox;
					this->ix[i] = x.trunc() + ox;
					this->iy[i] = y.trunc() + oy;
				}
			}
		}
		else static_assert(false, "Unsupported types for BilinearInterpolationContext");
	}

	InterpolandType interpolate(const std::array<InterpolandType, 4>& interpolands) const
	{
		InterpolandType ler1 = lerp(interpolands[0], interpolands[1], tx);
		InterpolandType ler2 = lerp(interpolands[2], interpolands[3], tx);
		InterpolandType ret = lerp(ler1, ler2, ty);
		//TODO: alpha adjustment. Copy from original pixel?
		return ret;
	}
};

struct Decoder
{
	static Vec4_f32x16 R10G11B10A1_gamma2_to_linear(int32x16 packed);
};

struct ColorPixelBufferGatherAccessor
{
public:
	void gatherLinearRGB(Vec4_f32x16& output) const;
	float32x16 gatherA() const;

	int32x16 gatherInd;
	Mask16 gatherMask;
	const ColorPixelBuffer* buf;
};
	
struct ColorPixelBufferGatherAccessor256
{
public:
	void gatherLinearRGB(Vec4_f32x8& output) const;
	float32x8 gatherA() const;

	__m256i gatherInd, gatherMask;
	const ColorPixelBuffer* buf;
};
	
//TODO: remake this class into templated PixelBuffer<PixelFormat>? Or even polymorphic type? Or even wrapper around SDL surface of same format?
class ColorPixelBuffer
{
public:
	ColorPixelBuffer(ColorPixelBuffer&& dying);
	ColorPixelBuffer(uint32_t w, uint32_t h); //initializes empty color buffer
	ColorPixelBuffer(const SDL_Surface* s);
	Vec4_f32x16 gatherLinearIntensities(float32x16 x, float32x16 y, Mask16 mask = 0xFFFF) const;
	//void setPixelLinearIntensityUnsafe(int x, int y, float r, float g, float b, float a);
	ColorPixelBufferGatherAccessor getGatherAccessor(float32x16 u, float32x16 v, Mask16 mask = 0xFFFF) const;
	ColorPixelBufferGatherAccessor256 getGatherAccessor(float32x8 u, float32x8 v, float32x8 mask) const;

	Vec4f getLinearIntensity(float u, float v) const;
	bool areAllPixelsOpaque() const;
private:
	void init(uint32_t w, uint32_t h);
	std::unique_ptr<uint32_t[]> packedColors, opacityMap;
	bool isFullyOpaque = true;
		
	static inline std::unique_ptr<float[]> toLinearLUT_fp32;
	static inline std::unique_ptr<int16_t[]> toLinearLUT_fp16;
	struct Sizes
	{
		Sizes() {};
		Sizes(uint32_t w, uint32_t h);
		uint32_t w, h;
		uint64_t intRcpW, intRcpH; //2^32 / w or h to replace division by multiplication and shift
		float fw, fh, rcpW, rcpH;
		float float_maxSafeX, float_maxSafeY, rcp_maxSafeX, rcp_maxSafeY;
	};
	Sizes sizes;

	friend class TextureManager;
	friend class ColorPixelBufferGatherAccessor;
	friend class ColorPixelBufferGatherAccessor256;
};
