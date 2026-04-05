#include <vector>
#include <memory>
#include "../../Vec.h"
#include <SDL3/SDL.h>

namespace Rasterizing
{
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

	struct ColorPixelBufferGatherAccessor
	{
	public:
		void gatherLinearRGB(Vec4_f32x16& output) const;
		float32x16 gatherA() const;

		int32x16 gatherInd;
		Mask16 gatherMask;
		const ColorPixelBuffer* buf;
	};
	
	//TODO: remake this class into templated PixelBuffer<PixelFormat>? Or even polymorphic type? Or even wrapper around SDL surface of same format?
	class ColorPixelBuffer
	{
	public:
		ColorPixelBuffer(ColorPixelBuffer&& dying);
		ColorPixelBuffer(int w, int h); //initializes empty color buffer
		ColorPixelBuffer(const SDL_Surface* s);
		Vec4_f32x16 gatherLinearIntensities(float32x16 x, float32x16 y, Mask16 mask = 0xFFFF) const;
		//void setPixelLinearIntensityUnsafe(int x, int y, float r, float g, float b, float a);
		ColorPixelBufferGatherAccessor getGatherAccessor(float32x16 u, float32x16 v, Mask16 mask = 0xFFFF) const;

		Vec4f getLinearIntensity(float u, float v) const;
	private:
		void init(int w, int h);
		std::unique_ptr<uint32_t[]> packedColors, opacityMap;
		
		struct Sizes
		{
		int w, h;
		float fw, fh, rcpW, rcpH;
		float float_maxSafeX, float_maxSafeY, rcp_maxSafeX, rcp_maxSafeY;
		};
		Sizes sizes;

		friend class Rasterizing::TextureManager;
		friend class Rasterizing::ColorPixelBufferGatherAccessor;
	};
}