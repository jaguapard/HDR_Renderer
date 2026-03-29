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
	/*
	class ColorPixelBufferGatherAccessor
	{
	public:

	private:
	};*/
	class TextureManager;

	//TODO: remake this class into templated PixelBuffer<PixelFormat>? Or even polymorphic type? Or even wrapper around SDL surface of same format?
	class ColorPixelBuffer
	{
	public:
		ColorPixelBuffer(int w, int h); //initializes empty color buffer
		ColorPixelBuffer(const SDL_Surface* s);
		Vec4_f32x16 gatherLinearIntensities(float32x16 x, float32x16 y, Mask16 mask = 0xFFFF) const;
		//void setPixelLinearIntensityUnsafe(int x, int y, float r, float g, float b, float a);
	private:
		void init(int w, int h);
		std::unique_ptr<uint32_t[]> packedColors;
		int w, h;
		float fw, fh, rcpW, rcpH;
		float float_maxSafeX, float_maxSafeY, rcp_maxSafeX, rcp_maxSafeY;

		friend class Rasterizing::TextureManager;
	};
}