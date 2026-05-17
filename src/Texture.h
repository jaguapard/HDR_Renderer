#pragma once
#include <vector>
#include <SDL3/SDL.h>
#include <array>
#include "Mappers.h"
#include "Vec.h"

struct MipLevel
{
	MipLevel(uint32_t w, uint32_t h);
	std::vector<uint32_t> colors;
	WrappingMapper mapper;

	Vec4_f32x16 gatherLinearIntensities(float32x16 u, float32x16 v, Mask16 mask = 0xFFFF) const;
};

class Texture
{
public:
	Texture(const SDL_Surface* s);

private:
	std::vector<MipLevel> mipLevels;
};