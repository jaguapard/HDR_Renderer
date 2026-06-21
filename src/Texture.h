#pragma once
#include <vector>
#include <SDL3/SDL.h>
#include <array>
#include "Mappers.h"
#include "Vec.h"
#include <memory>

struct MipLevel
{
	MipLevel(uint32_t w, uint32_t h);
	std::vector<uint32_t> colors, opacityMap;
	WrappingMapper mapper;
	uint32_t getPixelRGBA32(uint32_t x, uint32_t y) const;

	Vec4_f32x16 gatherLinearIntensities(const float32x16& u, const float32x16& v, mask16d mask = 0xFFFF) const;
	float32x16 gatherA(const float32x16& u, const float32x16& v, mask16d mask = 0xFFFF) const;
};

class Texture
{
public:
	Texture(const SDL_Surface* s);
	Vec4_f32x16 gatherLinearIntensities(const float32x16& u, const float32x16& v, const mask16d& mask = 0xFFFF) const;
	float32x16 gatherA(const float32x16& u, const float32x16& v, const mask16d& mask = 0xFFFF) const;

	//Writes out corresponding data from this texture if argument is non-null.
	void QueryTexture(uint32_t* w = nullptr, uint32_t* h = nullptr, std::unique_ptr<uint32_t[]>* retRGBA32 = nullptr, int mipLevel = 0) const;
private:
	std::vector<MipLevel> mipLevels;
};