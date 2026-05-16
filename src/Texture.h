#pragma once
#include <vector>
#include <SDL3/SDL.h>
#include <array>

struct MipLevel
{
	std::vector<uint32_t> colors;
	uint32_t w, h;
};

class Texture
{
public:
	Texture(const SDL_Surface* s);

private:
	std::vector<MipLevel> mipLevels;
};