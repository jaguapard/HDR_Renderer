#pragma once
#include <vector>
#include <SDL3/SDL.h>
#include <array>

constexpr static std::array<int16_t, 256> toLinear_FP16 = []() {
	std::array<int16_t, 256> result;
	for (int i = 0; i < 256; ++i)
	{
		float f32 = pow(i / 255.0, 2.2);
		result[i] = _mm_extract_epi16(_mm_cvtps_ph(_mm_set1_ps(f32), _MM_FROUND_TO_NEAREST_INT), 0);
	}
	return result;
	}();
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