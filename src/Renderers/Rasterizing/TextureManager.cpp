#include "TextureManager.h"
#include <stdexcept>
#include <SDL3\SDL_image.h>
#include "../../smart.h"
#include <iostream>

using namespace Rasterizing;
Rasterizing::TextureManager::TextureManager()
{
	constexpr int fallbackSize = 64;
	static_assert(fallbackSize % 2 == 0);
	auto& fallbackTexture = this->textures.emplace_back(fallbackSize, fallbackSize);
	for (int y = 0; y < fallbackSize; ++y)
	{
		for (int x = 0; x < fallbackSize; ++x)
		{
			int _xor = (x / (fallbackSize / 2)) ^ (y / (fallbackSize / 2));
			fallbackTexture.packedColors[y * fallbackSize + x] = _xor ? 0xFFFF00FF : 0xFF000000;
		}
	}
}
int Rasterizing::TextureManager::addTextureByPath(std::string path)
{
	auto initialSurf = Smart_Surface(IMG_Load(path.c_str()));
	if (!initialSurf)
	{
		//throw std::runtime_error("Unable to open texture at " + path);
		std::cout << "Unable to open texture at " << path << ", using fallback!\n";
		return 0;
	}
		
	auto converted = Smart_Surface(SDL_ConvertSurface(initialSurf.get(), SDL_PIXELFORMAT_RGBA32));
	if (!converted)
	{
		std::cout << "Unable to convert surface to RGBA32 format " << path << ": " << SDL_GetError() << ", using fallback!\n";
		return 0;
		//throw std::runtime_error("Unable to convert surface to ABGR128 format " + path);
	}

	ColorPixelBuffer buf = converted.get();
	std::lock_guard lck(this->mtx);
	this->textures.emplace_back(std::move(buf));
	return this->textures.size() - 1;	
}

const Rasterizing::ColorPixelBuffer& Rasterizing::TextureManager::getTextureByHandle(int i) const
{
	return this->textures[i];
}

bool Rasterizing::TextureManager::handleIsValid(int h) const
{
	return h >= 0;
}
