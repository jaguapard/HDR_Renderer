#include "TextureManager.h"
#include <stdexcept>
#include <SDL3\SDL_image.h>
#include "../smart.h"
#include <iostream>
#include "../helpers.h"

TextureManager::TextureManager()
{
	std::lock_guard lck(this->mtx);
	constexpr int fallbackSize = 64;
	constexpr int expectedPitch = fallbackSize * 4;
	static_assert(fallbackSize % 2 == 0);

	auto sdl = Smart_Surface(SDL_CreateSurface(fallbackSize, fallbackSize, SDL_PIXELFORMAT_ABGR8888));
	if (!sdl) throw std::runtime_error("Failed to create SDL surface for fallback texture");
	if (sdl->pitch != expectedPitch) throw std::runtime_error("Unsupported SDL pitch for fallback texture: " + std::to_string(sdl->pitch) + " expected " + std::to_string(expectedPitch));
	int pixelCount = fallbackSize * fallbackSize;
	uint32_t* pixels = (uint32_t*)(sdl->pixels);
	for (int y = 0; y < fallbackSize; ++y)
	{
		for (int x = 0; x < fallbackSize; ++x)
		{
			int _xor = (x / (fallbackSize / 2)) ^ (y / (fallbackSize / 2));
			pixels[y * fallbackSize + x] = _xor ? 0xFFFF00FF : 0xFF000000;
		}
	}

	int h = this->addTextureBySurface(sdl.get());
	if (h != FALLBACK_HANDLE) throw std::runtime_error("Unexpected handle for fallback texture: " + std::to_string(h) + ", expected 0.");
}
int TextureManager::addTextureBySurface(SDL_Surface* s)
{
	auto converted = Smart_Surface(SDL_ConvertSurface(s, SDL_PIXELFORMAT_RGBA32));
	if (!converted)
	{
		std::lock_guard lck(this->mtx);
		std::cout << "Unable to convert surface to RGBA32 format " << ": " << SDL_GetError() << "\n";
		return 0;
	}

	Texture buf = converted.get();
	std::lock_guard lck(this->mtx);
	this->textures.emplace_back(std::move(buf));

	/*
	this->bufferForTexture.clear();
	this->sizesForTexture.clear();
	for (auto& it : this->textures)
	{
		this->bufferForTexture.emplace_back(it.packedColors.get());
		this->sizesForTexture.emplace_back(it.sizes);
	}*/
	return this->textures.size() - 1;
}
int TextureManager::addTextureByPath(std::string path)
{
	auto found = this->pathToIndexMap.find(path);
	if (found != this->pathToIndexMap.end())
	{
		std::lock_guard lck(this->mtx);
		std::cout << "Texture at " << path << " was already loaded, returning existing index " << found->second << "\n";
		return found->second;
	}

	auto initialSurf = Smart_Surface(IMG_Load(path.c_str()));
	if (!initialSurf)
	{
		std::lock_guard lck(this->mtx);
		//throw std::runtime_error("Unable to open texture at " + path);
		std::cout << "Unable to open texture at " << path << ", using fallback!\n";
		return 0;
	}

	int h = this->addTextureBySurface(initialSurf.get());
	if (h == FALLBACK_HANDLE)
	{
		std::lock_guard lck(this->mtx);
		std::cout << "An error occurred while adding texture from " << path << " by surface, using fallback!\n";
		return 0;
	}

	//std::cout << "Successfully loaded texture from " << path << " and assigned it index " << h << "\n";
	this->pathToIndexMap[path] = h;
	return h;
}

const Texture& TextureManager::getTextureByHandle(int i) const
{
	return this->textures[i];
}

bool TextureManager::handleIsValid(int h) const
{
	return h != INVALID_HANDLE;
}

void TextureManager::clear()
{
	std::lock_guard lck(this->mtx);
	//since buffers are annoying with unique pointers, resize is not possible. We also have to keep fallback texture untouched. 
	//Thus, just pop until there's only fallback remaining
	while (this->textures.size() > 1) this->textures.pop_back();/*
	while (this->bufferForTexture.size() > 1) this->bufferForTexture.pop_back();
	while (this->sizesForTexture.size() > 1) this->sizesForTexture.pop_back();*/
}

Vec4_f32x16 TextureManager::gatherLinearIntesitiesFromMultipleTextures(const int32x16& textureInd, const float32x16& u, const float32x16& v, const Mask16& mask) const
{
	Vec4_f32x16 texturePixels = 0.f;
	int32x16 uniqueDiffuseMapIndices;
	uint32_t uniqueCount;
	deduplicate_epi32x16(textureInd, TextureManager::INVALID_HANDLE, mask, uniqueDiffuseMapIndices, &uniqueCount);
	for (uint32_t j = 0; j < uniqueCount; ++j) //TODO: can try to make this fixed-size loop so Clang can optimize memory reads to extracts from uniqueDiffuseMapIndices
	{
		int currDiffuseMapIndex = uniqueDiffuseMapIndices[j];
		Mask16 thisTextureMask = mask & (textureInd == currDiffuseMapIndex);
		Vec4_f32x16 gathered = this->getTextureByHandle(currDiffuseMapIndex).gatherLinearIntensities(u, v, thisTextureMask);
		for (int k = 0; k < 4; ++k) texturePixels[k] = _mm512_mask_mov_ps(texturePixels[k], thisTextureMask, gathered[k]);
		/*
		if (Statsman::ENABLED)
		{
			MyStatsman.rasterizing.textureGatheredLanes += 16;
			MyStatsman.rasterizing.textureGatherAliveLanes += _mm_popcnt_u32(thisTextureMask);
		}*/
	}
	return texturePixels;
}

float32x16 TextureManager::gatherAlphaFromMultipleTextures(const int32x16& textureInd, const float32x16& u, const float32x16& v, const Mask16& mask) const
{
	float32x16 ret = 0.f;
	int32x16 uniqueDiffuseMapIndices;
	uint32_t uniqueCount;
	deduplicate_epi32x16(textureInd, TextureManager::INVALID_HANDLE, mask, uniqueDiffuseMapIndices, &uniqueCount);
	for (uint32_t j = 0; j < uniqueCount; ++j) //TODO: can try to make this fixed-size loop so Clang can optimize memory reads to extracts from uniqueDiffuseMapIndices
	{
		int currDiffuseMapIndex = uniqueDiffuseMapIndices[j];
		Mask16 thisTextureMask = mask & (textureInd == currDiffuseMapIndex);
		float32x16 gathered = this->getTextureByHandle(currDiffuseMapIndex).gatherA(u, v, thisTextureMask);
		ret =_mm512_mask_mov_ps(ret, thisTextureMask, gathered);
		/*
		if (Statsman::ENABLED)
		{
			MyStatsman.rasterizing.textureGatheredLanes += 16;
			MyStatsman.rasterizing.textureGatherAliveLanes += _mm_popcnt_u32(thisTextureMask);
		}*/
	}
	return ret;
}
