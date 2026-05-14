#include "TextureManager.h"
#include <stdexcept>
#include <SDL3\SDL_image.h>
#include "../smart.h"
#include <iostream>

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
	if (h != 0) throw std::runtime_error("Unexpected handle for fallback texture: " + std::to_string(h) + ", expected 0.");
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

	ColorPixelBuffer buf = converted.get();
	std::lock_guard lck(this->mtx);
	this->textures.emplace_back(std::move(buf));

	this->bufferForTexture.clear();
	this->sizesForTexture.clear();
	for (auto& it : this->textures)
	{
		this->bufferForTexture.emplace_back(it.packedColors.get());
		this->sizesForTexture.emplace_back(it.sizes);
	}
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
	if (h == 0)
	{
		std::lock_guard lck(this->mtx);
		std::cout << "An error occurred while adding texture from " << path << " by surface, using fallback!\n";
		return 0;
	}

	//std::cout << "Successfully loaded texture from " << path << " and assigned it index " << h << "\n";
	this->pathToIndexMap[path] = h;
	return h;
}

const ColorPixelBuffer& TextureManager::getTextureByHandle(int i) const
{
	return this->textures[i];
}

bool TextureManager::handleIsValid(int h) const
{
	return h >= 0;
}

void TextureManager::clear()
{
	std::lock_guard lck(this->mtx);
	//since buffers are annoying with unique pointers, resize is not possible. We also have to keep fallback texture untouched. 
	//Thus, just pop until there's only fallback remaining
	while (this->textures.size() > 1) this->textures.pop_back();
	while (this->bufferForTexture.size() > 1) this->bufferForTexture.pop_back();
	while (this->sizesForTexture.size() > 1) this->sizesForTexture.pop_back();
}
Vec4_f32x16 TextureManager::gatherLinearIntensitiesFromMultipleTextures(int32x16 textureIndices, float32x16 u, float32x16 v, Mask16 mask) const
{
	__m512i ptrs0_7 = _mm512_mask_i64gather_epi64(_mm512_setzero_si512(), mask, _mm512_cvtepu32_epi64(_mm512_extracti32x8_epi32(textureIndices, 0)), this->bufferForTexture.data(), 8);
	__m512i ptrs8_15 = _mm512_mask_i64gather_epi64(_mm512_setzero_si512(), mask >> 8, _mm512_cvtepu32_epi64(_mm512_extracti32x8_epi32(textureIndices, 1)), this->bufferForTexture.data(), 8);

	u -= _mm512_floor_ps(u);
	v -= _mm512_floor_ps(v);
	float32x16 float_maxSafeX = _mm512_mask_i32gather_ps(float32x16(0.f), mask, textureIndices * sizeof(ColorPixelBuffer::Sizes) + offsetof(ColorPixelBuffer::Sizes, float_maxSafeX), this->sizesForTexture.data(), 1);
	float32x16 float_maxSafeY = _mm512_mask_i32gather_ps(float32x16(0.f), mask, textureIndices * sizeof(ColorPixelBuffer::Sizes) + offsetof(ColorPixelBuffer::Sizes, float_maxSafeY), this->sizesForTexture.data(), 1);
	int32x16 w = _mm512_mask_i32gather_epi32(int32x16(0), mask, textureIndices * sizeof(ColorPixelBuffer::Sizes) + offsetof(ColorPixelBuffer::Sizes, w), this->sizesForTexture.data(), 1);
	float32x16 pixelsX = u * float_maxSafeX;
	float32x16 pixelsY = v * float_maxSafeY;
	int32x16 pixelOffset = (pixelsY.trunc() * w + int32x16(pixelsX.trunc())) << 2;

	__m512i finAddrs0_7 = _mm512_add_epi64(ptrs0_7, _mm512_cvtepu32_epi64(_mm512_extracti32x8_epi32(pixelOffset, 0)));
	__m512i finAddrs8_15 = _mm512_add_epi64(ptrs8_15, _mm512_cvtepu32_epi64(_mm512_extracti32x8_epi32(pixelOffset, 1)));

	__m256i gathered_low = _mm512_mask_i64gather_epi32(_mm256_setzero_si256(), mask, finAddrs0_7, nullptr, 1);
	__m256i gathered_high = _mm512_mask_i64gather_epi32(_mm256_setzero_si256(), mask >> 8, finAddrs8_15, nullptr, 1);
	int32x16 gathered = _mm512_inserti32x8(_mm512_castsi256_si512(gathered_low), gathered_high, 1);

	int32x16 r = gathered & 1023;
	int32x16 g = _mm512_srli_epi32(gathered, 10);
	g &= 2047;
	int32x16 b = _mm512_srli_epi32(gathered, 21);
	b &= 1023;

	float32x16 fr = _mm512_cvtepu32_ps(r);
	float32x16 fg = _mm512_cvtepu32_ps(g);
	float32x16 fb = _mm512_cvtepu32_ps(b);
	float32x16 fa = _mm512_maskz_mov_ps(gathered < 0, float32x16(1)); //if uppermost bit is 1 (i.e. sign bit is 1, i.e negative), then alpha is 1
	fr *= 1.f / 1023;
	fg *= 1.f / 2047;
	fb *= 1.f / 1023;
	return { fr * fr, fg * fg, fb * fb, fa };
}
