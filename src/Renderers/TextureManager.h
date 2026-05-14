#pragma once
#include <vector>
#include <string>
#include "ColorPixelBuffer.h"
#include <mutex>
#include <unordered_map>

class TextureManager
{
public:
	TextureManager();
	int addTextureBySurface(SDL_Surface* s);
	int addTextureByPath(std::string path);
	const ColorPixelBuffer& getTextureByHandle(int i) const;
	bool handleIsValid(int h) const;
	Vec4_f32x16 gatherLinearIntensitiesFromMultipleTextures(int32x16 textureIndices, float32x16 u, float32x16 v, Mask16 mask) const;
	void clear();
private:
	std::vector<ColorPixelBuffer> textures;
	std::vector<uint32_t*> bufferForTexture;
	std::vector<ColorPixelBuffer::Sizes> sizesForTexture;
	std::unordered_map<std::string, int> pathToIndexMap;
	std::mutex mtx;
};