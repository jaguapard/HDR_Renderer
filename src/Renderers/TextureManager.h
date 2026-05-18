#pragma once
#include <vector>
#include <string>
#include "../Texture.h"
#include <mutex>
#include <unordered_map>

class TextureManager
{
public:
	TextureManager();
	int addTextureBySurface(SDL_Surface* s);
	int addTextureByPath(std::string path);
	const Texture& getTextureByHandle(int i) const;
	bool handleIsValid(int h) const;
	void clear();

	static inline constexpr int INVALID_HANDLE = -1;
	static inline constexpr int FALLBACK_HANDLE = 0;

	Vec4_f32x16 gatherLinearIntesitiesFromMultipleTextures(const int32x16& textureInd, const float32x16& u, const float32x16& v, const Mask16& mask) const;
	float32x16 gatherAlphaFromMultipleTextures(const int32x16& textureInd, const float32x16& u, const float32x16& v, const Mask16& mask) const;
private:
	std::vector<Texture> textures;
	//std::vector<uint32_t*> bufferForTexture;
	//std::vector<ColorPixelBuffer::Sizes> sizesForTexture;
	std::unordered_map<std::string, int> pathToIndexMap;
	std::recursive_mutex mtx;
};