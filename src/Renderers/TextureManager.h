#pragma once
#include <vector>
#include <string>
#include "../Texture.h"
#include <mutex>
#include <unordered_map>
#include "../Threadpool.h"
class TextureManager
{
public:
	TextureManager(const TextureManager&) = delete;
	TextureManager(TextureManager&&) = delete;
	TextureManager& operator=(const TextureManager&) = delete;
	TextureManager& operator=(TextureManager&&) = delete;

	static TextureManager& getInstance();
	int addTextureBySurface(SDL_Surface* s);

	//Loads texture by path specified and returns it's handle.
	int addTextureByPath(std::string path);

	//TODO: just use std::async/future/etc
	//Asynchronously loads a texture at specified path, returning the handle to import task.
	//When the task completes, stores the handle to the imported texture to retHandle
	//The caller must ensure that retHandle reference is valid until the load task completes
	Threadpool::TaskHandle addTextureByPathAsync(const std::string& path, int* retHandle);

	const Texture& getTextureByHandle(int i) const;
	bool handleIsValid(int h) const;
	void clear();

	static inline constexpr int INVALID_HANDLE = -1;
	static inline constexpr int FALLBACK_HANDLE = 0;

	Vec4_f32x16 gatherLinearIntesitiesFromMultipleTextures(const int32x16& textureInd, const float32x16& u, const float32x16& v, const Mask16& mask) const;
	float32x16 gatherAlphaFromMultipleTextures(const int32x16& textureInd, const float32x16& u, const float32x16& v, const Mask16& mask) const;
private:
	TextureManager();
	std::vector<Texture> textures;
	//std::vector<uint32_t*> bufferForTexture;
	//std::vector<ColorPixelBuffer::Sizes> sizesForTexture;
	std::unordered_map<std::string, int> pathToIndexMap;
	std::recursive_mutex mtx;
};