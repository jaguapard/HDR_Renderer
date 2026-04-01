#include <vector>
#include <string>
#include "ColorPixelBuffer.h"
#include <mutex>
namespace Rasterizing
{
	class TextureManager
	{
	public:
		TextureManager();
		int addTextureByPath(std::string path);
		const Rasterizing::ColorPixelBuffer& getTextureByHandle(int i) const;
		bool handleIsValid(int h) const;
	private:
		std::vector<Rasterizing::ColorPixelBuffer> textures;
		std::mutex mtx;
	};
}