#pragma once
#include <string>
#include <d3d11.h>
#undef min
#undef max
struct GameSettings;
class RendererBase
{
public:
	RendererBase() = default;
	virtual void loadScene(std::string path, std::string mode) = 0;
	virtual void renderFrame(const GameSettings& settings) = 0;
private:
};