#pragma once
#include <string>
#include <d3d11.h>
#include <vector>

#undef min
#undef max
struct GameSettings;

struct RendererLoadSceneData
{
	std::vector<std::pair<std::string, std::string>> files;
};
class RendererBase
{
public:
	RendererBase() = default;
	virtual void loadScene(RendererLoadSceneData scd) = 0;
	virtual void renderFrame(const GameSettings& settings) = 0;
private:
};