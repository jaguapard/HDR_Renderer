#pragma once
#include <string>
#include <d3d11.h>
#include <vector>
#include "../C_Input.h"
#include "../Vec.h"
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
	//virtual void handleInputEvent(const SDL_Event& ev, C_Input& input) = 0;
	virtual void loadScene(RendererLoadSceneData scd) = 0;
	virtual void renderFrame(const GameSettings& settings) = 0;
	static void mask_store_vec4_f32x16_to_framebuffer(const Vec4_f32x16& pack, void* frameBuffer, int x, int y, int w, Mask16 mask);
private:
};