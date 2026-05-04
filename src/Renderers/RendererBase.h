#pragma once
#include <string>
#include <d3d11.h>
#include <vector>
#include "../C_Input.h"
#include "../Vec.h"
#include "../helpers.h"
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

	static void calculateBarycentricCoordinates2D(const Vec4_f32x16& r, const Vec4_f32x16& r1, const Vec4_f32x16& r2, const Vec4_f32x16& r3, const float32x16& rcpSignedArea, float32x16& alpha, float32x16& beta, float32x16& gamma);
	static void calculateBarycentricCoordinates3D(const Vec4_f32x16& P, const Vec4_f32x16& A, const Vec4_f32x16& B, const Vec4_f32x16& C, float32x16& alpha, float32x16& beta, float32x16& gamma);
	static void mask_store_vec4_f32x16_to_framebuffer(const Vec4_f32x16& pack, void* frameBuffer, int x, int y, int w, Mask16 mask);
	static Vec4_f32x16 mask_load_vec4_f32x16_from_framebuffer(const void* frameBuffer, int x, int y, int w, Mask16 mask);
private:
};