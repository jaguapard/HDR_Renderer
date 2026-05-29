#pragma once
#include "libs.h"
class Graphics
{
public:
	Graphics(uint32_t w, uint32_t h);
	SDL_Window* window;
	uint32_t w, h;
	//static const std::strong_ordering SHADERS_FOLDER;

	Microsoft::WRL::ComPtr<IDXGISwapChain> swapChain;
	Microsoft::WRL::ComPtr<ID3D11Device> device;
	Microsoft::WRL::ComPtr<ID3D11DeviceContext> deviceContext;
	Microsoft::WRL::ComPtr<ID3D11RenderTargetView> mainRenderTargetView;
private:
};