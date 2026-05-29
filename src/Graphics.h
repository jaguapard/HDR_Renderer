#pragma once
#include "libs.h"

struct CPU_Renderer_Context
{
	Microsoft::WRL::ComPtr<ID3D11VertexShader> vs;
	Microsoft::WRL::ComPtr<ID3D11PixelShader> ps;
	Microsoft::WRL::ComPtr<ID3D11Texture2D> cpuTexture;
	Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> cpuTextureSRV;
	Microsoft::WRL::ComPtr<ID3D11SamplerState> samplerState;
	uint32_t outputTextureW, outputTextureH, screenW, screenH;
};
class Graphics
{
public:
	Graphics(uint32_t w, uint32_t h);
	Graphics(const Graphics&) = delete;
	Graphics(Graphics&&) = delete;
	Graphics& operator=(Graphics&&) = delete;
	Graphics& operator=(const Graphics&) = delete;
	SDL_Window* window;
	uint32_t w, h;
	//static const std::strong_ordering SHADERS_FOLDER;

	//Prepares Graphics pipeline for simple CPU Renderer mode. The returned struct must be preserved by the called and provided to Graphics instance when rendering
	CPU_Renderer_Context makeCPURendererContext();

	//Binds all shaders and resources, and returns the mapped CPU texture that can be written into by the output data to be shown on screen. Must be called before CPURendering_Present
	D3D11_MAPPED_SUBRESOURCE CPURendering_OnFrameStart(CPU_Renderer_Context& ctx);
	//Unmaps CPU texture and outputs the frame to the screen
	void CPURendering_Present(CPU_Renderer_Context& ctx);

	static inline Graphics* instance = nullptr;


	Microsoft::WRL::ComPtr<IDXGISwapChain> swapChain;
	Microsoft::WRL::ComPtr<ID3D11Device> device;
	Microsoft::WRL::ComPtr<ID3D11DeviceContext> deviceContext;
	Microsoft::WRL::ComPtr<ID3D11RenderTargetView> mainRenderTargetView;
	

	static Microsoft::WRL::ComPtr<ID3DBlob> CompileShader(const char* sourceCode, const char* entryPointName, const char* targetLevel);
private:
};