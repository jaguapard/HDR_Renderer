#pragma once
#include "RendererBase.h"
#include "../Graphics.h"
#include <memory>
#include <filesystem>
#include "../libs.h"
#include <d3dcompiler.h>
#include <DirectXMath.h>

namespace HardwareRasterizing
{
	struct ShaderCreationDesc
	{
		std::string path;
		//Input layout elements. If empty, this argument will be ignored, and no layout will be created.
		std::vector<D3D11_INPUT_ELEMENT_DESC> inputLayout = {};
		//If set to false, then path provided will be appended to default shader path. If not, will pass the path directly to Direct3D shader creation
		bool pathIsAbsolute = false;
		ID3D11ClassLinkage* classLinkange = nullptr; //passed directly to D3D11
	};

	static const std::string SHADERS_FOLDER = []() {
		//std::wstring modulePath = utils::getCurrModuleFullPath();

		wchar_t path[8192] = { 0 };
		HRESULT hr = GetModuleFileNameW(nullptr, path, sizeof(path) / sizeof(path[0]));
		if (FAILED(hr)) throw std::runtime_error("GetModuleFileNameW failed with HRESULT " + std::to_string(hr));
		std::wstring modulePath = std::wstring(path);

		auto p = std::filesystem::path(modulePath);
		auto r = p.remove_filename();
		r.append(L"Shaders\\");
		return r.string();
	}();
	template<typename T>
	struct Shader
	{
		Microsoft::WRL::ComPtr<T> shader;
		Microsoft::WRL::ComPtr<ID3D11InputLayout> inputLayout;
		Microsoft::WRL::ComPtr<ID3DBlob> bytecode;

		Shader() = default;
		Shader(Graphics& gfx, const ShaderCreationDesc& desc);
	};

	static std::wstring ascii_string_to_wstring(const std::string s)
	{
		std::wstring ws;
		for (auto& c : s)
			if (c >= 0 && c <= 127)
				ws.push_back(c);
			else RAISE_ERROR("Naive extension of non-ASCII string to wstring requested");
		return ws;
	}

	template<typename T>
	inline Shader<T>::Shader(Graphics& gfx, const ShaderCreationDesc& desc)
	{
		if (desc.path.empty()) RAISE_ERROR("Attempted to create shader with empty path.");
		std::string asciiPath;
		if (desc.pathIsAbsolute) asciiPath = desc.path;
		else asciiPath = SHADERS_FOLDER + desc.path;
		std::string baseMessage = "While creating shader from file " + asciiPath + ": ";

		std::wstring fullPath = ascii_string_to_wstring(asciiPath);
		DX_THROW_ON_FAIL(D3DReadFileToBlob(fullPath.c_str(), &this->bytecode), baseMessage + "D3DReadFileToBlob");
		if (!desc.inputLayout.empty())
		{
			DX_THROW_ON_FAIL(gfx.device->CreateInputLayout(desc.inputLayout.data(), desc.inputLayout.size(), this->bytecode->GetBufferPointer(), this->bytecode->GetBufferSize(), &this->inputLayout), baseMessage + "CreateInputLayout");
		}

		if constexpr (std::is_same_v<T, ID3D11VertexShader>)
		{
			DX_THROW_ON_FAIL(gfx.device->CreateVertexShader(this->bytecode->GetBufferPointer(), this->bytecode->GetBufferSize(), desc.classLinkange, &this->shader), baseMessage + "CreateVertexShader");
		}
		else if constexpr (std::is_same_v<T, ID3D11PixelShader>)
		{
			DX_THROW_ON_FAIL(gfx.device->CreatePixelShader(this->bytecode->GetBufferPointer(), this->bytecode->GetBufferSize(), desc.classLinkange, &this->shader), baseMessage + "CreatePixelShader");
		}
		else static_assert(false, "Unsupported type for Shader");
	}

	struct Model
	{
		Microsoft::WRL::ComPtr<ID3D11Buffer> vertexBuffer;
		int diffuseMapIndex;
		UINT startVertex, vertexCount;
	};

	struct alignas(16) ConstantBuffer
	{
		DirectX::XMMATRIX view, projection, viewProjection;
		DirectX::XMVECTOR time;
		DirectX::XMVECTOR camPos, lightDir;
	};

	struct Vertex
	{
		float x, y, z, u, v, nx, ny, nz;
	};

	struct HW_Texture2D
	{
		Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
		Microsoft::WRL::ComPtr<ID3D11SamplerState> samplerState;
		Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> srv;
	};
}
class HardwareRasterizingRenderer : public RendererBase
{
public:
	HardwareRasterizingRenderer();
	virtual void loadScene(RendererLoadSceneData scd);
	virtual void renderFrame(const GameSettings& settings);

private:
	Graphics& gfx = *Graphics::instance;
	HardwareRasterizing::ConstantBuffer mainCB_CPU;
	HardwareRasterizing::Shader<ID3D11VertexShader> mainVS, skyboxVS;
	HardwareRasterizing::Shader<ID3D11PixelShader> mainPS, skyboxPS;
	Microsoft::WRL::ComPtr<ID3D11Buffer> mainConstantBuffer, mainVertexBuffer;
	Microsoft::WRL::ComPtr<ID3D11Texture2D> depthStencil;
	Microsoft::WRL::ComPtr<ID3D11DepthStencilState> mainDepthStencilState, skyboxDepthStencilState;
	Microsoft::WRL::ComPtr<ID3D11DepthStencilView> depthStencilView;
	Microsoft::WRL::ComPtr<ID3D11SamplerState> skyboxSamplerState;

	std::vector<HardwareRasterizing::Model> sceneModels;
	std::vector<HardwareRasterizing::HW_Texture2D> textures;
	Vec4f lightDir = Vec4f(0.4, 0.2, 0.6, 0);
};