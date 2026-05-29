#include "HardwareRasterizingRenderer.h"
#include "../AssetLoader.h"
#include "TextureManager.h"

using namespace HardwareRasterizing;
HardwareRasterizingRenderer::HardwareRasterizingRenderer()
{
	//Create main vertex shader
	ShaderCreationDesc vsDesc;
	vsDesc.path = "BasicVS.cso";
	vsDesc.inputLayout = { {"Pos", 0, DXGI_FORMAT_R32G32_FLOAT, 0,0,D3D11_INPUT_PER_VERTEX_DATA, 0} };
	this->mainVS = Shader<ID3D11VertexShader>(this->gfx, vsDesc);

	ShaderCreationDesc skyboxVsDesc;
	skyboxVsDesc.path = "SkyboxVS.cso";
	skyboxVsDesc.inputLayout = {
		{"Pos", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0,0,D3D11_INPUT_PER_VERTEX_DATA, 0},
		{"UV", 0, DXGI_FORMAT_R32G32_FLOAT, 0,D3D11_APPEND_ALIGNED_ELEMENT,D3D11_INPUT_PER_VERTEX_DATA, 0},
	};
	this->skyboxVS = Shader<ID3D11VertexShader>(this->gfx, skyboxVsDesc);

	ShaderCreationDesc skyboxPsDesc;
	skyboxPsDesc.path = "SkyboxPS.cso";
	this->skyboxPS = Shader<ID3D11PixelShader>(this->gfx, skyboxPsDesc);


	ShaderCreationDesc psDesc;
	psDesc.path = "BasicPS.cso";
	this->mainPS = Shader<ID3D11PixelShader>(this->gfx, psDesc);
	this->gfx.deviceContext->PSSetShader(this->mainPS.shader.Get(), nullptr, 0);

	//Disable backface culling
	Microsoft::WRL::ComPtr<ID3D11RasterizerState> rasterizerState;
	D3D11_RASTERIZER_DESC rdsc = {};
	rdsc.FillMode = D3D11_FILL_SOLID;
	rdsc.CullMode = D3D11_CULL_NONE;
	DX_THROW_ON_FAIL(this->gfx.device->CreateRasterizerState(&rdsc, &rasterizerState), "Create rasterizer state");
	this->gfx.deviceContext->RSSetState(rasterizerState.Get());

	//Create and set depth buffer
	D3D11_TEXTURE2D_DESC descDepth = {};
	descDepth.Width = this->gfx.w;
	descDepth.Height = this->gfx.h;
	descDepth.MipLevels = 1;
	descDepth.ArraySize = 1;
	descDepth.Format = DXGI_FORMAT_D32_FLOAT;
	descDepth.SampleDesc.Count = 1;
	descDepth.SampleDesc.Quality = 0;
	descDepth.Usage = D3D11_USAGE_DEFAULT;
	descDepth.BindFlags = D3D11_BIND_DEPTH_STENCIL;
	DX_THROW_ON_FAIL(this->gfx.device->CreateTexture2D(&descDepth, nullptr, &this->depthStencil), "Create depth stencil texture");

	D3D11_DEPTH_STENCIL_VIEW_DESC descDSV = {};
	descDSV.Format = DXGI_FORMAT_D32_FLOAT;
	descDSV.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
	descDSV.Texture2D.MipSlice = 0;
	DX_THROW_ON_FAIL(this->gfx.device->CreateDepthStencilView(this->depthStencil.Get(), &descDSV, &this->depthStencilView), "Create depth stencil view");

	D3D11_DEPTH_STENCIL_DESC dsDesc = {};
	dsDesc.DepthEnable = true;
	dsDesc.StencilEnable = false;
	dsDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
	dsDesc.DepthFunc = D3D11_COMPARISON_GREATER; //using reverse depth, greater comparison is needed
	DX_THROW_ON_FAIL(this->gfx.device->CreateDepthStencilState(&dsDesc, &this->mainDepthStencilState), "Create depth stencil state");

	//to avoid headaches with the sky, just render it first without depth tests and writes
	dsDesc.DepthEnable = false;
	dsDesc.StencilEnable = false;
	dsDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
	this->gfx.device->CreateDepthStencilState(&dsDesc, &this->skyboxDepthStencilState);

	/*
	std::vector<Vertex3D> skyCubeVerts = this->gfx.generateRectangularCuboidNoDedup();
	this->skyCubeVertexCount = skyCubeVerts.size();

	D3D11_BUFFER_DESC vertexBufferDesc;
	vertexBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	vertexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
	vertexBufferDesc.CPUAccessFlags = 0;
	vertexBufferDesc.MiscFlags = 0;
	vertexBufferDesc.ByteWidth = verts.size() * sizeof(Vertex2D);
	vertexBufferDesc.StructureByteStride = sizeof(Vertex2D);

	D3D11_SUBRESOURCE_DATA vertexBufferSubresourceData;
	vertexBufferSubresourceData.pSysMem = verts.data();
	DX_THROW_ON_FAIL(this->gfx.device->CreateBuffer(&vertexBufferDesc, &vertexBufferSubresourceData, &this->heightmapVB), "Create Main VB", this->gfx.device.Get());

	vertexBufferDesc.ByteWidth = skyCubeVerts.size() * sizeof(Vertex3D);
	vertexBufferDesc.StructureByteStride = sizeof(Vertex3D);
	vertexBufferSubresourceData.pSysMem = skyCubeVerts.data();
	DX_THROW_ON_FAIL(this->gfx.device->CreateBuffer(&vertexBufferDesc, &vertexBufferSubresourceData, &this->skyboxVB), "Create skybox VB", this->gfx.device.Get());
	*/

	//Create constant buffer
	D3D11_BUFFER_DESC cbd;
	cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	cbd.Usage = D3D11_USAGE_DYNAMIC;
	cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	cbd.MiscFlags = 0;
	cbd.ByteWidth = sizeof(ConstantBuffer);
	cbd.StructureByteStride = 0;
	D3D11_SUBRESOURCE_DATA csd;
	csd.pSysMem = &this->mainCB_CPU;
	DX_THROW_ON_FAIL(this->gfx.device->CreateBuffer(&cbd, &csd, &this->mainConstantBuffer), "Create constant buffer");

	//Create skybox cubemap and sampler
	/*
	std::array<std::string, 6> skyboxCubemapPaths = {
		//https://svs.gsfc.nasa.gov/4851
		"images/sky/v2_px.png",
		"images/sky/v2_nx.png",
		"images/sky/v2_py.png",
		"images/sky/v2_ny.png",
		"images/sky/v2_pz.png",
		"images/sky/v2_nz.png",
	};
	for (int i = 0; i < 6; ++i)
	{
		//skyboxCubemapPaths[i] = "images/sky/" + std::to_string(i) + ".png";
	}
	this->skyboxCubemap = CubemapTexture(skyboxCubemapPaths, this->gfx);

	D3D11_SAMPLER_DESC samplerDesc = {};
	samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR; //D3D11_FILTER_ANISOTROPIC;
	samplerDesc.AddressU = samplerDesc.AddressV = samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_BORDER;
	float skyboxBorderColor[4] = { 1.f,0.f,1.f,1.f }; //magenta
	memcpy(samplerDesc.BorderColor, skyboxBorderColor, sizeof(skyboxBorderColor));
	DX_THROW_ON_FAIL(this->gfx.device->CreateSamplerState(&samplerDesc, &this->skyboxSamplerState));*/
}

void HardwareRasterizingRenderer::loadScene(RendererLoadSceneData scd)
{
	this->textureManager.clear();
	this->textures.clear();
	this->sceneModels.clear();

	AssetLoader ldr;
	std::vector<AssetLoader::ImportedModel> loadedModels;
	std::vector<Vertex> verts;
	for (auto [path, mode] : scd.files)
	{
		if (mode == "obj") { loadedModels = ldr.loadObj(path, ""); }
		else if (mode == "bmdl") { loadedModels = ldr.loadBmdl(path); }
		else throw std::runtime_error("Unsupported mode for RayCastingRenderer::loadScene: " + mode);

		size_t importModelCount = loadedModels.size();
		std::vector<int> diffuseMapIndices(importModelCount, -1);
		std::vector<Threadpool::TaskHandle> textureLoadingTasks(importModelCount);

		Threadpool::Task tsk;
		for (int i = 0; i < importModelCount; ++i)
		{
			tsk.func = [&, this, i]() {
				if (loadedModels[i].diffuseMapPath) diffuseMapIndices[i] = this->textureManager.addTextureByPath(*loadedModels[i].diffuseMapPath);
				else diffuseMapIndices[i] = 0;
				};
			textureLoadingTasks[i] = Threadpool::instance->addTask(tsk);
		}

		for (int i = 0; i < loadedModels.size(); ++i)
		{
			Model& currHwModel = this->sceneModels.emplace_back();
			currHwModel.startVertex = verts.size();
			for (auto& it : loadedModels[i].triangles)
			{
				for (int k = 0; k < 3; ++k)
				{
					auto& v = verts.emplace_back();
					v.x = it.v[k].space.x;
					v.y = it.v[k].space.y;
					v.z = it.v[k].space.z;
					v.u = it.v[k].diffuseMapCoords.x;
					v.v = it.v[k].diffuseMapCoords.y;
					v.nx = it.v[k].normal.x;
					v.ny = it.v[k].normal.y;
					v.nz = it.v[k].normal.z;
				}
			}
			currHwModel.vertexCount = verts.size() - currHwModel.startVertex;
		}

		Threadpool::instance->blockUntilComplete(textureLoadingTasks);
		for (int i = 0; i < loadedModels.size(); ++i)
			this->sceneModels[i].diffuseMapIndex = diffuseMapIndices[i];

	}

	//Now create one giant vertex buffer set it to all models. Indices are already there
	//Also, create DX textures and assign them
	D3D11_BUFFER_DESC vertexBufferDesc;
	vertexBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	vertexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
	vertexBufferDesc.CPUAccessFlags = 0;
	vertexBufferDesc.MiscFlags = 0;
	vertexBufferDesc.ByteWidth = verts.size() * sizeof(Vertex);
	vertexBufferDesc.StructureByteStride = sizeof(Vertex);

	D3D11_SUBRESOURCE_DATA vertexBufferSubresourceData;
	vertexBufferSubresourceData.pSysMem = verts.data();
	DX_THROW_ON_FAIL(this->gfx.device->CreateBuffer(&vertexBufferDesc, &vertexBufferSubresourceData, &this->mainVertexBuffer), "Create Main VB", this->gfx.device.Get());

	std::unordered_set<int> alreadyLoadedTextureIndices = { -1 };
	for (int i = 0; i < this->sceneModels.size(); ++i)
	{
		auto& currModel = this->sceneModels[i];
		currModel.vertexBuffer = this->mainVertexBuffer;
		if (alreadyLoadedTextureIndices.find(currModel.diffuseMapIndex) == alreadyLoadedTextureIndices.end()) continue;

		HW_Texture2D createdTexture;
		//createdTexture.texture
		D3D11_TEXTURE2D_DESC texDesc = {};
		texDesc.ArraySize = 1;
		texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
		texDesc.CPUAccessFlags = 0;
		texDesc.MipLevels = 1;
		texDesc.Usage = D3D11_USAGE_DEFAULT;
		texDesc.MiscFlags = 0; //TODO: verify this
		texDesc.SampleDesc.Count = 1;
		texDesc.SampleDesc.Quality = 0;
		texDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

		std::unique_ptr<uint32_t[]> mem;
		this->textureManager.getTextureByHandle(currModel.diffuseMapIndex).QueryTexture(&texDesc.Width, &texDesc.Height, &mem);

		D3D11_SUBRESOURCE_DATA data;
		data.pSysMem = mem.get();
		data.SysMemPitch = texDesc.Width * 4;
		data.SysMemSlicePitch = 0;
		DX_THROW_ON_FAIL(gfx.device->CreateTexture2D(&texDesc, &data, &createdTexture.texture));

		D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
		srvDesc.Format = texDesc.Format;
		srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURECUBE;
		srvDesc.Texture2D.MipLevels = texDesc.MipLevels;
		srvDesc.Texture2D.MostDetailedMip = 0;
		DX_THROW_ON_FAIL(gfx.device->CreateShaderResourceView(createdTexture.texture.Get(), &srvDesc, &createdTexture.srv));

		D3D11_SAMPLER_DESC samplerDesc = {};
		samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR; //D3D11_FILTER_ANISOTROPIC;
		samplerDesc.AddressU = samplerDesc.AddressV = samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
		//float skyboxBorderColor[4] = { 1.f,0.f,1.f,1.f }; //magenta
		//memcpy(samplerDesc.BorderColor, skyboxBorderColor, sizeof(skyboxBorderColor));
		DX_THROW_ON_FAIL(this->gfx.device->CreateSamplerState(&samplerDesc, &createdTexture.samplerState));

		if (this->textures.size() <= currModel.diffuseMapIndex) this->textures.resize(currModel.diffuseMapIndex + 1);
		this->textures[currModel.diffuseMapIndex] = createdTexture;
		alreadyLoadedTextureIndices.insert(currModel.diffuseMapIndex); //avoid creating same texture twice
	}

	//Create constant buffer
	D3D11_BUFFER_DESC cbd;
	cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	cbd.Usage = D3D11_USAGE_DYNAMIC;
	cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	cbd.MiscFlags = 0;
	cbd.ByteWidth = sizeof(ConstantBuffer);
	cbd.StructureByteStride = 0;
	D3D11_SUBRESOURCE_DATA csd;
	csd.pSysMem = &this->mainCB_CPU;
	DX_THROW_ON_FAIL(this->gfx.device->CreateBuffer(&cbd, &csd, &this->mainConstantBuffer), "Create constant buffer");
	/*
		if (this->discardUntexturedTriangles)
		{
			for (int i = 0; i < sceneModels.size(); ++i)
			{
				if (this->sceneModels[i].textureIndex == 0) {
					this->sceneModels[i--] = std::move(this->sceneModels.back());
					this->sceneModels.pop_back();
				}
			}
		}*/
}

void HardwareRasterizingRenderer::renderFrame(const GameSettings& settings)
{
}
