#include "Graphics.h"
#include "errors.h"
#include <d3dcompiler.h>

using namespace DirectX;
Microsoft::WRL::ComPtr<ID3DBlob> Graphics::CompileShader(const char* sourceCode, const char* entryPointName, const char* targetLevel)
{
    Microsoft::WRL::ComPtr<ID3DBlob> blob, errors;
    HRESULT hr = D3DCompile(sourceCode, strlen(sourceCode), nullptr, nullptr, nullptr,
        entryPointName, targetLevel, 0, 0, &blob, &errors);
    if (FAILED(hr)) {
        std::string errMsg;
        if (errors) {
            errMsg = static_cast<const char*>(errors->GetBufferPointer());
        }
        else {
            errMsg = "Unknown compilation error";
        }
        RAISE_ERROR(errMsg);
    }
    return blob;
}

Graphics::Graphics(uint32_t w, uint32_t h)
{
    if (Graphics::instance) throw std::runtime_error("Graphics class instance already exists. Graphics class is a singleton and cannot have more than one instance.");
    Graphics::instance = this;
    this->w = w;
    this->h = h;
    this->window = SDL_CreateWindow("Heightmap renderer", w, h, 0);
    if (!this->window) RAISE_ERROR("SDL_CreateWindow failed");

    SDL_PropertiesID props = SDL_GetWindowProperties(this->window);
    if (!props) RAISE_ERROR("SDL_GetWindowProperties returned NULL");

    void* rawHwnd = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr);
    HWND hwnd = reinterpret_cast<HWND>(rawHwnd);
    if (!hwnd) RAISE_ERROR("Failed to obtain HWND from SDL3 window properties");

    TTF_Init();

    DXGI_SWAP_CHAIN_DESC scd = {};
    scd.BufferCount = 2;
    scd.BufferDesc.Width = w;
    scd.BufferDesc.Height = h;
    scd.BufferDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.OutputWindow = hwnd;
    scd.SampleDesc.Count = 1;
    scd.Windowed = TRUE;
    scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;

    UINT createDeviceFlags = 0;
#ifndef NDEBUG
    createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    DX_THROW_ON_FAIL((D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE,
        nullptr, createDeviceFlags, nullptr, 0, D3D11_SDK_VERSION,
        &scd, &this->swapChain, &this->device, nullptr, &this->deviceContext)), "D3D11CreateDeviceAndSwapChain");

    Microsoft::WRL::ComPtr<ID3D11Texture2D> backBuffer;
    DX_THROW_ON_FAIL(this->swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), &backBuffer), "Get back buffer");
    DX_THROW_ON_FAIL(device->CreateRenderTargetView(backBuffer.Get(), nullptr, &this->mainRenderTargetView), "Create render target view on backbuffer");

    D3D11_VIEWPORT vp;
    vp.TopLeftX = 0;
    vp.TopLeftY = 0;
    vp.Width = w;
    vp.Height = h;
    vp.MinDepth = 0;
    vp.MaxDepth = 1;
    this->deviceContext->RSSetViewports(1, &vp);
}

// Simple shaders
static const char* src_cpuRendererVS = R"(
struct VS_OUTPUT {
    float4 Pos : SV_POSITION;
    float2 Tex : TEXCOORD0;
};

VS_OUTPUT main(uint id : SV_VertexID) {
    VS_OUTPUT output;
    float2 verts[3] = {
        float2(-1, -1),
        float2(-1, 3),
        float2(3, -1)
    };
    output.Pos = float4(verts[id], 0, 1);
    output.Tex = (verts[id] + 1) * 0.5;
    return output;
}
)";

const char* src_cpuRendererPS = R"(
Texture2D tex : register(t0);
SamplerState samp : register(s0);
float4 main(float4 pos : SV_POSITION, float2 uv : TEXCOORD0) : SV_Target {
    float2 flippedUV = float2(uv.x, uv.y);
    return tex.Sample(samp, flippedUV);
}
)";

CPU_Renderer_Context Graphics::makeCPURendererContext()
{
    CPU_Renderer_Context ret;
    D3D11_TEXTURE2D_DESC texDesc = {};

    constexpr int DEBUG_MODE_DOWNSCALE_MULT = 10;
#ifdef NDEBUG
    texDesc.Width = w;
    texDesc.Height = h;
#else
    //to have a slightest hope of rendering a frame during your lifetime, resolution has to be havily cut down in Debug mode
    texDesc.Width = w / DEBUG_MODE_DOWNSCALE_MULT;
    texDesc.Height = h / DEBUG_MODE_DOWNSCALE_MULT;
#endif
    texDesc.MipLevels = 1;
    texDesc.ArraySize = 1;
    texDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    texDesc.SampleDesc.Count = 1;
    texDesc.Usage = D3D11_USAGE_DYNAMIC;
    texDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    texDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

    if (texDesc.Width % 4 != 0 || texDesc.Height % 4 != 0) throw std::runtime_error("Unsupported output texture size! Each side length must be multiple of 4."); //I think this is correct, because it goes haywire if sizes are weird even if renderers are doing everything properly.

    DX_THROW_ON_FAIL(this->device->CreateTexture2D(&texDesc, nullptr, &ret.cpuTexture));
    DX_THROW_ON_FAIL(this->device->CreateShaderResourceView(ret.cpuTexture.Get(), nullptr, &ret.cpuTextureSRV));

    D3D11_SAMPLER_DESC sampDesc = {};
    sampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sampDesc.AddressU = sampDesc.AddressV = sampDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    DX_THROW_ON_FAIL(this->device->CreateSamplerState(&sampDesc, &ret.samplerState));

    auto vsBlob = Graphics::CompileShader(src_cpuRendererVS, "main", "vs_5_0");
    auto psBlob = Graphics::CompileShader(src_cpuRendererPS, "main", "ps_5_0");
    DX_THROW_ON_FAIL(this->device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &ret.vs));
    DX_THROW_ON_FAIL(this->device->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &ret.ps));

    ret.screenW = w;
    ret.screenH = h;
    ret.outputTextureW = texDesc.Width;
    ret.outputTextureH = texDesc.Height;

    return ret;
}

D3D11_MAPPED_SUBRESOURCE Graphics::CPURendering_OnFrameStart(CPU_Renderer_Context& ctx)
{
    D3D11_MAPPED_SUBRESOURCE mapped;
    DX_THROW_ON_FAIL(this->deviceContext->Map(ctx.cpuTexture.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped));
    return mapped;
}

void Graphics::CPURendering_Present(CPU_Renderer_Context& ctx)
{
    this->deviceContext->Unmap(ctx.cpuTexture.Get(), 0);
    this->deviceContext->OMSetRenderTargets(1, this->mainRenderTargetView.GetAddressOf(), nullptr);
    float clear[4] = { 0,0,0,1 };
    this->deviceContext->ClearRenderTargetView(this->mainRenderTargetView.Get(), clear);

    this->deviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    this->deviceContext->VSSetShader(ctx.vs.Get(), nullptr, 0);
    this->deviceContext->PSSetShader(ctx.ps.Get(), nullptr, 0);
    this->deviceContext->PSSetShaderResources(0, 1, ctx.cpuTextureSRV.GetAddressOf());
    this->deviceContext->PSSetSamplers(0, 1, ctx.samplerState.GetAddressOf());

    this->deviceContext->Draw(3, 0);
    DX_THROW_ON_FAIL(this->swapChain->Present(0, 0));
}

