#include "Graphics.h"
#include "errors.h"
using namespace DirectX;
Graphics::Graphics(uint32_t w, uint32_t h)
{
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
