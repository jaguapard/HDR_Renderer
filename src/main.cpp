#include <SDL3/SDL.h>
#include <SDL3/SDL_system.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <iostream>
#include <sstream>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "SDL3.lib")

// Simple shaders
const char* g_VS = R"(
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

const char* g_PS = R"(
Texture2D tex : register(t0);
SamplerState samp : register(s0);
float4 main(float4 pos : SV_POSITION, float2 uv : TEXCOORD0) : SV_Target {
    float2 flippedUV = float2(uv.x, 1.0 - uv.y);
    return tex.Sample(samp, flippedUV);
}
)";

static void __raise_error_internal(const char* filePath, int line, std::string errorMsg)
{
    std::stringstream ss;
    ss << "Error in file: " << filePath << "\n" << "Line " << line << "\n" << errorMsg;
    throw std::runtime_error(ss.str());
}
#define RAISE_ERROR(msg) (__raise_error_internal(__FILE__, __LINE__, std::string("Error: ")+msg))

ID3DBlob* CompileShader(const char* source, const char* entry, const char* target) {
    ID3DBlob* blob = nullptr;
    ID3DBlob* errors = nullptr;
    HRESULT hr = D3DCompile(source, strlen(source), nullptr, nullptr, nullptr,
        entry, target, 0, 0, &blob, &errors);
    if (FAILED(hr)) {
        std::string errMsg;
        if (errors) {
            errMsg = static_cast<const char*>(errors->GetBufferPointer());
            errors->Release();
        }
        else {
            errMsg = "Unknown compilation error";
        }
        RAISE_ERROR(errMsg);
    }
    if (errors) errors->Release();
    return blob;
}

int main(int argc, char* argv[]) 
{
    try
    {
        if (!SDL_Init(SDL_INIT_VIDEO)) RAISE_ERROR("SDL_Init failed");

        int w = 2560;
        int h = 1440;
        SDL_Window* window = SDL_CreateWindow("SDL3 + D3D11 Pixel Display", w, h, 0);
        if (!window) RAISE_ERROR("SDL_CreateWindow failed");

        SDL_PropertiesID props = SDL_GetWindowProperties(window);
        if (!props) RAISE_ERROR("SDL_GetWindowProperties returned NULL");

        void* rawHwnd = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr);
        HWND hwnd = reinterpret_cast<HWND>(rawHwnd);
        if (!hwnd) RAISE_ERROR("Failed to obtain HWND from SDL3 window properties");

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

        IDXGISwapChain* swapChain = nullptr;
        ID3D11Device* device = nullptr;
        ID3D11DeviceContext* context = nullptr;
        if (FAILED(D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE,
            nullptr, 0, nullptr, 0, D3D11_SDK_VERSION,
            &scd, &swapChain, &device, nullptr, &context)))
            RAISE_ERROR("D3D11CreateDeviceAndSwapChain failed");

        ID3D11Texture2D* backBuffer = nullptr;
        if (FAILED(swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&backBuffer)))
            RAISE_ERROR("GetBuffer failed");

        ID3D11RenderTargetView* rtv = nullptr;
        if (FAILED(device->CreateRenderTargetView(backBuffer, nullptr, &rtv))) {
            backBuffer->Release();
            RAISE_ERROR("CreateRenderTargetView failed");
        }
        backBuffer->Release();

        D3D11_VIEWPORT vp;
        vp.TopLeftX = 0;
        vp.TopLeftY = 0;
        vp.Width = w;
        vp.Height = h;
        vp.MinDepth = 0;
        vp.MaxDepth = 1;
        context->RSSetViewports(1, &vp);

        // Dynamic texture (CPU-writable + SRV)
        D3D11_TEXTURE2D_DESC texDesc = {};
        texDesc.Width = w;
        texDesc.Height = h;
        texDesc.MipLevels = 1;
        texDesc.ArraySize = 1;
        texDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        texDesc.SampleDesc.Count = 1;
        texDesc.Usage = D3D11_USAGE_DYNAMIC;
        texDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        texDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

        ID3D11Texture2D* cpuTexture = nullptr;
        if (FAILED(device->CreateTexture2D(&texDesc, nullptr, &cpuTexture)))
            RAISE_ERROR("CreateTexture2D (cpuTexture) failed");

        ID3D11ShaderResourceView* srv = nullptr;
        if (FAILED(device->CreateShaderResourceView(cpuTexture, nullptr, &srv)))
            RAISE_ERROR("CreateShaderResourceView failed");

        D3D11_SAMPLER_DESC sampDesc = {};
        sampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        sampDesc.AddressU = sampDesc.AddressV = sampDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;

        ID3D11SamplerState* sampler = nullptr;
        if (FAILED(device->CreateSamplerState(&sampDesc, &sampler)))
            RAISE_ERROR("CreateSamplerState failed");

        ID3DBlob* vsBlob = CompileShader(g_VS, "main", "vs_5_0");
        ID3DBlob* psBlob = CompileShader(g_PS, "main", "ps_5_0");

        ID3D11VertexShader* vs = nullptr;
        ID3D11PixelShader* ps = nullptr;
        if (FAILED(device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &vs)))
            RAISE_ERROR("CreateVertexShader failed");
        if (FAILED(device->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &ps)))
            RAISE_ERROR("CreatePixelShader failed");
        vsBlob->Release(); psBlob->Release();

        // Main loop
        bool running = true;
        SDL_Event e;
        uint64_t frameCounter = 0;
        uint64_t ticksOnStart = SDL_GetTicks();

        constexpr double PIXELS_PER_DOUBLING = 250;
        constexpr double MIDPOINT_NITS = 20;
        while (running) {
            frameCounter++;
            while (SDL_PollEvent(&e)) if (e.type == SDL_EVENT_QUIT) running = false;

            D3D11_MAPPED_SUBRESOURCE mapped;
            if (FAILED(context->Map(cpuTexture, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
                RAISE_ERROR("Map(cpuTexture) failed");

            float* pixels = (float*)mapped.pData;
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    
                    double xp = x / double(w);
                    double doublings = (x - double(w) / 2) / PIXELS_PER_DOUBLING;
                    double nits = MIDPOINT_NITS * pow(2, doublings);
                    float r = nits / 80;
                    float g = r;
                    float b = r;
                    float a = 1;
                    //int ind = x + y * (mapped.RowPitch / (4*sizeof(float)));
                    int ind = (y * w + x)*4;
                    pixels[ind + 0] = r;
                    pixels[ind + 1] = g;
                    pixels[ind + 2] = b;
                    pixels[ind + 3] = a;
                }
            }
            context->Unmap(cpuTexture, 0);

            context->OMSetRenderTargets(1, &rtv, nullptr);
            float clear[4] = { 0,0,0,1 };
            context->ClearRenderTargetView(rtv, clear);

            context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            context->VSSetShader(vs, nullptr, 0);
            context->PSSetShader(ps, nullptr, 0);
            context->PSSetShaderResources(0, 1, &srv);
            context->PSSetSamplers(0, 1, &sampler);

            context->Draw(3, 0);
            if (FAILED(swapChain->Present(1, 0)))
                RAISE_ERROR("swapChain->Present failed");

            if (frameCounter % 100 == 0)
            {
                uint64_t ticksOnEnd = SDL_GetTicks();
                uint64_t delta = ticksOnEnd - ticksOnStart;
                ticksOnStart = ticksOnEnd;
                double fps = 100/(delta / (1000.0));
                std::cout << fps << " FPS\n";
            }
        }

        // Cleanup
        vs->Release(); ps->Release(); sampler->Release(); srv->Release();
        cpuTexture->Release(); rtv->Release(); swapChain->Release();
        context->Release(); device->Release();

        SDL_DestroyWindow(window);
        SDL_Quit();
        return 0;
    }
    catch (const std::exception& e)
    {
        std::stringstream ss;
        ss << e.what() << "\nSDL error reports: " << SDL_GetError() << "\n";
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Error", ss.str().c_str(), nullptr);
    }
}
