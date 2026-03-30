#include <SDL3/SDL.h>
#include <SDL3/SDL_system.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <iostream>
#include <sstream>
#include "C_Input.h"
#include "Renderers\RayCasting\RayCastingRenderer.h"
#include "Renderers\Rasterizing\RasterizingRenderer.h"
#include "GameSettings.h"
#include <bob/Matrix4.h>
#include "Threadpool.h"
#include "OSD.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "SDL3.lib")
#pragma comment(lib, "SDL3_image.lib")
#pragma comment(lib, "SDL3_ttf.lib")
#define StatCount()

void* operator new(size_t n)
{
    StatCount(statsman.memory.allocsByNew++);
#ifdef __AVX512F__
    constexpr size_t alignmentRequirement = 64;
#elif __AVX__
    constexpr size_t alignmentRequirement = 32;
#else 
    constexpr size_t alignmentRequirement = 16;
#endif

    return _aligned_malloc(n, alignmentRequirement);
}

void operator delete(void* block)
{
    StatCount(statsman.memory.freesByDelete++);
    return _aligned_free(block);
}

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

std::string vec2str(Vec4f v, int componentsToPrint = 4)
{
    std::string ret;
    for (int i = 0; i < componentsToPrint; ++i)
    {
        ret += std::to_string(v[i]) + " ";
    }
    ret.pop_back();
    return ret;
}

Threadpool threadpool;

int main(int argc, char* argv[]) 
{
    try
    {
        if (!SDL_Init(SDL_INIT_VIDEO)) RAISE_ERROR("SDL_Init failed");
        C_Input::getInstance();
     
        int w = 2560;
        int h = 1440;
        SDL_Window* window = SDL_CreateWindow("SDL3 + D3D11 Pixel Display", w, h, 0);
        if (!window) RAISE_ERROR("SDL_CreateWindow failed");

        SDL_PropertiesID props = SDL_GetWindowProperties(window);
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
#ifdef NDEBUG
        texDesc.Width = w;
        texDesc.Height = h;
#else
        texDesc.Width = w/10;
        texDesc.Height = h/10;
#endif
        texDesc.MipLevels = 1;
        texDesc.ArraySize = 1;
        texDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT; //TODO: HUGE floats, change to less pcie bandwith crushing format
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
        uint64_t frameCounter = 0, oldFrameCounter = 0;
        uint64_t ticksOnStart = SDL_GetTicks();

        std::shared_ptr<RendererBase> currentRenderer = std::make_shared<RasterizingRenderer>();
        //currentRenderer->loadScene("scenes/old_sponza.bmdl", "bmdl");
        currentRenderer->loadScene("scenes/Sponza/old_sponza.bmdl", "bmdl");
        constexpr double PIXELS_PER_DOUBLING = 250;
        constexpr double MIDPOINT_NITS = 20;

        GameSettings gs;
        //gs.camPos = { -7.482602, -85.107704, 75.298897, 0.000000 };
        //gs.camAng = { -6.293743, 0.000000, -0.652987, 0.000000 };
        gs.camPos = { 20,-20,-100 };
        gs.camAng = { 0,0,0 };
        gs.outputTextureParams = texDesc;
        gs.threadpool = &threadpool;
        OSD osd;

        while (running) {
            osd.registerFrameBegin();
            frameCounter++;
            C_Input& inp = C_Input::getInstance();
            inp.beginNewFrame();
            while (SDL_PollEvent(&e))
            {
                inp.handleEvent(e);

                if (gs.mouseCaptured && e.type == SDL_EVENT_MOUSE_MOTION)
                {
                    gs.camAng.y += e.motion.xrel * 1e-3;
                    gs.camAng.z -= e.motion.yrel * 1e-3;
                }

                if (e.type == SDL_EVENT_QUIT) {
                    running = false; break;
                }
            }

            D3D11_MAPPED_SUBRESOURCE mapped;
            if (FAILED(context->Map(cpuTexture, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
                RAISE_ERROR("Map(cpuTexture) failed");

            //TODO: change coordinate system, so Y = up/down, Z = into the screen, X = left/right
            Matrix4 rotation = Matrix4::rotationXYZ(gs.camAng);
            /*
            Vec4f newRayForward = rotation * Vec4f(0, 0, 1, 0);
            Vec4f newRayRight = rotation * Vec4f(1, 0, 0, 0);
            Vec4f newRayDown = rotation * Vec4f(0, 1, 0, 0);*/

            /*
            Vec4f newRayForward = Vec4f(rotation[1][0], rotation[1][1], rotation[1][2], 0);
            Vec4f newRayRight = Vec4f(rotation[0][0], rotation[0][1], rotation[0][2], 0);
            Vec4f newRayDown = Vec4f(rotation[2][0], rotation[2][1], rotation[2][2], 0);*/

            Vec4f newRayForward = Vec4f(rotation[2][0], rotation[2][1], rotation[2][2], 0);
            Vec4f newRayRight = Vec4f(rotation[0][0], rotation[0][1], rotation[0][2], 0);
            Vec4f newRayDown = Vec4f(-rotation[1][0], -rotation[1][1], -rotation[1][2], 0);

            gs.viewMatrix = rotation;
            gs.forward = newRayForward;
            gs.right = newRayRight;
            gs.down = newRayDown;

            Vec4f camAdd = { 0,0,0,0 };
            if (inp.isButtonHeld(SDL_SCANCODE_W)) camAdd += newRayForward;
            if (inp.isButtonHeld(SDL_SCANCODE_S)) camAdd -= newRayForward;
            if (inp.isButtonHeld(SDL_SCANCODE_A)) camAdd -= newRayRight;
            if (inp.isButtonHeld(SDL_SCANCODE_D)) camAdd += newRayRight;
            if (inp.isButtonHeld(SDL_SCANCODE_Z)) camAdd -= Vec4f(0, 1, 0);
            if (inp.isButtonHeld(SDL_SCANCODE_X)) camAdd += Vec4f(0, 1, 0);
            if (float len = camAdd.len())
            {
                camAdd = camAdd / len * gs.flySpeed;
            }
            gs.camPos += camAdd;

            if (inp.wasButtonPressedOnThisFrame(SDL_SCANCODE_LCTRL))
            {
                gs.mouseCaptured ^= 1;
                SDL_SetWindowRelativeMouseMode(window, gs.mouseCaptured);
            }

            //std::cout << "cam pos" << vec2str(gs.camPos) << ", camAng: " << vec2str(gs.camAng) << "\n";
            gs.graphicsOutputBuffer = mapped.pData;
            currentRenderer->renderFrame(gs);
            std::cout << osd.composeString() << "\n";
            osd.registerFrameDone();
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
            if (FAILED(swapChain->Present(0, 0)))
                RAISE_ERROR("swapChain->Present failed");

            uint64_t ticksOnEnd = SDL_GetTicks();
            /*
            if (ticksOnEnd - ticksOnStart >= 1000)
            {
                uint64_t delta = ticksOnEnd - ticksOnStart;
                ticksOnStart = ticksOnEnd;

                double fps = (frameCounter-oldFrameCounter) / (delta / (1000.0));
                std::cout << fps << " FPS\n";
                oldFrameCounter = frameCounter;
                
            }*/
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
