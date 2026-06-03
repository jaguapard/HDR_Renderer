#include <SDL3/SDL.h>
#include <SDL3/SDL_system.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <iostream>
#include <sstream>
#include "C_Input.h"
#include "Renderers\RayCasting\RayCastingRenderer.h"
#include "Renderers\Rasterizing\RasterizingRenderer.h"
#include "Renderers\HardwareRasterizingRenderer.h"
#include "GameSettings.h"
#include <bob/Matrix4.h>
#include "Threadpool.h"
#include "OSD.h"
#include "Statsman.h"
#include "LUTMan.h"
#include <wrl/client.h>
#include "libs.h"
#include "Graphics.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "SDL3.lib")
#pragma comment(lib, "SDL3_image.lib")
#pragma comment(lib, "SDL3_ttf.lib")
#define StatCount()

void* operator new(size_t n)
{
    if (Statsman::ENABLED && !Statsman::statsmenForThreads.empty()) Statsman::statsmenForThreads.back().allocsByNew++;
    //StatCount(statsman.memory.allocsByNew++);
#ifdef __AVX512F__
    constexpr size_t alignmentRequirement = 64;
#elifdef __AVX__
    constexpr size_t alignmentRequirement = 32;
#else 
    constexpr size_t alignmentRequirement = 16;
#endif

    return _aligned_malloc(n, alignmentRequirement);
}

void operator delete(void* block)
{
    if (Statsman::ENABLED && !Statsman::statsmenForThreads.empty()) Statsman::statsmenForThreads.back().freesByDelete++;
    return _aligned_free(block);
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
    SDL_Window* window;
    try
    {
        Statsman::statsmenForThreads.resize(threadpool.getWorkerCount() + 1); //last one for main thread
        if (!SDL_Init(SDL_INIT_VIDEO)) RAISE_ERROR("SDL_Init failed"); 
        C_Input::getInstance();
        TextureManager::getInstance();

        int w = 2560;
        int h = 1440;
        Graphics graphics(w, h);
        window = graphics.window;

        TTF_Init();

        CPU_Renderer_Context cpuRenderingCtx = graphics.makeCPURendererContext();

        bool running = true;
        SDL_Event e;
        uint64_t frameCounter = 0, oldFrameCounter = 0;
        uint64_t ticksOnStart = SDL_GetTicks();

        enum class SceneEnum
        {
            OLD_SPONZA,
            NEW_SPONZA,
            //SINGLE_TRIANGLE_DEBUG,
            COUNT,
        };

        SceneEnum currSceneEnum = SceneEnum::OLD_SPONZA;
        std::shared_ptr<RendererBase> currentRenderer;
        RendererLoadSceneData oldSponza, newSponza;
        oldSponza.files = { { "H:/Sponza goodies/old_sponza/old_sponza.obj", "obj" } };
        /*newSponza.files = {{"H:/Sponza goodies/main1_sponza/NewSponza_Main_Yup_003.fbx", ""},
            { "H:/Sponza goodies/pkg_a_curtains/NewSponza_Curtains_FBX_YUp.fbx", "" },
            { "H:/Sponza goodies/pkg_c1_trees/NewSponza_CypressTree_FBX_YUp.fbx", "" },
            { "H:/Sponza goodies/pkg_b_ivy/NewSponza_IvyGrowth_FBX_YUp.fbx", "" }
        };*/
        newSponza.files = {
            {"H:/Sponza goodies/main1_sponza/new_sponza.bmdl2", "bmdl"},
            {"H:/Sponza goodies/pkg_a_curtains/curtains.bmdl2", "bmdl"},
            {"H:/Sponza goodies/pkg_c1_trees/tree.bmdl2", "bmdl"},
            {"H:/Sponza goodies/pkg_b_ivy/ivy.bmdl2", "bmdl"},
        };
       // currentRenderer->loadScene(oldSponza);
      
        constexpr double PIXELS_PER_DOUBLING = 250;
        constexpr double MIDPOINT_NITS = 20;

        GameSettings gs;
        //gs.camPos = { -7.482602, -85.107704, 75.298897, 0.000000 };
        //gs.camAng = { -6.293743, 0.000000, -0.652987, 0.000000 };
        //gs.camPos = { 20,-20,-100 };
        gs.camPos = { 1215.152100, 42.281734, 24.533436 };
        gs.camAng = { 0.000000, -1.588021, -0.288000 };
        gs.outputTextureW = cpuRenderingCtx.outputTextureW;
        gs.outputTextureH = cpuRenderingCtx.outputTextureH;
        gs.screenW = cpuRenderingCtx.screenW;
        gs.screenH = cpuRenderingCtx.screenH;
        gs.threadpool = &threadpool;
        uint32_t OSD_fontSize = std::max<uint32_t>(6, float(cpuRenderingCtx.outputTextureH) / 72);
        OSD osd(OSD_fontSize);
        uint64_t lastOsdInfoTicks = SDL_GetTicksNS();
        double lastOsdDrawMs = NAN;
        
        std::shared_ptr<RendererBase> scheduledRendererChange = nullptr;
        uint64_t prevFrameTicks = SDL_GetTicksNS();
        bool sceneReloadNeeded = false;
        bool skipThisFrame = false;
        while (running) {
            uint64_t thisFrameTicks = SDL_GetTicksNS();
            double dt = (thisFrameTicks - prevFrameTicks) / 1e9;
            dt = std::clamp(dt, 0.0, 0.1);
            prevFrameTicks = thisFrameTicks;
            gs.gameTime += dt;
            gs.gameTimeLastDt = dt;
            //std::cout << gs.gameTime << "sec \n";
            
            for (auto& it : Statsman::statsmenForThreads) it.reset();
            if (!currentRenderer && !scheduledRendererChange)
            {
                scheduledRendererChange = std::make_shared<RasterizingRenderer>();
                sceneReloadNeeded = true;
                skipThisFrame = true;
                //else if (dynamic_cast<RasterizingRenderer*>(currentRenderer.get())) scheduledRendererChange = std::make_shared<RayCastingRenderer>();
                //else if (dynamic_cast<RayCastingRenderer*>(currentRenderer.get())) scheduledRendererChange = std::make_shared<RasterizingRenderer>();
                //else throw std::runtime_error("Unknown renderer type ")
            }
            if (scheduledRendererChange) //change renderer if it's scheduled, or create default one if it doesn't exist
            {
                graphics.reset(); //to not force renderers to clean up their state, the main will reset it via a call to this.
                cpuRenderingCtx = graphics.makeCPURendererContext();
                currentRenderer = scheduledRendererChange;
                //now that renderer is OK to go live, it can be set up. Putting setup in constructor will make graphics.reset() 
                // call clean all of it's setup, and structuring everything to create renderer instance just in time is annoying. 
                // Thus, the setup method was born to mitigate this by explicitly marking the "OK to setup" stage.
                currentRenderer->setup();
                scheduledRendererChange = nullptr;
                sceneReloadNeeded = true;
                skipThisFrame = true;
            }
            if (sceneReloadNeeded)
            {
                skipThisFrame = true;
                RendererLoadSceneData ldscd;
                if (currSceneEnum == SceneEnum::OLD_SPONZA) ldscd = oldSponza;
                else if (currSceneEnum == SceneEnum::NEW_SPONZA) ldscd = newSponza;
                else throw std::runtime_error("Attempted to load unknown scene in scheduled renderer change!");
                currentRenderer->loadScene(ldscd); //TODO: make it clear!
                sceneReloadNeeded = false;
            }
            if (skipThisFrame)
            {
                skipThisFrame = false;
                continue;
            }

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
                    gs.camAng.z += e.motion.yrel * 1e-3;
                }

                if (e.type == SDL_EVENT_MOUSE_WHEEL)
                {
                    double scrollPower = pow(1.05, -e.wheel.y);
                    double tanFov = tan(gs.verticalFovDegrees * PI / 180);
                    tanFov *= scrollPower;
                    gs.verticalFovDegrees = atan(tanFov)*180/PI;
                }

                if (e.type == SDL_EVENT_QUIT) {
                    running = false; break;
                }
            }
            if (!running) break;

            if (inp.wasButtonPressedOnThisFrame(SDL_SCANCODE_KP_3)) //swap renderers
            {
                if (dynamic_pointer_cast<RasterizingRenderer>(currentRenderer)) scheduledRendererChange = std::make_shared<HardwareRasterizingRenderer>();
                else if (dynamic_pointer_cast<HardwareRasterizingRenderer>(currentRenderer)) scheduledRendererChange = std::make_shared<RayCastingRenderer>();
                else scheduledRendererChange = std::make_shared<RasterizingRenderer>();
                continue;
            }

            if (inp.wasButtonPressedOnThisFrame(SDL_SCANCODE_KP_1)) Statsman::ENABLED ^= 1;
            if (inp.wasCharPressedOnThisFrame('0') || inp.wasCharPressedOnThisFrame('9'))
            {
                currSceneEnum = inp.wasCharPressedOnThisFrame('0') ? SceneEnum::NEW_SPONZA : SceneEnum::OLD_SPONZA;
                sceneReloadNeeded = true;
                continue;
            }

            //TODO: change coordinate system, so Y = up/down, Z = into the screen, X = left/right
            Matrix4 rotation = Matrix4::rotationXYZ(gs.camAng);

            Vec4f newRayForward = Vec4f(rotation[2][0], rotation[2][1], rotation[2][2], 0);
            Vec4f newRayRight = Vec4f(rotation[0][0], rotation[0][1], rotation[0][2], 0);
            Vec4f newRayDown = Vec4f(rotation[1][0], rotation[1][1], rotation[1][2], 0);

            gs.viewMatrix = rotation;
            gs.forward = newRayForward;
            gs.right = newRayRight;
            gs.down = newRayDown;

            Vec4f camAdd = { 0,0,0,0 };
            if (inp.isButtonHeld(SDL_SCANCODE_W)) camAdd += newRayForward;
            if (inp.isButtonHeld(SDL_SCANCODE_S)) camAdd -= newRayForward;
            if (inp.isButtonHeld(SDL_SCANCODE_A)) camAdd -= newRayRight;
            if (inp.isButtonHeld(SDL_SCANCODE_D)) camAdd += newRayRight;
            if (inp.isButtonHeld(SDL_SCANCODE_Z)) camAdd += Vec4f(0, 1, 0);
            if (inp.isButtonHeld(SDL_SCANCODE_X)) camAdd -= Vec4f(0, 1, 0);
            if (float len = camAdd.len())
            {
                camAdd = camAdd / len * gs.flySpeed * gs.gameTimeLastDt;
            }
            gs.camPos += camAdd;

            if (inp.wasButtonPressedOnThisFrame(SDL_SCANCODE_LCTRL))
            {
                gs.mouseCaptured ^= 1;
                SDL_SetWindowRelativeMouseMode(window, gs.mouseCaptured);
            }

            if (inp.wasButtonPressedOnThisFrame(SDL_SCANCODE_O)) gs.osdEnabled ^= 1;
            if (inp.wasButtonPressedOnThisFrame(SDL_SCANCODE_T)) gs.texturingEnabled ^= 1;
            if (inp.wasButtonPressedOnThisFrame(SDL_SCANCODE_V)) gs.vsyncEnabled ^= 1;
            gs.cameraPlane_zDist = 1 / (2 * tan(gs.verticalFovDegrees / (2 * 180 / PI))); //projection plane dist. Low dist = more divergent rays = high FoV.

            //std::cout << "cam pos" << vec2str(gs.camPos) << ", camAng: " << vec2str(gs.camAng) << "\n";
            bool isHardwareRenderer = dynamic_pointer_cast<HardwareRasterizingRenderer>(currentRenderer) != nullptr;
            if (!isHardwareRenderer)
            {
                D3D11_MAPPED_SUBRESOURCE mapped = graphics.CPURendering_OnFrameStart(cpuRenderingCtx);
                gs.graphicsOutputBuffer = mapped.pData;
            }
            currentRenderer->renderFrame(gs);
            osd.registerFrameDone(currentRenderer.get());

            uint64_t ticksBeforeOSD = SDL_GetTicksNS();
            std::vector<std::pair<std::string, std::string>> additionalInfo = {
                {"Camera pos", vec2str(gs.camPos)},
                {"Camera ang", vec2str(gs.camAng)},
                {"Render resolution", std::to_string(gs.outputTextureW) + "x" + std::to_string(gs.outputTextureH) },
                {"Output resolution", std::to_string(graphics.w) + "x" + std::to_string(graphics.h) },
                {"OSD draw time: ", std::to_string(lastOsdDrawMs) + " ms"},
            };
            if (gs.osdEnabled && !isHardwareRenderer) //VERY slow, impacts FPS a lot, disabled by default
            {
                float scalingFactor = 1;// std::max(0.5f, float(texDesc.Height) / h); //very small OSD becomes unreadable
                auto osdSurface = osd.draw(scalingFactor, additionalInfo);
                int osdW = osdSurface->w;
                int osdH = osdSurface->h;
                const Vec4f* osdPixels = (Vec4f*)(osdSurface->pixels);
                uint64_t* output = (uint64_t*)gs.graphicsOutputBuffer;
                for (int y = 0; y < std::min<int>(osdH, gs.outputTextureH); ++y)
                {
                    for (int x = 0; x < std::min<int>(osdW, gs.outputTextureW); ++x)
                    {
                        Vec4f osdPixel = osdPixels[y * osdW + x];
                        if (osdPixel.x > 0 || osdPixel.y > 0 || osdPixel.z > 0) //alpha is always 1 in returned surface for some reason, so work around by testing manually
                        {
                            int outInd = (gs.outputTextureH - y - 1) * gs.outputTextureW + x; //currently, y is backwards (0 = bottom of the screen, h-1 = top). Renderers don't care, so just flip OSD instead.
                            output[outInd] = _mm_extract_epi64(_mm_cvtps_ph(osdPixel, _MM_FROUND_TO_NEAREST_INT), 0);
                        }
                    }
                }
            }
            else
            {
                uint64_t currTicks = SDL_GetTicksNS();
                if (currTicks - lastOsdInfoTicks > 1e9)
                {
                    std::string text = osd.composeString(additionalInfo);
                    std::cout << text << "\n";
                    lastOsdInfoTicks = currTicks;
                    //std::cout << "\033[s" << text << "\033[u";
                    /*
                    int nls = 0;
                    for (auto& c : text) if (c == '\n') nls++;
                    std::cout << text + std::string(20, ' ') << "\n";
                    std::cout << std::string(nls+1, '\r');*/
                }
            }
            if (!isHardwareRenderer) graphics.CPURendering_Present(cpuRenderingCtx);
            uint64_t ticksAfterOSD = SDL_GetTicksNS();
            lastOsdDrawMs = (ticksAfterOSD - ticksBeforeOSD) / 1e6;
        }
    }
    catch (const std::exception& e)
    {
        std::stringstream ss;
        ss << e.what() << "\nSDL error reports: " << SDL_GetError() << "\n" << "strerror: " << strerror(errno) << "\n";
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Error", ss.str().c_str(), nullptr);
    }
    SDL_DestroyWindow(window);
    SDL_Quit();
    std::_Exit(0); //TODO: stop tokens in threadpool jthreads
    return 0;
}
