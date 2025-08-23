#include <SDL3/SDL.h>
#include <exception>
#include <iostream>
//#include <SDL3/SDL_image.h>
#pragma comment(lib, "SDL3.lib")
//#pragma comment(lib, "SDL2_image.lib")

#undef main

void main()
{
	try
	{
		//SDL_Init()
		int w = 1920;
		int h = 1080;
		//SDL_Window* wnd = SDL_CreateWindow("HDR Renderer", w, h, SDL_WINDOW_VULKAN);
		SDL_Window* wnd = SDL_CreateWindow("HDR Renderer", w, h, 0);
		if (!wnd) throw std::runtime_error("Window creation");

		SDL_PropertiesID props = SDL_CreateProperties();
		if (!props) throw std::runtime_error("props creation");
		//SDL_SetStringProperty(props, SDL_PROP_RENDERER_DRIVER_STRING, "vulkan");
		if (!SDL_SetNumberProperty(props, SDL_PROP_RENDERER_CREATE_OUTPUT_COLORSPACE_NUMBER,
			SDL_COLORSPACE_SRGB_LINEAR)) throw std::runtime_error("set props: SRGB linear color space");
		if (!SDL_SetPointerProperty(props, SDL_PROP_RENDERER_CREATE_WINDOW_POINTER, wnd)) throw std::runtime_error("set props: Window pointer");
		//if (!SDL_SetNumberProperty(props, "window", (Sint64)wnd)) throw std::runtime_error("set props: Window pointer 2");

		SDL_Renderer* rend = SDL_CreateRendererWithProperties(props);
		if (!rend) throw std::runtime_error("Renderer creation");
		//SDL_DestroyProperties(props);

		//SDL_CreateRendererWithProperties()
		///SDL_SetGPUSwapchainParameters(nullptr, wnd, SDL_GPU_SWAPCHAINCOMPOSITION_HDR_EXTENDED_LINEAR, SDL_GPU_PRESENTMODE_VSYNC);

		SDL_Texture* t = SDL_CreateTexture(rend, SDL_PIXELFORMAT_RGBA128_FLOAT, SDL_TEXTUREACCESS_STREAMING, w, h);
		if (!t) throw std::runtime_error("texture creation");
		//SDL_COLORSPACE_SRGB_LINEAR
		int a = 0;

		float* pixels;
		int pitch;
		SDL_LockTexture(t, nullptr, (void**)&pixels, &pitch);
		for (int y = 0; y < h; ++y)
		{
			for (int x = 0; x < w; ++x)
			{
				size_t pix_ind = y * w + x;
				size_t arr_ind = pix_ind * 4;
				pixels[arr_ind] = y < h / 2 ? 0.5 : y < h / 4 ? 1 : 6;
				pixels[arr_ind + 3] = 1;
			}
		}
		SDL_UnlockTexture(t);
		SDL_RenderClear(rend);
		SDL_RenderTexture(rend, t, nullptr, nullptr);
		SDL_RenderPresent(rend);
	}
	catch (const std::exception& e)
	{
		std::cout << "Error: " << e.what() << ": " << SDL_GetError() << "\n";
	}
	system("pause");
	
	//SDL_SetSurfaceColorspace()
}