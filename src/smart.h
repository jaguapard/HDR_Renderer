#pragma once
#include <memory>
#include <SDL3\SDL.h>
/*
#include <SDL3\SDL_ttf.h>


struct FontDeleter
{
	void operator()(TTF_Font* f)
	{
		TTF_CloseFont(f);
	}
};
*/
struct SurfaceDeleter
{
	void operator()(SDL_Surface* s)
	{
		SDL_DestroySurface(s);
	}
};

//typedef std::unique_ptr<TTF_Font, FontDeleter> Smart_Font;
typedef std::unique_ptr<SDL_Surface, SurfaceDeleter> Smart_Surface;