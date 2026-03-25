#pragma once
#include "RendererBase.h"

class RayCastingRenderer : public RendererBase
{
public:
	virtual void loadScene(std::string path, std::string mode);
	virtual void renderFrame(const GameSettings& settings);
protected:
};