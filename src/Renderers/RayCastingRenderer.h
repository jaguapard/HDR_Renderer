#pragma once
#include "RendererBase.h"
#include "../Vec.h"
#include <vector>

class RayCastingRenderer : public RendererBase
{
public:
	virtual void loadScene(std::string path, std::string mode);
	virtual void renderFrame(const GameSettings& settings);
protected:
	struct TexVertex
	{
		Vec4f space, diffuse;
	};
	struct Triangle
	{
		TexVertex tv[3];
	};
	struct Model
	{
		std::vector<Triangle> triangles;
		int textureIndex = -1;
	};

	std::vector<Model> sceneModels;
};