#pragma once
#include "../RendererBase.h"
#include "../../Vec.h"
#include <vector>
#include <memory>
#include "../../BoundingBox.h"
#include "Octree.h"
class RayCastingRenderer;

namespace RayCasting
{
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
}

class RayCastingRenderer : public RendererBase
{
public:
	virtual void loadScene(RendererLoadSceneData scd) override;
	virtual void renderFrame(const GameSettings& settings) override;
protected:
	std::vector<RayCasting::Model> sceneModels;
	RayCasting::Octree octree;
	friend class RayCasting::Octree;
	friend class RayCasting::OctreeNode;
};