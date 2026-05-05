#pragma once
#include "../RendererBase.h"
#include "../../Vec.h"
#include <vector>
#include <memory>
#include "../../BoundingBox.h"

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

	struct OctreeContent
	{
		int modelIndex = -1, triangleIndex = -1;
	};
	struct OctreeNode
	{
		static constexpr int CHILD_COUNT = 8;
		BoundingBox bbox;
		std::unique_ptr<OctreeNode> children[CHILD_COUNT] = { nullptr };
		std::vector<OctreeContent> contents;

		BoundingBox getBoundingBoxForChildIndex(int i) const;
		bool tryAddTriangle(int modelIndex, int triangleIndex, const RayCastingRenderer& rend);
	};

	class Octree
	{
	public:
		Octree() = default;
		Octree(RayCastingRenderer& rend);
		std::unique_ptr<OctreeNode> root = nullptr;
	private:
		RayCastingRenderer* rend;		
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