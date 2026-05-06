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

	struct TraceResults
	{
		float32x16 t = INFINITY, worldBarycentrics[3], textureCoords[2];
		int32x16 modelIndices, triangleIndices;
		Vec4_f32x16 normals;
		Mask16 raysHit = 0;
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

	RayCasting::TraceResults traceRays(Vec4_f32x16 rayOrigins, Vec4_f32x16 rayDirs, Mask16 activeRays, bool shadowRays, uint32_t threadIndex);
};