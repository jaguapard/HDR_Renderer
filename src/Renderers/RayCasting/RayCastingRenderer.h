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
		Vec4f space, diffuse, normal;
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

	struct TraceResult
	{
		float t = INFINITY, worldBarycentrics[3], textureCoords[2];
		uint32_t modelIndex, triangleIndex;
		Vec4f normal;
	};
}

class RayCastingRenderer : public RendererBase
{
public:
	virtual void loadScene(RendererLoadSceneData scd) override;
	virtual void renderFrame(const GameSettings& settings) override;
protected:
	friend class RayCasting::Octree;
	friend struct RayCasting::OctreeNode;

	std::vector<RayCasting::Model> sceneModels;
	RayCasting::Octree octree;
	static inline constexpr bool discardUntexturedTriangles = true;

	RayCasting::TraceResults traceRays(Vec4_f32x16 rayOrigins, Vec4_f32x16 rayDirs, Mask16 activeRays, bool shadowRays, uint32_t threadIndex);
	RayCasting::TraceResult traceRay(Vec4f rayOrigin, Vec4f rayDir, bool shadow, uint32_t threadIndex);
};