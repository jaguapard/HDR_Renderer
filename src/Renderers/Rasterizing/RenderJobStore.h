#pragma once
#include <memory>
#include <vector>
#include <optional>
#include "../../Vec.h"
#include "../../BlockStore.h"
namespace Rasterizing
{
	struct RenderJob
	{
		float x[3], y[3], z[3], u[3], v[3], nx[3], ny[3], nz[3], rcpSignedArea;
		uint32_t diffuseMapIndex;
		RenderJob() {};
	};

	struct RenderJobLight
	{
		float screenX[3], screenY[3], rcpZ[3], rcpSignedArea;
		uint32_t diffuseMapIndex, vertexIndex[3];
		RenderJobLight() {};
	};

	struct VertexPack16;
	struct DrawCommand;
	class Vertice_Store;
	//uses block list to store render jobs with stable pointers
	struct RenderJobStore
	{
		RenderJobStore() = default;
		RenderJobStore(const Rasterizing::Vertice_Store* vertStore);
		//std::array<VertexPack16, 3> loadVertices16(size_t firstInd, Mask16 mask) const;
		//VertexPack16 gatherVertices16(int32x16 indices) const;

		void addMany(const VertexPack16* pStart, const VertexPack16* pEnd, const float32x16& rcpSignedArea, const int32x16& diffuseMapIndex, Mask16 activeElementsMask, const DrawCommand& subInfo, const int32x16* vertexIndexStart, const int32x16* vertexIndexEnd, bool areClipped);

		size_t size() const;
		RenderJob operator[](size_t i);
		void reset();
		void clear();
		const Rasterizing::Vertice_Store* verticeStore = nullptr;
	private:
		BlockStore<RenderJob, 1024> heavyRenderJobs;
		BlockStore<RenderJobLight, 8192> lightRenderJobs;
	};
}