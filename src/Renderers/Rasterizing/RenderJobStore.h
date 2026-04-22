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
}