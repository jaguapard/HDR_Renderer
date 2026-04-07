#include <vector>
#include "../RendererBase.h"
#include <map>
#include "CoordinateTransformer.h"
#include <array>
#include "TextureManager.h"
#include "../../helpers.h"

namespace Rasterizing
{
	/*
	struct Triangle
	{
		uint32_t vertIndex[3];
		uint32_t diffuseMapVertIndex[3]; //different triangles may have different texture coords, like cube's edges if all faces are distinct pictures

		//static std::vector<float> xStore, yStore, zStore, uStore, vStore;
	};*/
	
	struct Vertice_Store
	{
		std::vector<float> x, y, z, u, v;
		uint32_t insert(float x, float y, float z, float u, float v);
		size_t size() const;
	private:
		//TODO: if gonna make this dynamic, make it cleanable and check
		std::map<std::tuple<float, float, float, float, float>, uint32_t> dedup;
	};

	struct Triangle_Store
	{
		std::vector<uint32_t> vertInd[3];
		size_t size() const;
		//std::vector<uint32_t> modelInd;
	};

	//Inclusive
	struct SequentialRange
	{
		int min, max;
	};
	struct Model
	{
		Triangle_Store triangleStore;
		int diffuseMapIndex = 0;
		bool noBackfaceCulling = true;
	};

	struct ModelSlice
	{
		int modelIndex;
		int modelTriangleIndexBegin;
		int modelTriangleIndexEnd;
	};

	struct ModelGlobalTrianglesDescriptor
	{
		int modelIndex;
		//std::vector<SequentialRange> global_xyz_indices, global_uv_indices;
		//std::vector<SequentialRange> 
	};


	struct VertexPack16
	{
		Vec4_f32x16 space;
		float32x16 u, v;
		//VertexPack16(float32x16 x, )
		VertexPack16() {};
		static __forceinline VertexPack16 lerpVertices(const VertexPack16& from, const VertexPack16& to, const float32x16& alpha)
		{
			VertexPack16 ret;
			ret.space.x = lerp(from.space.x, to.space.x, alpha);
			ret.space.y = lerp(from.space.y, to.space.y, alpha);
			ret.space.z = lerp(from.space.z, to.space.z, alpha);
			ret.u = lerp(from.u, to.u, alpha);
			ret.v = lerp(from.v, to.v, alpha);
			return ret;
		}

		static __forceinline VertexPack16 maskMove(const VertexPack16& zero, const VertexPack16& one, Mask16 mask)
		{
			VertexPack16 ret;
			ret.space.x = _mm512_mask_mov_ps(zero.space.x, mask, one.space.x);
			ret.space.y = _mm512_mask_mov_ps(zero.space.y, mask, one.space.y);
			ret.space.z = _mm512_mask_mov_ps(zero.space.z, mask, one.space.z);
			ret.u = _mm512_mask_mov_ps(zero.u, mask, one.u);
			ret.v = _mm512_mask_mov_ps(zero.v, mask, one.v);
			return ret;
		}

		static __forceinline VertexPack16 lerpToClippingZ(const VertexPack16& from, const VertexPack16& to, const float32x16& clippingZ)
		{
			float32x16 alpha = (clippingZ - from.space.z) / (to.space.z - from.space.z);
			return VertexPack16::lerpVertices(from, to, alpha);
		}
	};

	struct RenderJob_Store
	{
		std::array<std::vector<float>, 3> x, y, z, u, v;
		std::vector<float> rcpSignedArea;
		std::vector<int> modelIndex;
		size_t realSize = 0;
		size_t capacity = 0;

		size_t size() const;
		void clear(bool forceClear = false); //sets the realSize to 0. If forceClear is true also cleans the vectors.
		void makeSpace(size_t newSize);
		void add(const std::array<VertexPack16, 3>& verts, const float32x16& rcpSignedArea, const int32x16& modelIndex, Mask16 activeElementsMask);
	};

	struct RenderJob
	{
		float x[3], y[3], z[3], u[3], v[3];
		int modelIndex;
	};
}

class RasterizingRenderer : public RendererBase
{
public:
	virtual void loadScene(RendererLoadSceneData scd);
	virtual void renderFrame(const GameSettings& settings);
private:
	std::vector<Rasterizing::Model> sceneModels;
	Rasterizing::Vertice_Store original_verticeStore;
	Rasterizing::Triangle_Store original_triangleStore;

	std::vector<uint8_t> postTransformationsActiveMasks;
	const GameSettings* currGs;
	std::vector<std::vector<Rasterizing::ModelSlice>> modelSlicesForThreads;

	std::vector<std::vector<Rasterizing::ModelSlice>> makeModelSliceList() const;
	void doTransformationsAndClipping(int threadIndex);

	void drawRenderJobs(int threadIndex);

	std::vector<float> zBuffer;

	std::vector<std::vector<Rasterizing::RenderJob_Store>> renderJobsFromThreadToThread;

	CoordinateTransformer ctr;
	Rasterizing::TextureManager textureManager;
};