#include <vector>
#include "../RendererBase.h"
#include <map>
#include "CoordinateTransformer.h"
#include <array>
#include "TextureManager.h"

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

	/*
	struct VerticeArray
	{
		std::vector<float> x, y, z, u, v;
	};*/
	//Inclusive
	struct SequentialRange
	{
		int min, max;
	};
	struct Model
	{
		Triangle_Store triangleStore;
		int diffuseMapIndex = -1;
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

	struct RenderJob_Store
	{
		std::array<std::vector<float>, 3> x, y, z, u, v;
		std::vector<float> minX, maxX, minY, maxY, rcpSignedArea;
		std::vector<int> modelIndex, firstThread, lastThread; //inclusive!
		size_t size() const;
		void clear();
		void resize(size_t ind, bool overprovision = true);
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
	std::vector<Rasterizing::RenderJob_Store> renderJobsFromThreads;

	std::vector<std::vector<Rasterizing::ModelSlice>> makeModelSliceList() const;
	void doTransformationsAndClipping(int threadIndex);

	void drawRenderJobs(int threadIndex);

	std::vector<float> zBuffer;

	CoordinateTransformer ctr;
	Rasterizing::TextureManager textureManager;
};