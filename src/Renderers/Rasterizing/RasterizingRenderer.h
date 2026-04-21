#include <vector>
#include "../RendererBase.h"
#include <map>
#include "CoordinateTransformer.h"
#include <array>
#include "TextureManager.h"
#include "../../helpers.h"
#include "RenderJobStore.h"
#include "primitives.h"
namespace Rasterizing
{
	
	/*
	struct Triangle
	{
		uint32_t vertIndex[3];
		uint32_t diffuseMapVertIndex[3]; //different triangles may have different texture coords, like cube's edges if all faces are distinct pictures

		//static std::vector<float> xStore, yStore, zStore, uStore, vStore;
	};*/

	//Inclusive
	struct SequentialRange
	{
		int min, max;
	};
	struct Model
	{
		SequentialRange globalTriangleRange;
	};

	struct ModelGlobalTrianglesDescriptor
	{
		int modelIndex;
		//std::vector<SequentialRange> global_xyz_indices, global_uv_indices;
		//std::vector<SequentialRange> 
	};	
}
class RasterizingRenderer : public RendererBase
{
public:
	virtual void loadScene(RendererLoadSceneData scd);
	virtual void renderFrame(const GameSettings& settings);
	//virtual void handleInputEvent(const SDL_Event& ev, C_Input& input);
private:
	void loadScene(bool debugScene);
	void clearScene();
	bool singleTriangleDebugMode = false;
	Rasterizing::ShadingMode shadingMode = Rasterizing::ShadingMode::SMOOTH;
	std::vector<Rasterizing::Model> sceneModels;
	Rasterizing::Vertice_Store original_verticeStore;
	Rasterizing::Triangle_Store original_triangleStore;

	std::array<Rasterizing::DrawCommand, 2> drawCommands;

	std::vector<uint8_t> postTransformationsActiveMasks;
	const GameSettings* currGs;

	void joinMainWithShadowMap(int threadIndex);

	std::vector<float> zBuffer, shadowMap_zBuffer;
	std::vector<uint32_t> deferredTriangleIndices;
	std::array<std::vector<Rasterizing::TriangleIndexStore>, 2> trianglesByZones;

	struct TriangleBatch
	{
		Rasterizing::Vertice_Store vertexData[3];
		std::vector<float> minX, minY, maxX, maxY, rcpSignedArea;
		std::vector<int> diffuseMapIndex, triangleIndex;
		float batchMinX, batchMinY, batchMaxX, batchMaxY;
		int batchSize = 0;
		void resize(size_t newSize)
		{
			for (auto& it : vertexData) it.resize(newSize);
			minX.resize(newSize);
			minY.resize(newSize);
			maxX.resize(newSize);
			maxY.resize(newSize);
			diffuseMapIndex.resize(newSize);
			triangleIndex.resize(newSize);
			rcpSignedArea.resize(newSize);
		}
	};

	void performNearPlaneClipping(float clippingZ, std::array<Rasterizing::VertexPack16, 6>& outVerts, int32x16 behindPlaneCount, std::array<Mask16, 3> behindPlaneMasks) const;


	void drawTriangleBatch(const TriangleBatch& batch, const int threadIndex, const Rasterizing::DrawCommand& drawCmd);
	void workerRoutine(const int threadIndex);
	Rasterizing::TextureManager textureManager;
	uint64_t lastTicks = SDL_GetTicksNS(), totalTicks = 0;

	Vec4f lightPos, lightAng, lightColor, skyColor = Vec4f(0.3, 0.7, 1, 1);
	//float lightIntensity, skyLightIntensity;
	float ambientLightIntensity = 0.3, lightIntesity = 3;
	bool drawShadowMapDebug = false, skipTrianglesWithFallbackTexure = true, useShadowMapBias = false, useShadowMapFrontFaceCulling = true;
	Rasterizing::FaceCullingType faceCullingType = Rasterizing::FaceCullingType::NONE;
};