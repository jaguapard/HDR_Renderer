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

	struct Vertice_Store
	{
		std::vector<float> x, y, z, u, v, nx, ny, nz;
		uint32_t insert(float x, float y, float z, float u, float v, float nx, float ny, float nz);
		size_t size() const;
	private:
		//TODO: if gonna make this dynamic, make it cleanable and check
		std::map<std::tuple<float, float, float, float, float, float, float, float>, uint32_t> dedup;
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
}
class RasterizingRenderer : public RendererBase
{
public:
	virtual void loadScene(RendererLoadSceneData scd);
	virtual void renderFrame(const GameSettings& settings);
	//virtual void handleInputEvent(const SDL_Event& ev, C_Input& input);
private:
	Rasterizing::ShadingMode shadingMode = Rasterizing::ShadingMode::SMOOTH;
	std::vector<Rasterizing::Model> sceneModels;
	Rasterizing::Vertice_Store original_verticeStore;
	Rasterizing::Triangle_Store original_triangleStore;

	std::array<Rasterizing::DrawCommand, 2> drawCommands;

	std::vector<uint8_t> postTransformationsActiveMasks;
	const GameSettings* currGs;
	std::vector<std::vector<Rasterizing::ModelSlice>> modelSlicesForThreads;

	std::vector<std::vector<Rasterizing::ModelSlice>> makeModelSliceList() const;
	void doTransformationsAndClipping(int threadIndex);
	void drawRenderJobs(int threadIndex);
	void joinMainWithShadowMap(int threadIndex);

	std::vector<float> zBuffer, shadowMap_zBuffer;
	std::vector<uint64_t> deferrendRenderJobPtrs;
	std::vector<Rasterizing::RenderJobStore> mainRenderJobs, shadowMapRenderJobs;

	Rasterizing::TextureManager textureManager;
	uint64_t lastTicks = SDL_GetTicksNS(), totalTicks = 0;

	Vec4f lightPos, lightAng, lightColor, skyColor = Vec4f(0.3, 0.7, 1, 1);
	//float lightIntensity, skyLightIntensity;
	float ambientLightIntensity = 0.3, lightIntesity = 3;
	bool drawShadowMapDebug = false, missingTexturesSetToPlaceholder = true;
	Rasterizing::FaceCullingType faceCullingType = Rasterizing::FaceCullingType::NONE;
};