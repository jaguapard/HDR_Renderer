#include <vector>
#include "../RendererBase.h"
#include <map>
#include "CoordinateTransformer.h"
#include <array>
#include <memory>
#include <mutex>
#include <atomic>
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
	Rasterizing::VertexStore vertexStore;
	Rasterizing::TriangleStore triangleStore;
	bool shadowMapEnabled = true;

	std::vector<Rasterizing::DrawCommand> drawCommands;

	std::vector<uint8_t> postTransformationsActiveMasks;
	const GameSettings* currGs;

	void joinMainWithShadowMap(int threadIndex);

	std::vector<float> zBuffer, shadowMap_zBuffer;
	std::vector<uint32_t> deferredTriangleIndices;

	struct VertexStageInput
	{
		int32x16 triangleIndices;
		std::array<int32x16, 3> vertexIndices;
		Mask16 validInputs;
		float nearPlaneZ;
		//double threadCount;
		//int threadIndex, stage;
		int stage;
		int firstCmd, lastCmd;
	};

	struct VertexStageOutputTriangle
	{
		std::array<Rasterizing::VertexPack16, 3> vertices;
		float32x16 minX, minY, maxX, maxY, rcpSignedArea;
		Mask16 activeTrianges = 0;
	};
	struct VertexStageOutput
	{
		std::array<VertexStageOutputTriangle, 2> outputTriangles;
		int32x16 behindNearPlaneCount; //counts of vertices behind near plane for each triangle composed by input vertices
		std::array<Mask16, 3> behindNearPlaneMasks; //which vertices of the input ones are behind the near plane
	};

	struct TriangleWorkItem
	{
		int cmdIndex = 0;
		int32x16 progenitorTriangleIndices;
		VertexStageOutput vertexOutput;
	};
	struct RasterMailbox
	{
		std::mutex mtx;
		std::vector<std::shared_ptr<TriangleWorkItem>> pending;
	};
	std::vector<RasterMailbox> rasterMailboxes;
	std::atomic<int> transformersDone = 0;

	struct PixelStageInput
	{
		int32x16 progenitorTriangleIndices;
		const Rasterizing::DrawCommand* cmd;
		const VertexStageOutput* vertexStageOutput;
		float my_xMin, my_yMin, my_xMax, my_yMax;
	};

	struct PixelStageOutput
	{
	};
	void performNearPlaneClipping(float clippingZ, std::array<VertexStageOutputTriangle, 2>& input, int32x16 behindPlaneCount, std::array<Mask16, 3> behindPlaneMasks) const;
	//output must point to memory block large enough to contain at least (count of draw commands) VertexStageOutput structs.
	//For draw command i the output will be written to output[i]
	//Stage 1 assumes sequential input triangle indices, gathers only vertices' world coords and processes all draw commands.
	//Stage 2 processes ONLY the draw command at inputted index and gathers it's required attributes for vertices. Doesn't assume any order for input triangle indices.
	void transformVertices(const VertexStageInput& input, VertexStageOutput* output) const;

	void binTrianglesIntoZones(const int threadIndex);

	void drawTriangleBatch(const PixelStageInput& inp, const int threadIndex);

	//Performs transformations and rasterization of binned triangles
	void rasterizerRoutine(const int threadIndex);
	Rasterizing::TextureManager textureManager;
	uint64_t lastTicks = SDL_GetTicksNS(), totalTicks = 0;

	Vec4f lightPos, lightAng, lightColor, skyColor = Vec4f(0.3, 0.7, 1, 1);
	//float lightIntensity, skyLightIntensity;
	float ambientLightIntensity = 0.3, lightIntesity = 3;
	bool drawShadowMapDebug = false, skipTrianglesWithFallbackTexure = true, useShadowMapBias = false, useShadowMapFrontFaceCulling = true;
	Rasterizing::FaceCullingType faceCullingType = Rasterizing::FaceCullingType::NONE;
};