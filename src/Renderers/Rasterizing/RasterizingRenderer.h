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
	Rasterizing::ShadingMode shadingMode = Rasterizing::ShadingMode::SMOOTH;
	std::vector<Rasterizing::Model> sceneModels;
	Rasterizing::Vertice_Store original_verticeStore;
	Rasterizing::Triangle_Store original_triangleStore;

	std::array<Rasterizing::DrawCommand, 2> drawCommands;

	std::vector<uint8_t> postTransformationsActiveMasks;
	const GameSettings* currGs;

	void doTransformationsAndClipping(int threadIndex);
	void drawRenderJobs(int threadIndex);
	void joinMainWithShadowMap(int threadIndex);

	std::vector<float> zBuffer, shadowMap_zBuffer;
	std::vector<uint64_t> deferrendRenderJobPtrs;
	std::array<std::vector<Rasterizing::TriangleIndexStore>, 2> trianglesByZones;

	//Performs binning of the triangles to rasterizer threads' zones
	void transformerRoutine(int threadIndex);
	void performNearPlaneClipping(float clippingZ, std::array<Rasterizing::VertexPack16, 6>& input, int32x16 behindPlaneCount, std::array<Mask16, 3> behindPlaneMasks) const;

	struct VertexStageInput
	{
		int32x16 triangleIndices;
		std::array<int32x16, 3> vertexIndices;
		Mask16 validInputs;
		float nearPlaneZ;
		//double threadCount;
		//int threadIndex, stage;
		int stage;
		int drawCommandIndex;
	};

	struct VertexStageOutput
	{
		//std::array<Rasterizing::VertexPack16, 3> untransformedVertices; //some fields may be empty due to input flags. World X, Y, Z are guaranteed to be filled 
		std::array<Rasterizing::VertexPack16, 6> output; //up to 6 vertices may be returned by the transform (2 triangles for single vertex behind near plane case). Use activeTriangles mask to mask off invalid ones
		int32x16 behindNearPlaneCount; //counts of vertices behind near plane for each triangle composed by input vertices
		std::array<float32x16, 2> minX, minY, maxX, maxY, rcpSignedArea;
		//int validOutputVertexPackCount;
		std::array<Mask16, 2> activeTriangles; //Masks that mark which triangles are valid post-transform. Values returned in other fields of this struct are garbage for inactive triangles
		std::array<Mask16, 3> behindNearPlaneMasks; //which vertices of the input ones are behind the near plane
	};

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

	//Transforms vertices by data supplied via input for all draw commands.
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
	bool drawShadowMapDebug = false, missingTexturesSetToPlaceholder = true;
	Rasterizing::FaceCullingType faceCullingType = Rasterizing::FaceCullingType::NONE;
};