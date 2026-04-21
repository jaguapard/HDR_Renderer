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

	static inline constexpr int WORKER_JOB_BATCH_SIZE = 2048;
	static inline constexpr int WORKER_PROVISION_SIZE = WORKER_JOB_BATCH_SIZE + 32;
	struct VertexBatch
	{
		std::array<float, WORKER_PROVISION_SIZE> x, y, z, u, v, nx, ny, nz;
	};

	struct TriangleBatchPool;
	struct TriangleBatch;
	class TriangleBatchHandle
	{
	public:
		TriangleBatch* operator&();
		TriangleBatch* operator->();
		TriangleBatch& operator*();
		TriangleBatchHandle& operator=(const TriangleBatchHandle& other);
		TriangleBatchHandle(const TriangleBatchHandle& other);
		~TriangleBatchHandle() noexcept;
		friend struct TriangleBatchPool;
	private:
		TriangleBatchHandle() {};
		TriangleBatchPool* pool;
		int indexInPool;
	};
	struct TriangleBatch
	{
		TriangleBatch() = delete;
		TriangleBatch(const TriangleBatch&) = delete;
		TriangleBatch(const TriangleBatch&&) = delete;
		TriangleBatch& operator=(const TriangleBatch&) = delete;
		TriangleBatch& operator=(const TriangleBatch&&) = delete;
		VertexBatch vertexData[3];		
		//overprovision some space to not worry about bounds
		std::array<float, WORKER_PROVISION_SIZE> minX, minY, maxX, maxY, rcpSignedArea;
		std::array<int, WORKER_PROVISION_SIZE> diffuseMapIndex, triangleIndex;
		float batchMinX, batchMinY, batchMaxX, batchMaxY;
		int batchSize = 0;
		int drawCmdIndex = INT32_MIN;
	};

	struct TriangleBatchPool
	{
		static inline constexpr int MAX_OUTSTANDING_BATCHES = 1024;
		friend struct TriangleBatch;
		TriangleBatchHandle allocate();
		void free(const TriangleBatchHandle& h);
		TriangleBatchPool();
		friend struct TriangleBatchHandle;
	private:
		TriangleBatch* memory;
		std::vector<int> refCount, freeBatchIndices;
		std::recursive_mutex mtx;
	};

	class ThreadBatchList
	{
	public:
		std::mutex mtx;
		std::vector<TriangleBatchHandle> unprocessedBatches;
	};

	void performNearPlaneClipping(float clippingZ, std::array<Rasterizing::VertexPack16, 6>& outVerts, int32x16 behindPlaneCount, std::array<Mask16, 3> behindPlaneMasks) const;

	std::unique_ptr<ThreadBatchList[]> threadBatchLists;
	TriangleBatchPool triangleBatchPool;

	void drawTriangleBatch(const TriangleBatch& batch, const int threadIndex);
	void workerRoutine(const int threadIndex);
	Rasterizing::TextureManager textureManager;
	uint64_t lastTicks = SDL_GetTicksNS(), totalTicks = 0;

	void processBatchesSentByOtherThreads(const int threadIndex);

	Vec4f lightPos, lightAng, lightColor, skyColor = Vec4f(0.3, 0.7, 1, 1);
	//float lightIntensity, skyLightIntensity;
	float ambientLightIntensity = 0.3, lightIntesity = 3;
	bool drawShadowMapDebug = false, skipTrianglesWithFallbackTexure = true, useShadowMapBias = false, useShadowMapFrontFaceCulling = true;
	Rasterizing::FaceCullingType faceCullingType = Rasterizing::FaceCullingType::NONE;
};