#include <vector>
#include "../RendererBase.h"
#include <map>
#include "CoordinateTransformer.h"
#include <array>
#include "TextureManager.h"
#include "../../helpers.h"
#include "RenderJobStore.h"
#include "primitives.h"
#include <memory>

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

	struct TriangleBatch
	{
		static inline constexpr int MAX_BATCH_SIZE = 2048;
		static inline constexpr int BATCH_ALLOCATE_SIZE = MAX_BATCH_SIZE + 32; //to not bother about OOB, overprovision slightly

		uint32_t batchSize = 0;
		std::array<float, BATCH_ALLOCATE_SIZE> maxX, minX, maxY, minY, rcpSignedArea;
		std::array<std::array<float, BATCH_ALLOCATE_SIZE>, 3> scrX, scrY, rcpZ, zDividedU, zDividedV;
		std::array<uint32_t, BATCH_ALLOCATE_SIZE> diffuseMapIndices, drawCmdIndices, triangleIndices;
		
		TriangleBatch() { reset(); }
		void reset()
		{
			this->batchSize = 0;
		}
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
		TriangleBatchHandle& operator=(TriangleBatchHandle&& other) noexcept;
		TriangleBatchHandle(const TriangleBatchHandle& other);
		TriangleBatchHandle(TriangleBatchHandle&& other) noexcept;
		~TriangleBatchHandle() noexcept;
		friend struct TriangleBatchPool;
	private:
		TriangleBatchHandle() {};
		void decrementRefCnt() const;
		void incrementRefCnt() const;
		mutable TriangleBatchPool* pool = nullptr;
		int indexInPool;
	};

	struct TriangleBatchPool
	{
		static inline constexpr int MAX_OUTSTANDING_BATCHES = 8192;
		friend struct TriangleBatch;
		TriangleBatchHandle allocate();
		void free(const TriangleBatchHandle& h);
		TriangleBatchPool();
		friend struct TriangleBatchHandle;
	private:
		TriangleBatch* memory;
		std::array<std::atomic_int, MAX_OUTSTANDING_BATCHES> refCount;
		std::array<int, MAX_OUTSTANDING_BATCHES> freeBatchIndices;
		int topIndex;
		std::recursive_mutex mtx;
	};
	struct ThreadMailbox
	{
		std::mutex mtx;
		std::vector<TriangleBatchHandle> pendingBatches;
	};
}
class RasterizingRenderer : public RendererBase
{
public:
	virtual void loadScene(RendererLoadSceneData scd);
	virtual void renderFrame(const GameSettings& settings);
	//virtual void handleInputEvent(const SDL_Event& ev, C_Input& input);
private:
	Rasterizing::TriangleBatchPool triangleBatchPool;
	void loadScene(bool debugScene);
	void clearScene();
	void processMailbox(uint32_t threadIndex);
	bool singleTriangleDebugMode = false;
	Rasterizing::ShadingMode shadingMode = Rasterizing::ShadingMode::SMOOTH;
	std::vector<Rasterizing::Model> sceneModels;
	Rasterizing::VertexStore vertexStore;
	Rasterizing::TriangleStore triangleStore;
	bool shadowMapEnabled = true;

	void workerRoutine(const uint32_t threadIndex);

	std::vector<Rasterizing::DrawCommand> drawCommands;

	std::vector<uint8_t> postTransformationsActiveMasks;
	const GameSettings* currGs;

	void joinMainWithShadowMap(int threadIndex);

	void sendBatchToThreadAndSwapToNew(Rasterizing::TriangleBatchHandle& batch, int receiver);

	std::vector<float> zBuffer, shadowMap_zBuffer;
	std::vector<uint32_t> deferredTriangleIndices;
	std::array<std::vector<Rasterizing::TriangleIndexStore>, 2> trianglesByZones;
	uint32_t threadCount;
	
	std::unique_ptr<Rasterizing::ThreadMailbox[]> threadMailboxes;
	size_t prevThreadCount = -1;
	std::atomic<uint64_t> claimedTriangles = 0;


	struct TrianglePack16
	{
		std::array<Rasterizing::VertexPack16, 3> vertices;
		float32x16 minX, minY, maxX, maxY, rcpSignedArea;
		int32x16 diffuseMapIndices, drawCmdIndices, triangleIndices;
		Mask16 activeTrianges = 0;
	};
	void performNearPlaneClipping(float clippingZ, std::array<Rasterizing::VertexPack16, 6>& toClip, int32x16 behindPlaneCount, std::array<Mask16, 3> behindPlaneMasks) const;

	//void drawTriangleBatch(const PixelStageInput& inp, const int threadIndex);
	void drawTrianglePack(const TrianglePack16& pack, const uint32_t threadIndex);

	Rasterizing::TextureManager textureManager;
	uint64_t lastTicks = SDL_GetTicksNS(), totalTicks = 0;

	Vec4f lightPos, lightAng, lightColor, skyColor = Vec4f(0.3, 0.7, 1, 1);
	//float lightIntensity, skyLightIntensity;
	float ambientLightIntensity = 0.3, lightIntesity = 3;
	bool drawShadowMapDebug = false, skipTrianglesWithFallbackTexure = true, useShadowMapBias = false, useShadowMapFrontFaceCulling = true;
	Rasterizing::FaceCullingType faceCullingType = Rasterizing::FaceCullingType::NONE;
};