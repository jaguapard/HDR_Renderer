#include <vector>
#include "../RendererBase.h"
#include <map>
#include "CoordinateTransformer.h"
#include <array>
#include "TextureManager.h"
#include "../../helpers.h"

namespace Rasterizing
{
	enum class ShadingMode
	{
		SMOOTH,
		FLAT,
		NONE,
		COUNT,
	};
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


	struct VertexPack16
	{
		Vec4_f32x16 space, normal;
		float32x16 u, v;
		//VertexPack16(float32x16 x, )
		VertexPack16() {};
		static __forceinline VertexPack16 lerpVertices(const VertexPack16& from, const VertexPack16& to, const float32x16& alpha)
		{
			VertexPack16 ret;
			ret.space.x = lerp(from.space.x, to.space.x, alpha);
			ret.space.y = lerp(from.space.y, to.space.y, alpha);
			ret.space.z = lerp(from.space.z, to.space.z, alpha);
			ret.normal.x = lerp(from.normal.x, to.normal.x, alpha);
			ret.normal.y = lerp(from.normal.y, to.normal.y, alpha);
			ret.normal.z = lerp(from.normal.z, to.normal.z, alpha);
			ret.normal /= ret.normal.len3d();
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
			ret.normal.x = _mm512_mask_mov_ps(zero.normal.x, mask, one.normal.x);
			ret.normal.y = _mm512_mask_mov_ps(zero.normal.y, mask, one.normal.y);
			ret.normal.z = _mm512_mask_mov_ps(zero.normal.z, mask, one.normal.z);
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

	struct DrawCommand;

	struct RenderJob
	{
		float x[3], y[3], z[3], u[3], v[3], nx[3], ny[3], nz[3], rcpSignedArea;
		int modelIndex;

		RenderJob() {};
	};
	/*
	struct RenderJobBlock
	{
		std::unique_ptr<RenderJob[]> 
	};*/
	struct RenderJob_Store
	{
		static constexpr size_t MAX_RENDER_JOBS_PER_BLOCK = 600 * 16;
		//using RenderJobBlock = std::unique_ptr<RenderJob[]>;
		
		std::unique_ptr<RenderJob[]> block = std::make_unique<RenderJob[]>(MAX_RENDER_JOBS_PER_BLOCK);
		std::unique_ptr<RenderJob_Store> next = nullptr;
		RenderJob_Store* end = this; //unmanaged pointer for fast insertions. The real owner is it's parent!
		size_t occupiedElementCount = 0;

		RenderJob_Store& getInsertTarget(size_t numElementsToInsert);
		//void getInsertTargetBlockAndSize(size_t numElementsToInsert, std::unique_ptr<RenderJob[]>** outBlock, size_t& outBlockSize);

		//std::array<VertexPack16, 3> loadVertices16(size_t firstInd, Mask16 mask) const;
		//VertexPack16 gatherVertices16(int32x16 indices) const;
		void clear(bool forceClear = false); //sets the size to 0 for this and all children. If forceClear is true, will delete all children and this store's block instead.
		void add(const VertexPack16* pStart, const VertexPack16* pEnd, const float32x16& rcpSignedArea, const int32x16& modelIndex, Mask16 activeElementsMask, const DrawCommand& subInfo);
	};

	enum class FaceCullingType
	{
		NONE,
		BACKFACE,
		FRONTFACE,
		COUNT,
	};

	struct GenericBuffer
	{
		void* data = nullptr;
		int w, h;
		GenericBuffer(void* p, int w, int h) : data(p), w(w), h(h) {};
	};

	enum class DrawRecipe
	{
		MAIN_DEPTH_PREPASS,
		SHADOW_MAP_DEPTH,
	};
	struct DrawCommand
	{
		CoordinateTransformer ctr;
		bool needsUVs = true, needsNormals = true;
		FaceCullingType faceCullingType = FaceCullingType::NONE;
		Rasterizing::ShadingMode shadingMode = Rasterizing::ShadingMode::SMOOTH;

		std::vector<Rasterizing::RenderJob_Store>* transformedVertices = nullptr;
		std::vector<GenericBuffer> buffers;
		uint32_t renderW, renderH;
		uint32_t threadCount = -1;
		DrawRecipe recipe;
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
	std::vector<Rasterizing::RenderJob_Store> mainRenderJobs, shadowMapRenderJobs;

	Rasterizing::TextureManager textureManager;
	uint64_t lastTicks = SDL_GetTicksNS(), totalTicks = 0;

	Vec4f lightPos, lightAng, lightColor, skyColor = Vec4f(0.3, 0.7, 1, 1);
	//float lightIntensity, skyLightIntensity;
	float ambientLightIntensity = 0.3, lightIntesity = 3;
	bool drawShadowMapDebug = false, missingTexturesSetToPlaceholder = true;
	Rasterizing::FaceCullingType faceCullingType = Rasterizing::FaceCullingType::NONE;
};