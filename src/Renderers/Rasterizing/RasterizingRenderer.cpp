#include "RasterizingRenderer.h"
#include "../../AssetLoader.h"
#include <stdexcept>
#include <unordered_map>
#include "../../Vec.h"
#include "../../GameSettings.h"
#include "../../Threadpool.h"
#include <map>
#include <iostream>
#include "../../Statsman.h"
#include "../../helpers.h"
#include "../../C_Input.h"
#include "../../EnumCycler.h"
#include "RenderJobStore.h"
using namespace Rasterizing;

std::vector<SequentialRange> intsToMergedRanges(std::vector<int> ints)
{
	if (ints.empty()) return {};
	if (ints.size() == 1) return { {ints[0], ints[0]} };

	std::sort(ints.begin(), ints.end());

	std::vector<SequentialRange> ret;
	int i = 0;
	do {
		SequentialRange currRange;
		currRange.min = currRange.max = ints[i++];
		//since numbers are already sorted in non-descending order, we can safely append next to current range if it doesn't jump abruptly
		while (i < ints.size() && ints[i] - currRange.max <= 1) 
		{
			currRange.max = ints[i];
			++i;
		}
		ret.push_back(currRange);
	} while (i < ints.size());
	return ret;
}
void RasterizingRenderer::loadScene(RendererLoadSceneData scd)
{
	Uint64 ticksBegin = SDL_GetTicksNS();
	std::mutex mtx;
	std::vector<task_id> tasks;
	std::string savePaths[] = {"new_sponza.bmdl2", "curtains.bmdl2", "tree.bmdl2", "ivy.bmdl2"};
	auto currSavePath = std::begin(savePaths);
	bool onlyConvertToBmdl = false;
	for (auto& [path, mode] : scd.files)
	{
		AssetLoader ldr;
		std::vector<AssetLoader::ImportedModel> loadedModels;
		if (mode == "obj" || mode == "") { loadedModels = ldr.loadObj(path, onlyConvertToBmdl ? "H:/Sponza goodies/BMDL/" + *currSavePath : ""); }
		else if (mode == "bmdl") { loadedModels = ldr.loadBmdl(path); }
		else throw std::runtime_error("Unsupported mode for RasterizingRenderer::loadScene: " + mode);
		if (onlyConvertToBmdl)
		{
			++currSavePath;
			continue;
		}

		size_t importModelCount = loadedModels.size();
		std::vector<int> diffuseMapIndices(importModelCount, -1);
		std::vector<task_id> textureLoadingTasks(importModelCount);

		for (int i = 0; i < importModelCount; ++i)
		{
			textureLoadingTasks[i] = Threadpool::instance->addTask([&, this, i]() {
				if (loadedModels[i].diffuseMapPath) diffuseMapIndices[i] = this->textureManager.addTextureByPath(*loadedModels[i].diffuseMapPath);
			});
		}

		size_t loadedTriangles = 0;
		size_t firstModelInd = this->sceneModels.size();
		for (int i = 0; i < loadedModels.size(); ++i)
		{
			size_t discardedTriangles = 0;
			Model& m = this->sceneModels.emplace_back();
			m.globalTriangleRange.min = this->triangleStore.vertInd[0].size(); //since textures are not yet loaded, diffuseMapIndex is not filled, and triangle store size() will be wrong!
			for (auto& it : loadedModels[i].triangles)
			{
				Vec4f v1 = { it.v[0].space.x, it.v[0].space.y, it.v[0].space.z, 0 };
				Vec4f v2 = { it.v[1].space.x, it.v[1].space.y, it.v[1].space.z, 0 };
				Vec4f v3 = { it.v[2].space.x, it.v[2].space.y, it.v[2].space.z, 0 };
				Vec4f dv21 = v2 - v1;
				Vec4f dv32 = v3 - v2;
				if (dv21.lenSq() * dv32.lenSq() == 0) //degenerate triangles
				{
					++discardedTriangles; continue;
				}
				for (int k = 0; k < 3; ++k)
				{
					uint32_t vertInd = this->vertexStore.insert(
						it.v[k].space.x, it.v[k].space.y, it.v[k].space.z, it.v[k].diffuseMapCoords.x, 1 - it.v[k].diffuseMapCoords.y, it.v[k].normal.x, it.v[k].normal.y, it.v[k].normal.z
					);

					this->triangleStore.vertInd[k].push_back(vertInd);
					//TODO: clean from degenerate triangles (i.e 2 or 3 vertices same or all 3 collinear)?
				}
			}
			m.globalTriangleRange.max = this->triangleStore.vertInd[0].size() - 1;
			//std::cout << "Loaded " << m.triangleStore.size() << " triangles out of " << loadedModels[i].triangles.size() << " (" << discardedTriangles << " discarded) from " << path << "\n";
		}

		Threadpool::instance->waitForMultipleTasks(textureLoadingTasks);
		size_t lastModelInd = this->sceneModels.size() - 1;
		for (int i = 0; i < loadedModels.size(); ++i)
		{
			auto& currModel = this->sceneModels[firstModelInd + i];
			//triangleStore.diffuseMapIndex.resize()
			bool noBackfaceCulling = !this->textureManager.getTextureByHandle(diffuseMapIndices[i]).areAllPixelsOpaque();
			ModelFlags flags = noBackfaceCulling ? NO_BACKFACE_CULLING : NONE;
			for (int j = currModel.globalTriangleRange.min; j <= currModel.globalTriangleRange.max; ++j)
			{
				this->triangleStore.diffuseMapIndex.push_back(diffuseMapIndices[i]);
				this->triangleStore.modelFlags.push_back(flags);
			}
		}
	}

	Threadpool::instance->waitForMultipleTasks(tasks);
	Uint64 ticksEnd = SDL_GetTicksNS();
	std::cout << "Scene loaded in " << (ticksEnd - ticksBegin) / 1e9 << " sec.\n";

	if (onlyConvertToBmdl) throw std::runtime_error("BMDL conversion complete. This is not an error, but models are removed from memory immediately after converting to BMDL. Disable conversion and load BMDL directly on next launch.");
}

void RasterizingRenderer::renderFrame(const GameSettings& settings)
{
	this->currGs = &settings;
	if (this->singleTriangleDebugMode)
	{
		const_cast<GameSettings*>(this->currGs)->camPos = { 13.475824, -1.453772, 69.824371, 0.000000 };
		const_cast<GameSettings*>(this->currGs)->camAng = { 0.000000, -2.604962, 0.182000, 0.000000 };
		this->singleTriangleDebugMode = false;
	}
	int mainBufSize = settings.outputTextureParams.Width * settings.outputTextureParams.Height;
	this->zBuffer.resize(mainBufSize);
#ifdef NDEBUG
	int shadowMapW = 512*3;
	int shadowMapH = 288*3;
#else
	int shadowMapW = 51;
	int shadowMapH = 28;
#endif
	this->deferredTriangleIndices.resize(mainBufSize);
	
	C_Input& inp = C_Input::getInstance();
	if (inp.wasCharPressedOnThisFrame('N')) this->shadingMode = EnumCycler::next(this->shadingMode);
	if (inp.wasCharPressedOnThisFrame('M')) this->drawShadowMapDebug ^= 1;
	if (inp.wasCharPressedOnThisFrame('B')) this->faceCullingType = EnumCycler::next(this->faceCullingType);
	if (inp.wasButtonPressedOnThisFrame(SDL_SCANCODE_KP_6)) this->shadowMapEnabled ^= 1;
	if (inp.wasButtonPressedOnThisFrame(SDL_SCANCODE_KP_7)) this->useShadowMapBias ^= 1;
	if (inp.wasButtonPressedOnThisFrame(SDL_SCANCODE_KP_8)) this->useShadowMapFrontFaceCulling ^= 1;
	if (inp.wasButtonPressedOnThisFrame(SDL_SCANCODE_KP_9)) this->skipTrianglesWithFallbackTexure ^= 1;
	if (inp.wasButtonPressedOnThisFrame(SDL_SCANCODE_KP_0))
	{
		this->clearScene();
		this->loadScene(true);
		return;
	}


	Threadpool* threadpool = settings.threadpool;
	int threadCount = threadpool->getThreadCount();

	int w = (int)settings.outputTextureParams.Width;
	int h = (int)settings.outputTextureParams.Height;
	this->drawCommands.clear();
	DrawCommand& mainDrawCmd = this->drawCommands.emplace_back();
	mainDrawCmd.ctr = { w,h }; 
	mainDrawCmd.ctr.prepare(settings.camPos, settings.camAng);
	mainDrawCmd.shadingMode = this->shadingMode;
	mainDrawCmd.buffers.emplace_back(this->zBuffer.data(), w, h);
	mainDrawCmd.buffers.emplace_back(this->currGs->graphicsOutputBuffer, w, h);
	mainDrawCmd.buffers.emplace_back(this->deferredTriangleIndices.data(), w, h);
	mainDrawCmd.needsUVs = true;
	mainDrawCmd.needsNormals = false;
	mainDrawCmd.faceCullingType = this->faceCullingType;
	mainDrawCmd.trianglesToZones = &this->trianglesByZones[0];
	mainDrawCmd.threadCount = threadCount;
	mainDrawCmd.renderW = w;
	mainDrawCmd.renderH = h;
	mainDrawCmd.recipe = DrawRecipe::MAIN_DEPTH_PREPASS;
	mainDrawCmd.zoneManager = BufferZoneManager(threadCount, w, h);

	if (this->shadowMapEnabled)
	{
		this->shadowMap_zBuffer.resize(shadowMapW * shadowMapH);
		DrawCommand& shadowMapDrawCmd = this->drawCommands.emplace_back();
		shadowMapDrawCmd.ctr = { (int)shadowMapW, (int)shadowMapH };
		shadowMapDrawCmd.ctr.prepare(Vec4f(1281.845703, 2235.967773, 178.236572, 0.000000), Vec4f(0.000000, 4.523108, 0.797002, 0.000000));
		//shadowMapDrawCmd.ctr.prepare(Vec4f(-86.050537, 1644.088623, 710.859253, 0.000000), Vec4f(0.000000, -3.165947,0.366014,0.000000));
		//shadowMapDrawCmd.ctr.prepare(Vec4f(44.960358, 2656.120605,-223.813354, 0.000000), Vec4f(0.000000,1.054968,0.813000,0.000000));
		//shadowMapDrawCmd.ctr.prepare(Vec4f(44.960358, 2656.120605,-223.813354, 0.000000), Vec4f(0.000000,1.054968,0.813000,0.000000));
		//shadowMapDrawCmd.ctr.prepare(settings.camPos, settings.camAng);
		shadowMapDrawCmd.trianglesToZones = &this->trianglesByZones[1];
		shadowMapDrawCmd.renderW = shadowMapW; //TODO: transformer has W and H already, infer it?
		shadowMapDrawCmd.renderH = shadowMapH;
		shadowMapDrawCmd.buffers.emplace_back(this->shadowMap_zBuffer.data(), shadowMapW, shadowMapH);
		shadowMapDrawCmd.needsUVs = true;
		shadowMapDrawCmd.needsNormals = false;
		shadowMapDrawCmd.faceCullingType = this->useShadowMapFrontFaceCulling ? FaceCullingType::FRONTFACE : FaceCullingType::NONE; //FaceCullingType::FRONT
		shadowMapDrawCmd.threadCount = threadCount;
		shadowMapDrawCmd.renderW = shadowMapW;
		shadowMapDrawCmd.renderH = shadowMapH;
		shadowMapDrawCmd.recipe = DrawRecipe::SHADOW_MAP_DEPTH;
		shadowMapDrawCmd.zoneManager = BufferZoneManager(threadCount, shadowMapW, shadowMapH);
	}

	int tCntSq = threadCount * threadCount;
	for (auto& currSub : this->drawCommands)
	{
		if (currSub.trianglesToZones->size() != tCntSq) currSub.trianglesToZones->resize(tCntSq);
		//for (auto& it : *currSub.trianglesToZones) it.verticeStore = &this->vertexStore;
	}

	std::vector<task_id> transformTasks, drawTasks;
	for (int threadIndex = 0; threadIndex < threadCount; ++threadIndex)
	{
		transformTasks.emplace_back(threadpool->addTask(
			[&, threadIndex]() {
				this->binTrianglesIntoZones(threadIndex);
			}
		));
	}

	uint64_t bufCleanTicksBegin = SDL_GetTicksNS();
	for (auto& it : zBuffer) it = -INFINITY;
	if (this->shadowMapEnabled)
	{
		for (auto& it : shadowMap_zBuffer) it = -INFINITY;
	}
	for (auto& it : this->deferredTriangleIndices) it = -1;
	uint64_t zBufCleanTicks = SDL_GetTicksNS();	
	
	int sz = settings.outputTextureParams.Width * settings.outputTextureParams.Height;
	uint64_t skyColor = _mm_extract_epi64(_mm_cvtps_ph(this->skyColor, _MM_FROUND_NO_EXC), 0);
	uint64_t* pp = (uint64_t*)(settings.graphicsOutputBuffer);
	for (int i = 0; i < sz; ++i) pp[i] = skyColor;
	uint64_t framebufCleanTicks = SDL_GetTicksNS();

	Statsman::statsmenForThreads.back().time.zBufferCleanMs = (zBufCleanTicks - bufCleanTicksBegin) / 1e6;
	Statsman::statsmenForThreads.back().time.frameBufferCleanMs = (framebufCleanTicks - zBufCleanTicks) / 1e6;

	threadpool->waitForMultipleTasks(transformTasks);

	size_t renderJobCount = 0;
	//for (auto& it : this->renderJobsFromThreads) renderJobCount += it.size();
	//std::cout << renderJobCount << " render jobs\n";
	for (int threadIndex = 0; threadIndex < threadCount; ++threadIndex)
	{
		drawTasks.emplace_back(threadpool->addTask(
			[&, threadIndex]() {
				this->rasterizerRoutine(threadIndex);
			}
		));
	}
	threadpool->waitForMultipleTasks(drawTasks);
	drawTasks.clear();
	
	for (int threadIndex = 0; threadIndex < threadCount; ++threadIndex)
	{
		drawTasks.emplace_back(threadpool->addTask(
			[&, threadIndex]() {
				this->joinMainWithShadowMap(threadIndex);
			}
		));
	}
	threadpool->waitForMultipleTasks(drawTasks);
	
	for (auto& currSub : this->drawCommands)
	{
		for (auto& store : *currSub.trianglesToZones)
			store.reset();
	}
	//for (auto& it : renderJobsFromThreads) it.clear();
}

void RasterizingRenderer::loadScene(bool debugScene)
{
	this->ambientLightIntensity = this->lightIntesity = 1;
	this->skyColor = { 0.0,0.014,0,1 };
	Model& m = this->sceneModels.emplace_back();
	Vec4f vertices[3] = {
		{-50, 0, 50},
		{50, 0, 50},
		{50, 20, 50},
	};
	Vec4f uvs[3] = {
		{0,0},
		{3,0},
		{3,3},
	};
	for (int k = 0; k < 3; ++k)
	{
		uint32_t vertInd = this->vertexStore.insert(vertices[k].x, -vertices[k].y, vertices[k].z, uvs[k].x, uvs[k].y, 1, 1, 1);
		this->triangleStore.vertInd[k].push_back(vertInd);
	}
	this->triangleStore.diffuseMapIndex.push_back(0);
	this->triangleStore.modelFlags.push_back(NONE);
	//GS are not yet set, so this ptr is null at this time
	this->singleTriangleDebugMode = true;
}

void RasterizingRenderer::clearScene()
{
	this->sceneModels.clear();
	this->triangleStore.clear();
	this->vertexStore.clear();
}

struct Vertex
{
	float x, y, z, u, v;
	Vertex() {};
	Vertex(const VertexPack16& pack, int i) { x = pack.space.x[i]; y = pack.space.y[i]; z = pack.space.z[i], u = pack.u[i], v = pack.v[i]; }
	static Vertex lerpVertices(const Vertex& from, const Vertex& to, float alpha) {
		Vertex ret;
		ret.x = lerp(from.x, to.x, alpha);
		ret.y = lerp(from.y, to.y, alpha);
		ret.z = lerp(from.z, to.z, alpha);
		ret.u = lerp(from.u, to.u, alpha);
		ret.v = lerp(from.v, to.v, alpha);
		return ret;
	}
	static Vertex lerpToPlane(const Vertex& from, const Vertex& to, float clippingZ)
	{
		float alpha = (clippingZ - from.z) / (to.z - from.z);
		assert(alpha >= 0 && alpha <= 1);
		return Vertex::lerpVertices(from, to, alpha);
	}
	const Vertex& writeToPack(VertexPack16& pack, int i) const
	{
		pack.space.x[i] = x;
		pack.space.y[i] = y;
		pack.space.z[i] = z;
		pack.u[i] = u;
		pack.v[i] = v;
		return *this;
	}
};

void RasterizingRenderer::performNearPlaneClipping(float clippingZ, std::array<VertexStageOutputTriangle, 2>& inputTriangles, int32x16 behindPlaneCount, std::array<Mask16, 3> behindPlaneMasks) const
{
	if (!(behindPlaneCount > 0)) return;
	//if behind plane count == 0: block is skipped, triangle is fully ahead of clipping plane
	//if behind plane count == 3: triangle discarded completely
	//if behind plane count == 1: this triangle becomes 2 new triangles (required creating a new one, second can be adjusted in-place)
	//if behind plane count == 2: this triangle becomes 1 new triangle (can be adjusted in-place)
	//If clipping is needed, there is at least 1 vertice in front and 1 vertice behind the plane

	//only these 6 tuples are possible if we got into this branch
	//	is vertex behind plane?	behindPlaneCount
	//	vertex0	vertex1	vertex2	one	two
	//	0		0		1		1	0		
	//	0		1		0		1	0
	//	1		0		0		1	0
	// 
	//	0		1		1		0	1
	//	1		0		1		0	1
	//	1		1		0		0	1
	/* TODO: can use this values instead of recalculating in the loop
	float32x16 z0 = transformedVertices[0].space.z;
	float32x16 z1 = transformedVertices[1].space.z;
	float32x16 z2 = transformedVertices[2].space.z;
	float32x16 alpha1 = (clippingZ - z0) / (z1 - z0);
	float32x16 alpha2 = (clippingZ - z0) / (z2 - z0);
	float32x16 alpha3 = (clippingZ - z1) / (z2 - z1);
	VertexPack16 lerp01 = VertexPack16::lerpVertices(transformedVertices[0], transformedVertices[1], alpha1);
	VertexPack16 lerp02 = VertexPack16::lerpVertices(transformedVertices[0], transformedVertices[2], alpha2);
	VertexPack16 lerp12 = VertexPack16::lerpVertices(transformedVertices[1], transformedVertices[2], alpha3);*/

	//TODO: this network can be adjusted to preserve input windings, it currently doesn't but we don't care for now (face culling is done before, so it seem to not matter at all?)

	std::array<VertexPack16, 6> clipOutput;
	for (int i = 0; i < 3; ++i) clipOutput[i] = inputTriangles[0].vertices[i];
	for (int frontVertex = 0; frontVertex < 3; ++frontVertex)
	{
		//these are behind
		int prevVertex = frontVertex == 0 ? 2 : frontVertex - 1;
		int nextVertex = frontVertex == 2 ? 0 : frontVertex + 1;
		Mask16 caseMask = ~behindPlaneMasks[frontVertex] & behindPlaneMasks[prevVertex] & behindPlaneMasks[nextVertex];
		clipOutput[0] = VertexPack16::maskMove(clipOutput[0], VertexPack16::lerpToClippingZ(inputTriangles[0].vertices[prevVertex], inputTriangles[0].vertices[frontVertex], clippingZ), caseMask);
		clipOutput[1] = VertexPack16::maskMove(clipOutput[1], inputTriangles[0].vertices[frontVertex], caseMask);
		clipOutput[2] = VertexPack16::maskMove(clipOutput[2], VertexPack16::lerpToClippingZ(inputTriangles[0].vertices[frontVertex], inputTriangles[0].vertices[nextVertex], clippingZ), caseMask);
	}

	for (int behindVertex = 0; behindVertex < 3; ++behindVertex)
	{
		//these are ahead
		int prevVertex = behindVertex == 0 ? 2 : behindVertex - 1;
		int nextVertex = behindVertex == 2 ? 0 : behindVertex + 1;
		Mask16 caseMask = behindPlaneMasks[behindVertex] & ~behindPlaneMasks[prevVertex] & ~behindPlaneMasks[nextVertex];
		clipOutput[0] = VertexPack16::maskMove(clipOutput[0], inputTriangles[0].vertices[nextVertex], caseMask);
		clipOutput[1] = VertexPack16::maskMove(clipOutput[1], inputTriangles[0].vertices[prevVertex], caseMask);
		clipOutput[2] = VertexPack16::maskMove(clipOutput[2], VertexPack16::lerpToClippingZ(inputTriangles[0].vertices[nextVertex], inputTriangles[0].vertices[behindVertex], clippingZ), caseMask);
		clipOutput[3] = VertexPack16::maskMove(clipOutput[3], inputTriangles[0].vertices[prevVertex], caseMask);
		clipOutput[4] = VertexPack16::maskMove(clipOutput[4], VertexPack16::lerpToClippingZ(inputTriangles[0].vertices[prevVertex], inputTriangles[0].vertices[behindVertex], clippingZ), caseMask);
		clipOutput[5] = VertexPack16::maskMove(clipOutput[5], VertexPack16::lerpToClippingZ(inputTriangles[0].vertices[nextVertex], inputTriangles[0].vertices[behindVertex], clippingZ), caseMask);
	}

	for (int i = 0; i < 6; ++i)
	{
		inputTriangles[i / 3].vertices[i % 3] = clipOutput[i];
	}
}


void RasterizingRenderer::transformVertices(const VertexStageInput& input, VertexStageOutput* output) const
{
	//assert(std::fmod(input.threadCount, 1) == 0);
	std::array<VertexPack16, 3> originalVertices;
	for (int i = 0; i < 3; ++i)
	{
		this->vertexStore.gatherWorldXYZ(input.vertexIndices[i], input.validInputs, originalVertices[i].space);
		originalVertices[i].space.w = 1;
	}
	bool UVs_loaded = false, normals_loaded = false;

	for (int cmdIndex = input.firstCmd; cmdIndex <= input.lastCmd; ++cmdIndex)
	{
		auto& currCmd = this->drawCommands[cmdIndex];
		auto& currOutput = output[cmdIndex];
		for (auto& it : currOutput.outputTriangles) it.activeTrianges = 0;
		auto* currOutputTriangle = &currOutput.outputTriangles[0];
		int32x16 behindNearPlaneCount = 0;
		for (int i = 0; i < 3; ++i)
		{
			Vec4_f32x16 rotatedTranslated = currCmd.ctr.rotateAndTranslate(originalVertices[i].space);
			Mask16 vertexBehindClippingPlane = rotatedTranslated.z < input.nearPlaneZ;
			currOutput.behindNearPlaneMasks[i] = rotatedTranslated.z < input.nearPlaneZ;
			behindNearPlaneCount = _mm512_mask_add_epi32(behindNearPlaneCount, vertexBehindClippingPlane, behindNearPlaneCount, int32x16(1));
			currOutputTriangle->vertices[i].space = rotatedTranslated;
		}

		Mask16 activeTriangles = input.validInputs & (behindNearPlaneCount != 3);
		if (!activeTriangles) continue;

		if (currCmd.faceCullingType != FaceCullingType::NONE)
		{
			int32x16 modelFlags;
			const auto* flagsPtr = this->triangleStore.modelFlags.data();
			if (input.stage == 1) modelFlags = _mm512_maskz_loadu_epi32(activeTriangles, flagsPtr + input.triangleIndices[0]);
			else modelFlags = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(0), activeTriangles, input.triangleIndices, flagsPtr, 4);
			
			Vec4_f32x16 transformedFaceNormals = getFaceNormalsForTriangles16(currOutputTriangle->vertices[0].space, currOutputTriangle->vertices[1].space, currOutputTriangle->vertices[2].space);
			float32x16 dot = currOutputTriangle->vertices[0].space.dot3d(transformedFaceNormals);
			switch (currCmd.faceCullingType)
			{
				case FaceCullingType::BACKFACE: activeTriangles &= (modelFlags & NO_BACKFACE_CULLING) != 0 | dot < 0.f; break;
				case FaceCullingType::FRONTFACE: activeTriangles &= (modelFlags & NO_FRONTFACE_CULLING) != 0 | dot >= 0.f; break;
				default: break;
			}
			if (!activeTriangles) continue;
		}
		
		if (input.stage != 1)
		{
			if (currCmd.needsUVs)
			{
				if (!UVs_loaded)
				{
					for (int i = 0; i < 3; ++i)
					{
						//activeTriangles may be different between commands, so gather by least restrictive valid mask, which is the input valid mask
						this->vertexStore.gatherUV(input.vertexIndices[i], input.validInputs, originalVertices[i].u, originalVertices[i].v);
					}
					UVs_loaded = true;
				}

				for (int i = 0; i < 3; ++i)
				{
					currOutputTriangle->vertices[i].u = originalVertices[i].u;
					currOutputTriangle->vertices[i].v = originalVertices[i].v;
				}
			}

			if (currCmd.needsNormals)
			{
				if (!normals_loaded)
				{
					for (int i = 0; i < 3; ++i)
					{
						this->vertexStore.gatherNormals(input.vertexIndices[i], input.validInputs, originalVertices[i].normal);
					}
					normals_loaded = true;
				}
				for (int i = 0; i < 3; ++i) currOutputTriangle->vertices[i].normal = originalVertices[i].normal;
			}
		}
		
		this->performNearPlaneClipping(input.nearPlaneZ, currOutput.outputTriangles, behindNearPlaneCount, currOutput.behindNearPlaneMasks);

		float w = currCmd.renderW;
		float h = currCmd.renderH;
		currOutput.behindNearPlaneCount = behindNearPlaneCount;

		Mask16 oldActiveTriangles = activeTriangles;
		for (int outputTriangleIndex = 0; outputTriangleIndex < 2; outputTriangleIndex++)
		{
			if (outputTriangleIndex == 1)
			{
				if (behindNearPlaneCount == 1) //load new triangle if there is new
				{
					activeTriangles = oldActiveTriangles & (behindNearPlaneCount == 1);
				}
				else break;
			}
			if (!activeTriangles) break;
			currOutputTriangle = &currOutput.outputTriangles[outputTriangleIndex];
			
			float32x16 fovMult = 1; //TODO: adjustable game setting?
			float32x16 minX = INFINITY, maxX = -INFINITY, minY = INFINITY, maxY = -INFINITY;
			for (int i = 0; i < 3; ++i)
			{
				auto& currVertex = currOutputTriangle->vertices[i];
				float32x16 zInv = fovMult / currVertex.space.z;
				currVertex.space = currCmd.ctr.screenSpaceToPixels(currVertex.space * zInv);
				minX = _mm512_min_ps(minX, currVertex.space.x);
				minY = _mm512_min_ps(minY, currVertex.space.y);
				maxX = _mm512_max_ps(maxX, currVertex.space.x);
				maxY = _mm512_max_ps(maxY, currVertex.space.y);
				if (input.stage == 2)
				{
					currVertex.space.z = zInv;
					currVertex.u *= zInv;
					currVertex.v *= zInv;
					currVertex.normal.x *= zInv;
					currVertex.normal.y *= zInv;
					currVertex.normal.z *= zInv;
				}
			}

			const Vec4_f32x16& r1 = currOutputTriangle->vertices[0].space;
			const Vec4_f32x16& r2 = currOutputTriangle->vertices[1].space;
			const Vec4_f32x16& r3 = currOutputTriangle->vertices[2].space;

			float32x16 signedArea = (r1 - r3).cross2d(r2 - r3);
			Mask16 nonZeroSignedAreaMask = signedArea != 0.f;
			activeTriangles = (minX < w & maxX >= 0.f) & (minY < h & maxY >= 0.f) & (activeTriangles & nonZeroSignedAreaMask);
			if (!activeTriangles) continue;

			currOutputTriangle->minX = _mm512_floor_ps(minX);
			currOutputTriangle->minY = _mm512_floor_ps(minY);
			currOutputTriangle->maxX = _mm512_ceil_ps(maxX);
			currOutputTriangle->maxY = _mm512_ceil_ps(maxY);
			currOutputTriangle->rcpSignedArea = float32x16(1) / signedArea;
			currOutputTriangle->activeTrianges = activeTriangles;
		}
	}
}

void RasterizingRenderer::binTrianglesIntoZones(int threadIndex)
{
	auto [d_low, d_high] = Threadpool::instance->getLimitsForThread(threadIndex, 0, this->triangleStore.size());
	size_t startInd = d_low, stopInd = d_high;
	int threadCount = this->currGs->threadpool->getThreadCount();

	VertexStageInput inp;
	inp.nearPlaneZ = this->currGs->cameraPlane_zDist;
	inp.stage = 1;
	inp.firstCmd = 0;
	inp.lastCmd = this->drawCommands.size() - 1;
	auto transformedResults = std::make_unique<VertexStageOutput[]>(this->drawCommands.size()); //this is called only once per frame per thread anyway, so no need to torture yourself with static arrays and checks
	for (size_t currTriangleIndex = startInd; currTriangleIndex < stopInd; currTriangleIndex += 16)
	{
		int32x16 triangleIndices = int32x16::sequence() + currTriangleIndex;
		Mask16 groupActiveTriangles = triangleIndices < stopInd;
		if (this->skipTrianglesWithFallbackTexure)
		{
			int32x16 diffuseMapIndices = _mm512_maskz_loadu_epi32(groupActiveTriangles, this->triangleStore.diffuseMapIndex.data() + currTriangleIndex);
			groupActiveTriangles &= diffuseMapIndices != 0;
			if (!groupActiveTriangles) continue;
		}

		inp.triangleIndices = triangleIndices;
		inp.validInputs = groupActiveTriangles;
		for (int i = 0; i < 3; ++i)
		{
			inp.vertexIndices[i] = _mm512_maskz_loadu_epi32(groupActiveTriangles, this->triangleStore.vertInd[i].data() + currTriangleIndex);
		}

		this->transformVertices(inp, transformedResults.get());
		
		for (int cmdIndex = 0; cmdIndex < this->drawCommands.size(); ++cmdIndex)
		{
			for (int outputTriangleIndex = 0; outputTriangleIndex < 2; ++outputTriangleIndex)
			{
				const auto& currTriangles = transformedResults[cmdIndex].outputTriangles[outputTriangleIndex];
				Mask16 currActiveTriangles = currTriangles.activeTrianges;
				if (!currActiveTriangles) break; //yes, break, not continue. If first outputted triangle is invalid, then none are (at least in current pipeline)
				auto& currCmd = this->drawCommands[cmdIndex];
				float rcpScreenHeightPerThread = double(threadCount) / currCmd.renderH;
				auto& currOutput = transformedResults[cmdIndex];

				int32x16 vecFirstThread = _mm512_cvttps_epi32(currTriangles.minY * rcpScreenHeightPerThread);
				int32x16 vecLastThread = _mm512_cvttps_epi32(currTriangles.maxY * rcpScreenHeightPerThread);
				currActiveTriangles &= (vecLastThread >= 0) & (vecFirstThread < threadCount);
				if (!currActiveTriangles) continue;

				vecFirstThread = vecFirstThread.clamp(0, threadCount - 1);
				vecLastThread = vecLastThread.clamp(0, threadCount - 1);

				for (int i = 0; i < 16; ++i)
				{
					if ((currActiveTriangles.mask & (1 << i)) == 0) continue;
					for (int currReceiverThread = vecFirstThread[i]; currReceiverThread <= vecLastThread[i]; ++currReceiverThread)
					{
						auto& targetStore = (*currCmd.trianglesToZones)[threadIndex * threadCount + currReceiverThread];
						targetStore.append(currTriangleIndex + i);
					}
				}
			}
		}
	}
}

__forceinline void calculateBarycentricCoordinates2D(const Vec4_f32x16& r, const Vec4_f32x16& r1, const Vec4_f32x16& r2, const Vec4_f32x16& r3, const float32x16& rcpSignedArea, float32x16& alpha, float32x16& beta, float32x16& gamma)
{
	alpha = (r - r3).cross2d(r2 - r3) * rcpSignedArea;
	beta = (r - r3).cross2d(r3 - r1) * rcpSignedArea;
	gamma = (r - r1).cross2d(r1 - r2) * rcpSignedArea; //do NOT change this to 1-alpha-beta or 1-(alpha+beta). That causes wonkiness in textures
}

__forceinline void calculateBarycentricCoordinates3D(const Vec4_f32x16& P, const Vec4_f32x16& A, const Vec4_f32x16& B, const Vec4_f32x16& C, float32x16& alpha, float32x16& beta, float32x16& gamma)
{
	/* //this version is less precise, causes texture issues in some places
	Vec4_f32x16 v0 = B - A;
	Vec4_f32x16 v1 = C - A;
	Vec4_f32x16 v2 = P - A;

	float32x16 d00 = v0.dot3d(v0);
	float32x16 d01 = v0.dot3d(v1);
	float32x16 d11 = v1.dot3d(v1);
	float32x16 d20 = v2.dot3d(v0);
	float32x16 d21 = v2.dot3d(v1);
	float32x16 den = d00 * d11 - (d01 * d01);
	beta = (d11 * d20 - d01 * d21) / den;
	gamma = (d00 * d21 - d01 * d20) / den;
	alpha = float32x16(1) - beta - gamma; //doesn't seem to hurt calculating it like this*/

	Vec4_f32x16 n = (B - A).cross3d(C - A);
	alpha = ((B - P).cross3d(C - P)).dot3d(n) / n.dot3d(n);
	beta  = ((C - P).cross3d(A - P)).dot3d(n) / n.dot3d(n);
	gamma = ((A - P).cross3d(B - P)).dot3d(n) / n.dot3d(n);
}

void mask_store_vec4_f32x16_to_framebuffer(const Vec4_f32x16& pack, void* frameBuffer, int x, int y, int w, Mask16 mask)
{
	//we have px[0] == r0,r1,r2...,r15, px[1] == g0,..g15, ...
	//DX wants: r0,g0,b0,a0,r1,g1,b1,a1, etc
	//Meanings, that first 16-wide register to store should be r0,g0,b0,a0,...,r3,g3,b3,a3
	//Second - 4-7, third - 8-11, fourth - 12-15
	constexpr int DC = 0; //garbage value
	//duplicate each opaquePixelsMask bit 4 times, i.e: 0123 -> 0000111122223333, 16 bits -> 64
	__m512i expanded = _mm512_maskz_mov_epi32(mask, _mm512_set1_epi32(-1));
	__mmask64 duplicated = _mm512_cmpneq_epi8_mask(expanded, _mm512_set1_epi32(0));

	__m256i ph_r = _mm512_cvtps_ph(pack.r, _MM_FROUND_NO_EXC);
	__m256i ph_g = _mm512_cvtps_ph(pack.g, _MM_FROUND_NO_EXC);
	__m256i ph_b = _mm512_cvtps_ph(pack.b, _MM_FROUND_NO_EXC);
	__m256i ph_a = _mm512_cvtps_ph(pack.a, _MM_FROUND_NO_EXC);
	for (int i = 0; i < 16; i += 4)
	{
		__m256i rg_ind = _mm256_add_epi16(_mm256_set1_epi16(i), _mm256_setr_epi16(0, 16, DC, DC, 1, 17, DC, DC, 2, 18, DC, DC, 3, 19, DC, DC));
		__m256i ba_ind = _mm256_add_epi16(_mm256_set1_epi16(i), _mm256_setr_epi16(DC, DC, 0, 16, DC, DC, 1, 17, DC, DC, 2, 18, DC, DC, 3, 19));
		__m256i rgxx = _mm256_permutex2var_epi16(ph_r, rg_ind, ph_g);
		__m256i xxba = _mm256_permutex2var_epi16(ph_b, ba_ind, ph_a);
		__m256i rgba = _mm256_mask_mov_epi16(rgxx, 0b1100110011001100, xxba);
		int storeInd = (y * w + x + i) * 4;
		_mm256_mask_storeu_epi16((int16_t*)frameBuffer + storeInd, duplicated >> (i * 4), rgba);
	}
}

Vec4_f32x16 mask_load_vec4_f32x16_from_framebuffer(const void* frameBuffer, int x, int y, int w, Mask16 mask)
{
	int loadInd = y * w + x;
	const int64_t* p = (const int64_t*)frameBuffer;
	__m512i rgba0_7 = _mm512_maskz_loadu_epi64(mask, p + loadInd);
	__m512i rgba8_15 = _mm512_maskz_loadu_epi64(mask >> 8, p + loadInd + 8);

	__m512i r0_15_g0_15_ph = _mm512_permutex2var_epi16(rgba0_7, _mm512_setr_epi16(0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61), rgba8_15);
	__m512i b0_15_a0_15_ph = _mm512_permutex2var_epi16(rgba0_7, _mm512_setr_epi16(2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63), rgba8_15);

	Vec4_f32x16 ret;
	ret.r = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(r0_15_g0_15_ph, 0));
	ret.g = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(r0_15_g0_15_ph, 1));
	ret.b = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(b0_15_a0_15_ph, 0));
	ret.a = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(b0_15_a0_15_ph, 1));
	return ret;
}

void RasterizingRenderer::drawTriangleBatch(const PixelStageInput& inp, const int threadIndex)
{
	const auto& drawCmd = *inp.cmd;
	float* zBuffer = (float*)drawCmd.buffers[0].data;
	uint64_t* frameBuffer = drawCmd.buffers.size() >= 2 ? (uint64_t*)drawCmd.buffers[1].data : nullptr;
	float my_xMin, my_xMax, my_yMin, my_yMax;
	drawCmd.zoneManager.getLimitsForThread(threadIndex, my_xMin, my_yMin, my_xMax, my_yMax);
	int w = drawCmd.renderW;
	bool depthOnly = drawCmd.recipe == DrawRecipe::MAIN_DEPTH_PREPASS || drawCmd.recipe == DrawRecipe::SHADOW_MAP_DEPTH;

	for (int outputTriangleIndex = 0; outputTriangleIndex < 2; ++outputTriangleIndex)
	{
		auto& currTriangles = inp.vertexStageOutput->outputTriangles[outputTriangleIndex];
		Mask16 currActiveTriangles = currTriangles.activeTrianges;
		if (!currActiveTriangles) break; //yes, break, not continue. If first triangle is invalid, then all are (at least in current pipeline)
		int32x16 diffuseMapIndex = _mm512_mask_i32gather_epi32(_mm512_setzero_epi32(), currActiveTriangles, inp.progenitorTriangleIndices, this->triangleStore.diffuseMapIndex.data(), 4);

		//TODO: doubling triangles if they are clipped!
		float32x16 group_xBeg = _mm512_max_ps(_mm512_set1_ps(my_xMin), currTriangles.minX);
		float32x16 group_yBeg = _mm512_max_ps(_mm512_set1_ps(my_yMin), currTriangles.minY);
		float32x16 group_xEnd = _mm512_min_ps(_mm512_set1_ps(my_xMax), currTriangles.maxX);
		float32x16 group_yEnd = _mm512_min_ps(_mm512_set1_ps(my_yMax), currTriangles.maxY);
		const VertexPack16& v0 = currTriangles.vertices[0];
		const VertexPack16& v1 = currTriangles.vertices[1];
		const VertexPack16& v2 = currTriangles.vertices[2];

		float32x16 group_initialAlpha, group_initialBeta, group_initialGamma;
		calculateBarycentricCoordinates2D({ group_xBeg, group_yBeg, 0.f, 0.f }, v0.space, v1.space, v2.space, currTriangles.rcpSignedArea, group_initialAlpha, group_initialBeta, group_initialGamma);
		float32x16 group_dAlpha_dx = (v1.space.y - v2.space.y) * currTriangles.rcpSignedArea;
		float32x16 group_dAlpha_dy = (v2.space.x - v1.space.x) * currTriangles.rcpSignedArea;
		float32x16 group_dBeta_dx = (v2.space.y - v0.space.y) * currTriangles.rcpSignedArea;
		float32x16 group_dBeta_dy = (v0.space.x - v2.space.x) * currTriangles.rcpSignedArea;
		float32x16 group_dGamma_dx = -group_dAlpha_dx - group_dBeta_dx; //TODO: replace with proper calculation, precision issues!
		float32x16 group_dGamma_dy = -group_dAlpha_dy - group_dBeta_dy; //TODO: replace with proper calculation, precision issues!

		for (int i = 0; i < 16; ++i)
		{
			if ((currActiveTriangles.mask & (1 << i)) == 0) continue;
			int currDiffuseMapIndex = diffuseMapIndex[i];

			const auto& texture = this->textureManager.getTextureByHandle(currDiffuseMapIndex);
			for (float y = group_yBeg[i]; y <= group_yEnd[i]; ++y)
			{
				size_t yInt = y;
				size_t xInt = group_xBeg[i];
				float32x16 dy = y - group_yBeg[i];
				for (float32x16 x = float32x16::sequence() + group_xBeg[i]; Mask16 xBoundsMask = (x <= group_xEnd[i]); x += 16, xInt += 16)
				{
					float32x16 dx = x - group_xBeg[i];
					float32x16 alpha = dy * group_dAlpha_dy[i] + dx * group_dAlpha_dx[i] + group_initialAlpha[i];
					float32x16 beta = dy * group_dBeta_dy[i] + dx * group_dBeta_dx[i] + group_initialBeta[i];
					float32x16 gamma = dy * group_dGamma_dy[i] + dx * group_dGamma_dx[i] + group_initialGamma[i];
					Mask16 pointsInsideTriangleMask = (xBoundsMask & alpha >= 0.0) & (beta >= 0.0 & gamma >= 0.0);
					if (Statsman::ENABLED)
					{
						MyStatsman.rendering.barycentricsCalculated += 16;
						MyStatsman.rendering.pointsInsideTriangles += _mm_popcnt_u32(pointsInsideTriangleMask.mask);
					}
					if (!pointsInsideTriangleMask) continue;

					Vec4_f32x16 interpolatedDividedUv = Vec4_f32x16(v0.u[i], v0.v[i], v0.space.z[i], 0.f) * alpha +
						Vec4_f32x16(v1.u[i], v1.v[i], v1.space.z[i], 0.f) * beta +
						Vec4_f32x16(v2.u[i], v2.v[i], v2.space.z[i], 0.f) * gamma;
					float32x16 currDepthValues = _mm512_maskz_loadu_ps(pointsInsideTriangleMask, zBuffer + yInt * w + xInt);
					//depth test: bigger Z pre-divide = further. However, we have reciprocal Z stored in interpolatedDividedUv.z, and Z <= 1 are culled during clipping stage, thus 1/z < z at all times
					//example: Z post rotate and translate (but before divide) for 2 pixels are 2 and 3. After Z divide they become 0.5 and 0.333. 0.5 should win the depth test, since it's closer
					Mask16 notOccludedPoints = pointsInsideTriangleMask & currDepthValues < interpolatedDividedUv.z;
					if (Statsman::ENABLED)
					{
						MyStatsman.rendering.zBufferFetchLanes += 16;
						MyStatsman.rendering.zBufferFetchAliveLanes += _mm_popcnt_u32(pointsInsideTriangleMask.mask);
						MyStatsman.rendering.notOccludedPoints += _mm_popcnt_u32(notOccludedPoints.mask);
					}
					if (!notOccludedPoints) continue; //if all points are occluded, then skip

					Vec4_f32x16 uvCorrected = interpolatedDividedUv / interpolatedDividedUv.z;
					Vec4_f32x16 texturePixels;
					if (depthOnly)
					{
						auto accessor = texture.getGatherAccessor(uvCorrected.x, uvCorrected.y, notOccludedPoints);
						texturePixels.a = accessor.gatherA();
					}

					Mask16 opaquePixelsMask = notOccludedPoints & (texturePixels.a > 0.0f);
					if (!opaquePixelsMask) continue;

					_mm512_mask_storeu_ps(zBuffer + yInt * w + xInt, opaquePixelsMask, interpolatedDividedUv.z);

					if (drawCmd.recipe == DrawRecipe::MAIN_DEPTH_PREPASS)
					{
						_mm512_mask_storeu_epi32((uint32_t*)drawCmd.buffers[2].data + yInt * w + xInt, opaquePixelsMask, _mm512_set1_epi32(inp.progenitorTriangleIndices[i]));
					}

					if (Statsman::ENABLED)
					{
						MyStatsman.rendering.zBufferWriteLanes += 16;
						MyStatsman.rendering.zBufferWriteAliveLanes += _mm_popcnt_u32(opaquePixelsMask.mask);
						MyStatsman.rendering.frameBufWriteLanes += 16;
						MyStatsman.rendering.frameBufWriteAliveLanes += _mm_popcnt_u32(opaquePixelsMask.mask);
						MyStatsman.rendering.opaquePixels += _mm_popcnt_u32(opaquePixelsMask.mask);
					}
				}
			}
		}
	}
}

void RasterizingRenderer::rasterizerRoutine(int threadIndex)
{
	int threadCount = this->currGs->threadpool->getThreadCount();
	auto transformedResults = std::make_unique<VertexStageOutput[]>(this->drawCommands.size()); //this is called only once per frame per thread anyway, so no need to torture yourself with static arrays and checks

	VertexStageInput inp;
	inp.stage = 2;
	inp.nearPlaneZ = this->currGs->cameraPlane_zDist;
	for (int senderThreadIndex = 0; senderThreadIndex < threadCount; ++senderThreadIndex)
	{
		int storeIndex = senderThreadIndex * threadCount + threadIndex;
		for (int cmdIndex = 0; cmdIndex < this->drawCommands.size(); ++cmdIndex)
		{
			auto& currCmd = this->drawCommands[cmdIndex];
			auto& currStore = (*currCmd.trianglesToZones)[storeIndex];
			int storeSize = currStore.size();
			for (int currIndex = 0; currIndex < storeSize; currIndex += 16)
			{
				Mask16 storeBounds = (int32x16::sequence() + currIndex) < storeSize;
				static_assert(currStore.ELEMENTS_PER_BLOCK % 16 == 0, "Triangle bin block store is expected to be 16-element aligned.");
				int32x16 triangleIndices = _mm512_maskz_loadu_epi32(storeBounds, &currStore[currIndex]); //this will read out of block's bounds if ELEMENTS_PER_BLOCK is not divisible by 16.
				inp.triangleIndices = triangleIndices;
				inp.validInputs = storeBounds;
				inp.firstCmd = inp.lastCmd = cmdIndex;
				for (int i = 0; i < 3; ++i)
				{
					inp.vertexIndices[i] = _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), storeBounds, triangleIndices, this->triangleStore.vertInd[i].data(), 4);
				}

				this->transformVertices(inp, transformedResults.get());
				PixelStageInput pxInp;
				pxInp.cmd = &currCmd;
				pxInp.vertexStageOutput = &transformedResults[cmdIndex];
				pxInp.progenitorTriangleIndices = triangleIndices;
				this->drawTriangleBatch(pxInp, threadIndex);
			}
		}
	}
}
__m512 gather_render_job_attributes_from_render_job_ptrs(__m512i ptrs0_7, __m512i ptrs8_15, int attrOffsetInRenderJob, Mask16 mask)
{
	__m512i addr0_7 = _mm512_add_epi64(ptrs0_7, _mm512_set1_epi64(attrOffsetInRenderJob));
	__m512i addr8_15 = _mm512_add_epi64(ptrs8_15, _mm512_set1_epi64(attrOffsetInRenderJob));
	__m256 attr0 = _mm512_mask_i64gather_ps(_mm256_setzero_ps(), mask, addr0_7, nullptr, 1);
	__m256 attr1 = _mm512_mask_i64gather_ps(_mm256_setzero_ps(), mask >> 8, addr8_15, nullptr, 1);
	return _mm512_insertf32x8(_mm512_castps256_ps512(attr0), attr1, 1);
}


void RasterizingRenderer::joinMainWithShadowMap(int threadIndex)
{
	float my_xMin, my_xMax, my_yMin, my_yMax;
	this->drawCommands[0].zoneManager.getLimitsForThread(threadIndex, my_xMin, my_yMin, my_xMax, my_yMax);
	int w = this->drawCommands[0].renderW;
	bool texturingEnabled = this->currGs->texturingEnabled;

	float* shadowMap_zBuffer = this->shadowMapEnabled ? (float*)this->drawCommands[1].buffers[0].data : nullptr;
	float* main_zBuffer = (float*)this->drawCommands[0].buffers[0].data;
	float* main_frameBuffer = (float*)this->drawCommands[0].buffers[1].data;
	uint32_t* renderJobPtrsBuffer = (uint32_t*)this->drawCommands[0].buffers[2].data;
	ShadingMode shadingMode = this->drawCommands[0].shadingMode;
	for (float y = my_yMin; y < my_yMax; ++y)
	{
		size_t yInt = y;
		for (float32x16 x = float32x16::sequence() + my_xMin; Mask16 xBoundsMask = x <= my_xMax; x += 16)
		{
			size_t xInt = x[0];
			if (this->drawShadowMapDebug) //debug draw shadow map to screen
			{
				float smw = this->drawCommands[1].renderW;
				float smh = this->drawCommands[1].renderH;
				Mask16 sm_xBounds = x < smw;
				if (sm_xBounds && y < smh)
				{
					float32x16 z = _mm512_maskz_loadu_ps(sm_xBounds, shadowMap_zBuffer + yInt * this->drawCommands[1].renderW + xInt);
					float32x16 dz = float32x16(1) / z;
					float32x16 distIntensity = float32x16(1) - dz / (dz + 100.f);

					Vec4_f32x16 colOut;
					colOut.r = colOut.g = colOut.b = distIntensity;
					colOut.a = 1;
					mask_store_vec4_f32x16_to_framebuffer(colOut, main_frameBuffer, xInt, yInt, this->drawCommands[0].renderW, sm_xBounds);
					continue;
				}
			}

			float32x16 zInvSrc = _mm512_maskz_loadu_ps(xBoundsMask, main_zBuffer + yInt * w + xInt);
			Vec4_f32x16 screenPos(x, y, 1, zInvSrc);
			Vec4_f32x16 worldCoords = this->drawCommands[0].ctr.inverseScreenPixelsToWorld(screenPos);

			int32x16 triangleIndices = _mm512_maskz_loadu_epi32(xBoundsMask, renderJobPtrsBuffer + yInt * w + xInt);
			Mask16 filledPixels = xBoundsMask & (triangleIndices != -1);
			if (!filledPixels) continue;

			VertexPack16 untransformedVerts[3];
			for (int i = 0; i < 3; ++i)
			{
				int32x16 vInd = _mm512_mask_i32gather_epi32(_mm512_setzero_epi32(), filledPixels, triangleIndices, this->triangleStore.vertInd[i].data(), 4);
				this->vertexStore.gatherWorldXYZ(vInd, filledPixels, untransformedVerts[i].space);

				this->vertexStore.gatherUV(vInd, filledPixels, untransformedVerts[i].u, untransformedVerts[i].v);
				if (shadingMode == ShadingMode::SMOOTH)
				{
					this->vertexStore.gatherNormals(vInd, filledPixels, untransformedVerts[i].normal);
				}
			}

			const Vec4_f32x16& r1 = untransformedVerts[0].space;
			const Vec4_f32x16& r2 = untransformedVerts[1].space;
			const Vec4_f32x16& r3 = untransformedVerts[2].space;

			float32x16 alpha, beta, gamma;
			calculateBarycentricCoordinates3D(worldCoords, r1, r2, r3, alpha, beta, gamma);

			Vec4_f32x16 uv = Vec4_f32x16(untransformedVerts[0].u, untransformedVerts[0].v, 0.f, 0.f) * alpha +
				Vec4_f32x16(untransformedVerts[1].u, untransformedVerts[1].v, 0.f, 0.f) * beta +
				Vec4_f32x16(untransformedVerts[2].u, untransformedVerts[2].v, 0.f, 0.f) * gamma;
			//Vec4_f32x16 normals = Vec4_f32x16(untransformedVerts[0].normal.x, untransformedVerts[0].normal.y, 0.f, 0.f) * alpha +
			//	Vec4_f32x16(untransformedVerts[1].u, untransformedVerts[1].v, 0.f, 0.f) * beta +
			//	Vec4_f32x16(untransformedVerts[2].u, untransformedVerts[2].v, 0.f, 0.f) * gamma;

			Vec4_f32x16 texturePixels;
			if (texturingEnabled)
			{
#if defined(VS_CLANG) || 1 //TODO: Clang crashes immediately with multitexturing for some reason, so this workaround just scalarizes it for Clang

				for (int j = 0; j < 16; ++j)
				{
					if (!(filledPixels.mask & (1 << j))) continue;
					int diffuseMapIndex = this->triangleStore.diffuseMapIndex[triangleIndices[j]];
					Vec4f pixel = this->textureManager.getTextureByHandle(diffuseMapIndex).getLinearIntensity(uv.x[j], uv.y[j]);
					texturePixels.x[j] = pixel.x;
					texturePixels.y[j] = pixel.y;
					texturePixels.z[j] = pixel.z;
					texturePixels.w[j] = pixel.w;
				}
#else
				int32x16 diffuseMapIndices = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(0), filledPixels, triangleIndices, this->original_triangleStore.diffuseMapIndex.data(), 4);
				texturePixels = this->textureManager.gatherLinearIntensitiesFromMultipleTextures(diffuseMapIndices, uv.x, uv.y, filledPixels);
#endif
			}
			else
			{
				float32x16 dz = float32x16(1) / zInvSrc;
				float32x16 distIntensity = float32x16(1) - dz / (dz + 100.f);
				texturePixels.r = texturePixels.g = texturePixels.b = distIntensity;
				texturePixels.a = 1;
			}
			filledPixels &= texturePixels.a > 0.f;
			texturePixels.x = _mm512_mask_mov_ps(_mm512_set1_ps(this->skyColor.x), filledPixels, texturePixels.x);
			texturePixels.y = _mm512_mask_mov_ps(_mm512_set1_ps(this->skyColor.y), filledPixels, texturePixels.y);
			texturePixels.z = _mm512_mask_mov_ps(_mm512_set1_ps(this->skyColor.z), filledPixels, texturePixels.z);

			Mask16 pointsInShadow = 0;
			if (this->shadowMapEnabled)
			{
				const auto& currentShadowMap = this->drawCommands[1];
				Vec4_f32x16 sunWorldPositions = currentShadowMap.ctr.getCurrentTransformationMatrix() * worldCoords;
				float32x16 zInv = float32x16(1) / sunWorldPositions.z;
				Vec4_f32x16 sunScreenPositions = currentShadowMap.ctr.screenSpaceToPixels(sunWorldPositions * zInv);
				sunScreenPositions.z = zInv;
				sunScreenPositions.y = sunScreenPositions.y;

				Mask16 inShadowMapBounds = xBoundsMask & (sunScreenPositions.x >= 0.f) & (sunScreenPositions.x < float(this->drawCommands[1].renderW)) & (sunScreenPositions.y >= 0.f) & (sunScreenPositions.y < float(this->drawCommands[1].renderH));
				int32x16 gatherInd = int32x16(sunScreenPositions.y.trunc()) * this->drawCommands[1].renderW + int32x16(sunScreenPositions.x.trunc());
				float32x16 shadowMapDepths = _mm512_mask_i32gather_ps(_mm512_set1_ps(INFINITY), inShadowMapBounds, gatherInd, shadowMap_zBuffer, 4);

				pointsInShadow = ~inShadowMapBounds;
				if (this->useShadowMapBias)
				{
					float32x16 bias = 10.f; //still some acne, noticable panning
					float32x16 shadowMapProperDepth = float32x16(1) / shadowMapDepths;
					float32x16 geometryProperDepth = float32x16(1) / sunScreenPositions.z;
					pointsInShadow |= (shadowMapProperDepth + bias < geometryProperDepth);
				}
				else
				{
					pointsInShadow |= shadowMapDepths > sunScreenPositions.z;
				}
			}

			Vec4_f32x16 totalLight;
			for (int i = 0; i < 3; ++i) totalLight[i] = this->ambientLightIntensity;
			float32x16 normalDot;
			if (shadingMode == ShadingMode::SMOOTH)
			{
				Vec4_f32x16 normals = untransformedVerts[0].normal * alpha + untransformedVerts[1].normal * beta + untransformedVerts[2].normal * gamma;
				normals /= normals.len3d();
				Vec4f lightFrom = { 13.978434,1933.787476,117.000008 }, lightTo = { -874.297729,136.884766,0.909166 };
				Vec4_f32x16 lightDir = lightTo - lightFrom;
				lightDir /= lightDir.len3d();
				normalDot = -normals.dot3d(lightDir);
				totalLight += _mm512_max_ps(float32x16(0.f), normalDot * this->lightIntesity);
			}
			else totalLight += this->lightIntesity;

			for (int i = 0; i < 3; ++i) totalLight[i] = _mm512_mask_mov_ps(totalLight[i], pointsInShadow, float32x16(this->ambientLightIntensity));
			for (int i = 0; i < 3; ++i) texturePixels[i] = _mm512_mask_mul_ps(texturePixels[i], filledPixels, totalLight[i], texturePixels[i]); //unfilled pixels (sky) is invulnerable to lighting!
			mask_store_vec4_f32x16_to_framebuffer(texturePixels, main_frameBuffer, xInt, yInt, this->drawCommands[0].renderW, xBoundsMask);
		}
	}
}

/*
std::array<VertexPack16,3> Rasterizing::RenderJob_Store::loadVertices16(size_t firstInd, Mask16 mask) const
{
	std::array<VertexPack16,3> ret;
	for (int i = 0; i < 3; ++i) 
	{
		
		for (int j = 0; j < 16; ++j)
		{
			if ((mask.mask & (1 << j)) == 0) continue;
			int ind = firstInd + j;
			ret[i].space.x[j] = this->jobs[ind].x[i];
			ret[i].space.y[j] = this->jobs[ind].y[i];
			ret[i].space.z[j] = this->jobs[ind].z[i];
			ret[i].normal.x[j] = this->jobs[ind].nx[i];
			ret[i].normal.y[j] = this->jobs[ind].ny[i];
			ret[i].normal.z[j] = this->jobs[ind].nz[i];
			ret[i].u[j] = this->jobs[ind].u[i];
			ret[i].v[j] = this->jobs[ind].v[i];
		}
	}
	return ret;
}
*/