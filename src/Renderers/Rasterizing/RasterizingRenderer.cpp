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
	this->vertexStore.clear();
	this->triangleStore.clear();
	this->textureManager.clear();
	this->sceneModels.clear();
	this->sceneExposionInProgress = std::nullopt;

	std::mutex mtx;
	std::vector<Threadpool::TaskHandle> tasks;
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
		std::vector<Threadpool::TaskHandle> textureLoadingTasks(importModelCount);

		Threadpool::Task tsk;
		for (int i = 0; i < importModelCount; ++i)
		{
			tsk.func = [&, this, i]() {
				if (loadedModels[i].diffuseMapPath) diffuseMapIndices[i] = this->textureManager.addTextureByPath(*loadedModels[i].diffuseMapPath);
				};
			textureLoadingTasks[i] = Threadpool::instance->addTask(tsk);
		}

		size_t loadedTriangles = 0;
		size_t firstModelInd = this->sceneModels.size();
		for (int i = 0; i < loadedModels.size(); ++i)
		{
			size_t discardedTriangles = 0;
			Model& m = this->sceneModels.emplace_back();
			m.globalTriangleRange.min = this->triangleStore.size(); //since textures are not yet loaded, diffuseMapIndex is not filled, and triangle store size() will be wrong!
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
				uint32_t vi[3];
				for (int k = 0; k < 3; ++k)
				{
					vi[k] = this->vertexStore.insert(
						it.v[k].space.x, it.v[k].space.y, it.v[k].space.z, it.v[k].diffuseMapCoords.x, 1 - it.v[k].diffuseMapCoords.y, it.v[k].normal.x, it.v[k].normal.y, it.v[k].normal.z
					);

					//TODO: clean from degenerate triangles (i.e 2 or 3 vertices same or all 3 collinear)?
				}
				this->triangleStore.insert(vi[0], vi[1], vi[2], -1);
			}
			m.globalTriangleRange.max = this->triangleStore.size() - 1;
			//std::cout << "Loaded " << m.triangleStore.size() << " triangles out of " << loadedModels[i].triangles.size() << " (" << discardedTriangles << " discarded) from " << path << "\n";
		}

		Threadpool::instance->blockUntilComplete(textureLoadingTasks);
		size_t lastModelInd = this->sceneModels.size() - 1;
		for (int i = 0; i < loadedModels.size(); ++i)
		{
			auto& currModel = this->sceneModels[firstModelInd + i];
			//triangleStore.diffuseMapIndex.resize()
			bool noBackfaceCulling = !this->textureManager.getTextureByHandle(diffuseMapIndices[i]).areAllPixelsOpaque();
			ModelFlags flags = noBackfaceCulling ? NO_BACKFACE_CULLING : NONE;
			for (int j = currModel.globalTriangleRange.min; j <= currModel.globalTriangleRange.max; ++j)
			{
				this->triangleStore.setDiffuseMapIndex(j, diffuseMapIndices[i]);
				this->triangleStore.modelFlags.push_back(flags);
			}
		}
	}

	Threadpool::instance->blockUntilComplete(tasks);
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

	
#ifdef NDEBUG
	int shadowMapW = 512*3;
	int shadowMapH = 288*3;
#else
	int shadowMapW = 51;
	int shadowMapH = 28;
#endif
	C_Input& inp = C_Input::getInstance();
	if (inp.wasCharPressedOnThisFrame('N')) this->shadingMode = EnumCycler::next(this->shadingMode);
	if (inp.wasCharPressedOnThisFrame('M')) this->drawShadowMapDebug ^= 1;
	if (inp.wasCharPressedOnThisFrame('B')) this->faceCullingType = EnumCycler::next(this->faceCullingType);
	if (inp.wasButtonPressedOnThisFrame(SDL_SCANCODE_KP_5)) this->sceneExposionInProgress = ExplodeAndRestoreSceneEffect(settings.gameTime, 0.1, 15, this->triangleStore.size());
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

	
	if (this->sceneExposionInProgress)
	{
		this->sceneExposionInProgress->onFrameStart(settings.gameTime);
		if (this->sceneExposionInProgress->isFinished()) this->sceneExposionInProgress = std::nullopt;
	}

	Threadpool* threadpool = settings.threadpool;
	int threadCount = threadpool->getWorkerCount();

	int w = (int)settings.outputTextureParams.Width;
	int h = (int)settings.outputTextureParams.Height;
	uint64_t skyColorFP16 = _mm_extract_epi64(_mm_cvtps_ph(this->skyColor, _MM_FROUND_TO_NEAREST_INT), 0);
	this->depthBufMain.resize(w, h);
	this->triangleIndexBuf.resize(w, h);
	this->frameBuf = Buffer<uint64_t>((uint64_t*)this->currGs->graphicsOutputBuffer, w, h, skyColorFP16);
	this->drawCommands.clear();
	this->depthBufMain.clearValue = this->depthBufShadowMap.clearValue = -FLT_MAX;
	this->triangleIndexBuf.clearValue = -1;

	DrawCommand& mainDrawCmd = this->drawCommands.emplace_back();
	mainDrawCmd.ctr = { w,h }; 
	mainDrawCmd.ctr.prepare(settings.camPos, settings.camAng);
	mainDrawCmd.shadingMode = this->shadingMode;
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
		this->depthBufShadowMap.resize(shadowMapW, shadowMapH);
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

	std::vector<Threadpool::TaskHandle> transformTasks, drawTasks;
	Threadpool::Task tsk;
	for (int threadIndex = 0; threadIndex < threadCount; ++threadIndex)
	{
		tsk.func = [&, threadIndex]() {
			this->binTrianglesIntoZones(threadIndex);
		};
		transformTasks.emplace_back(threadpool->addTask(tsk));
	}

	tsk.dependencies = transformTasks;

	size_t renderJobCount = 0;
	//for (auto& it : this->renderJobsFromThreads) renderJobCount += it.size();
	//std::cout << renderJobCount << " render jobs\n";
	for (int threadIndex = 0; threadIndex < threadCount; ++threadIndex)
	{
		tsk.func = [&, threadIndex]() {
			this->rasterizerRoutine(threadIndex);
			};
		drawTasks.emplace_back(threadpool->addTask(tsk));
	}
	tsk.dependencies = drawTasks;
	drawTasks.clear();
	
	for (int threadIndex = 0; threadIndex < threadCount; ++threadIndex)
	{
		tsk.func = [&, threadIndex]() {
			this->joinMainWithShadowMap(threadIndex);
		};
		drawTasks.emplace_back(threadpool->addTask(tsk));
	}
	threadpool->blockUntilComplete(drawTasks);
	
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

	uint32_t vi[3];
	for (int k = 0; k < 3; ++k)
	{
		vi[k] = this->vertexStore.insert(vertices[k].x, -vertices[k].y, vertices[k].z, uvs[k].x, uvs[k].y, 1, 1, 1);
	}
	this->triangleStore.insert(vi[0], vi[1], vi[2], 0, 0, ModelFlags::NONE);
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
	if (!input.validInputs) return;
	//assert(std::fmod(input.threadCount, 1) == 0);
	std::array<VertexPack16, 3> originalVertices;
	for (int i = 0; i < 3; ++i)
	{
		this->vertexStore.gatherXYZUV(input.vertexIndices[i], input.validInputs, originalVertices[i].space, originalVertices[i].u, originalVertices[i].v);
		originalVertices[i].space.w = 1;
	}

	if (this->sceneExposionInProgress)
	{
		this->sceneExposionInProgress->applyToTrianglesInPlace(originalVertices, input.triangleIndices, input.validInputs);
	}
	bool UVs_loaded = true, normals_loaded = false;

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
				/*
				if (!UVs_loaded)
				{
					for (int i = 0; i < 3; ++i)
					{
						//activeTriangles may be different between commands, so gather by least restrictive valid mask, which is the input valid mask
						this->vertexStore.gatherUV(input.vertexIndices[i], input.validInputs, originalVertices[i].u, originalVertices[i].v);
					}
					UVs_loaded = true;
				}*/

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
			float32x16 minX = FLT_MAX, maxX = -FLT_MAX, minY = FLT_MAX, maxY = -FLT_MAX;
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
	uint64_t beginTicks = SDL_GetTicksNS();
	this->frameBuf.clearThreadZone(threadIndex);
	uint64_t framebufCleanTicks = SDL_GetTicksNS();
	this->depthBufMain.clearThreadZone(threadIndex);
	uint64_t depthBufCleanTicks = SDL_GetTicksNS();
	this->triangleIndexBuf.clearThreadZone(threadIndex);
	uint64_t triangleIndexBufCleanTicks = SDL_GetTicksNS();
	uint64_t shadowMapDepthCleanTicks = 0;
	if (this->shadowMapEnabled)
	{
		this->depthBufShadowMap.clearThreadZone(threadIndex);
		shadowMapDepthCleanTicks = SDL_GetTicksNS();
	}

	if (Statsman::ENABLED)
	{
		MyStatsman.rasterizing.frameBufferCleanMs = (framebufCleanTicks - beginTicks) / 1e6;
		MyStatsman.rasterizing.zBufferCleanMs = (depthBufCleanTicks - framebufCleanTicks) / 1e6;
		MyStatsman.rasterizing.triangleIndexBufferCleanMs = (triangleIndexBufCleanTicks - depthBufCleanTicks) / 1e6;
		if (shadowMapDepthCleanTicks) MyStatsman.rasterizing.shadowMapDepthBufferCleanMs = (shadowMapDepthCleanTicks - triangleIndexBufCleanTicks) / 1e6;
	}

	auto [d_low, d_high] = Threadpool::instance->getLimitsForThread(threadIndex, 0, this->triangleStore.size());
	size_t startInd = d_low, stopInd = d_high;
	int threadCount = this->currGs->threadpool->getWorkerCount();

	VertexStageInput inp;
	inp.nearPlaneZ = this->currGs->cameraPlane_zDist;
	inp.stage = 1;
	inp.firstCmd = 0;
	inp.lastCmd = this->drawCommands.size() - 1;
	auto transformedResults = std::make_unique<VertexStageOutput[]>(this->drawCommands.size()); //this is called only once per frame per thread anyway, so no need to torture yourself with static arrays and checks
	for (size_t currTriangleIndex = startInd; currTriangleIndex < stopInd; currTriangleIndex += 16)
	{
		int32x16 triangleIndices = int32x16::sequence() + currTriangleIndex;
		int32x16 diffuseMapIndices;
		Mask16 groupActiveTriangles = triangleIndices < stopInd;
		this->triangleStore.loadVertexAndDiffuseMapIndices16(currTriangleIndex, groupActiveTriangles, inp.vertexIndices[0], inp.vertexIndices[1], inp.vertexIndices[2], diffuseMapIndices);

		if (this->skipTrianglesWithFallbackTexure)
		{
			groupActiveTriangles &= diffuseMapIndices != 0;
			if (!groupActiveTriangles) continue;
		}

		inp.triangleIndices = triangleIndices;
		inp.validInputs = groupActiveTriangles;

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

void RasterizingRenderer::drawTriangleBatch(const PixelStageInput& inp, const int threadIndex)
{
	const auto& drawCmd = *inp.cmd;

	float* zBuffer = (drawCmd.recipe == DrawRecipe::MAIN_DEPTH_PREPASS ? this->depthBufMain : this->depthBufShadowMap).get();
	uint64_t* frameBuffer = this->frameBuf.get();
	uint32_t* triangleIndBuf = this->triangleIndexBuf.get();
	float my_xMin, my_xMax, my_yMin, my_yMax;
	drawCmd.zoneManager.getLimitsForThread(threadIndex, my_xMin, my_yMin, my_xMax, my_yMax);
	int w = drawCmd.renderW;
	bool depthOnly = drawCmd.recipe == DrawRecipe::MAIN_DEPTH_PREPASS || drawCmd.recipe == DrawRecipe::SHADOW_MAP_DEPTH;

	for (int outputTriangleIndex = 0; outputTriangleIndex < 2; ++outputTriangleIndex)
	{
		auto& currTriangles = inp.vertexStageOutput->outputTriangles[outputTriangleIndex];
		Mask16 currActiveTriangles = currTriangles.activeTrianges;
		if (!currActiveTriangles) break; //yes, break, not continue. If first triangle is invalid, then all are (at least in current pipeline)

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
		float32x16 group_dGamma_dx = (v0.space.y - v1.space.y) * currTriangles.rcpSignedArea; //this should have better precision than -group_dAlpha_dx - group_dBeta_dx since y2 should cancel out completely algebraically;
		float32x16 group_dGamma_dy = (v1.space.x - v0.space.x) * currTriangles.rcpSignedArea; //same for -group_dAlpha_dy - group_dBeta_dy and x2

		for (int i = 0; i < 16; ++i)
		{
			if ((currActiveTriangles.mask & (1 << i)) == 0) continue;
			int currDiffuseMapIndex = inp.diffuseMapIndices[i];

			const auto& texture = this->textureManager.getTextureByHandle(currDiffuseMapIndex);
			//4x4 packed layout is much more friendly to small geometry compared to 1x16 (much less dead lanes),
			//while penalties from having to split one 512 bit memory operation with 4x128 are minimal
			for (float32x16 y = float32x16(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3) + group_yBeg[i]; y <= group_yEnd[i]; y += 4)
			{
				float32x16 dy = y - group_yBeg[i];
				uint32_t yStart = y[0];
				for (float32x16 x = float32x16(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3) + group_xBeg[i]; Mask16 boundsMask = (y <= group_yEnd[i]) & (x <= group_xEnd[i]); x += 4)
				{
					uint32_t xStart = x[0];
					float32x16 dx = x - group_xBeg[i];
					float32x16 alpha = dy * group_dAlpha_dy[i] + dx * group_dAlpha_dx[i] + group_initialAlpha[i];
					float32x16 beta = dy * group_dBeta_dy[i] + dx * group_dBeta_dx[i] + group_initialBeta[i];
					float32x16 gamma = dy * group_dGamma_dy[i] + dx * group_dGamma_dx[i] + group_initialGamma[i];
					Mask16 pointsInsideTriangleMask = (boundsMask & alpha >= 0.0) & (beta >= 0.0 & gamma >= 0.0);
					if (Statsman::ENABLED)
					{
						MyStatsman.rasterizing.barycentricsCalculated += 16;
						MyStatsman.rasterizing.pointsInsideTriangles += _mm_popcnt_u32(pointsInsideTriangleMask.mask);
					}
					if (!pointsInsideTriangleMask) continue;

					Vec4_f32x16 interpolatedDividedUv = Vec4_f32x16(v0.u[i], v0.v[i], v0.space.z[i], 0.f) * alpha +
						Vec4_f32x16(v1.u[i], v1.v[i], v1.space.z[i], 0.f) * beta +
						Vec4_f32x16(v2.u[i], v2.v[i], v2.space.z[i], 0.f) * gamma;

					float32x16 currDepthValues = mask_load_rows_4x128_to_512_ps(pointsInsideTriangleMask, zBuffer, xStart, yStart, w);

					//float32x16 currDepthValues = _mm512_maskz_loadu_ps(pointsInsideTriangleMask, zBuffer + yInt * w + xInt);
					//depth test: bigger Z pre-divide = further. However, we have reciprocal Z stored in interpolatedDividedUv.z, and Z <= 1 are culled during clipping stage, thus 1/z < z at all times
					//example: Z post rotate and translate (but before divide) for 2 pixels are 2 and 3. After Z divide they become 0.5 and 0.333. 0.5 should win the depth test, since it's closer
					Mask16 notOccludedPoints = pointsInsideTriangleMask & currDepthValues < interpolatedDividedUv.z;
					if (Statsman::ENABLED)
					{
						MyStatsman.rasterizing.zBufferFetchLanes += 16;
						MyStatsman.rasterizing.zBufferFetchAliveLanes += _mm_popcnt_u32(pointsInsideTriangleMask.mask);
						MyStatsman.rasterizing.notOccludedPoints += _mm_popcnt_u32(notOccludedPoints.mask);
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

					mask_store_rows_512_to_4x128_ps(interpolatedDividedUv.z, opaquePixelsMask, zBuffer, xStart, yStart, w);

					if (drawCmd.recipe == DrawRecipe::MAIN_DEPTH_PREPASS)
					{
						mask_store_rows_512_to_4x128_ps(_mm512_castsi512_ps(_mm512_set1_epi32(inp.progenitorTriangleIndices[i])), opaquePixelsMask, triangleIndBuf, xStart, yStart, w);
					}

					if (Statsman::ENABLED)
					{
						MyStatsman.rasterizing.zBufferWriteLanes += 16;
						MyStatsman.rasterizing.zBufferWriteAliveLanes += _mm_popcnt_u32(opaquePixelsMask.mask);
						MyStatsman.rasterizing.frameBufWriteLanes += 16;
						MyStatsman.rasterizing.frameBufWriteAliveLanes += _mm_popcnt_u32(opaquePixelsMask.mask);
						MyStatsman.rasterizing.opaquePixels += _mm_popcnt_u32(opaquePixelsMask.mask);
					}
				}
			}
		}
	}
}

void RasterizingRenderer::rasterizerRoutine(int threadIndex)
{
	int threadCount = this->currGs->threadpool->getWorkerCount();
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
				PixelStageInput pxInp;
				this->triangleStore.gatherVertexAndDiffuseMapIndices(triangleIndices, storeBounds, inp.vertexIndices[0], inp.vertexIndices[1], inp.vertexIndices[2], pxInp.diffuseMapIndices);

				this->transformVertices(inp, transformedResults.get());
				
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

	float* shadowMap_zBuffer = this->shadowMapEnabled ? this->depthBufShadowMap.get() : nullptr;
	float* main_zBuffer = this->depthBufMain.get();
	uint64_t* main_frameBuffer = this->frameBuf.get();
	uint32_t* renderJobPtrsBuffer = this->triangleIndexBuf.get();
	ShadingMode shadingMode = this->drawCommands[0].shadingMode;
	for (float y = my_yMin; y < my_yMax; ++y)
	{
		size_t yInt = y;
		for (float32x16 x = float32x16::sequence() + my_xMin; Mask16 xBoundsMask = x <= my_xMax; x += 16)
		{
			size_t xInt = x[0];
			if (this->drawShadowMapDebug && this->shadowMapEnabled) //debug draw shadow map to screen
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

			std::array<VertexPack16, 3> untransformedVerts;
			int32x16 vInd[3];
			int32x16 diffuseMapIndices;
			this->triangleStore.gatherVertexAndDiffuseMapIndices(triangleIndices, filledPixels, vInd[0], vInd[1], vInd[2], diffuseMapIndices);
			for (int i = 0; i < 3; ++i)
			{
				this->vertexStore.gatherXYZUV(vInd[i], filledPixels, untransformedVerts[i].space, untransformedVerts[i].u, untransformedVerts[i].v);
				if (shadingMode == ShadingMode::SMOOTH)
				{
					this->vertexStore.gatherNormals(vInd[i], filledPixels, untransformedVerts[i].normal);
				}
			}

			//TODO: not a good fix, but works. When adding more effects, this will have to be changed too.
			if (this->sceneExposionInProgress)
			{
				this->sceneExposionInProgress->applyToTrianglesInPlace(untransformedVerts, triangleIndices, filledPixels);
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
					int diffuseMapIndex = diffuseMapIndices[j];
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
			float32x16 shadowMult = 1.f;
			if (this->shadowMapEnabled)
			{
				shadowMult = 0.f;
				const auto& currentShadowMap = this->drawCommands[1];
				Vec4_f32x16 sunWorldPositions = currentShadowMap.ctr.getCurrentTransformationMatrix() * worldCoords;
				float32x16 zInv = float32x16(1) / sunWorldPositions.z;
				Vec4_f32x16 sunScreenPositions = currentShadowMap.ctr.screenSpaceToPixels(sunWorldPositions * zInv);
				//sunScreenPositions.x = _mm512_roundscale_ps(sunScreenPositions.x, _MM_FROUND_TO_NEAREST_INT);
				//sunScreenPositions.y = _mm512_roundscale_ps(sunScreenPositions.y, _MM_FROUND_TO_NEAREST_INT);
				sunScreenPositions.z = zInv;

				for (float oy = -1; oy <= 1; ++oy)
				{
					for (float ox = -1; ox <= 1; ++ox)
					{
						float32x16 sx = sunScreenPositions.x + ox;
						float32x16 sy = sunScreenPositions.y + oy;
						float32x16 sx0 = _mm512_floor_ps(sx);
						float32x16 sy0 = _mm512_floor_ps(sy);
						float32x16 sx1 = sx0 + 1;
						float32x16 sy1 = sy0 + 1;
						float32x16 fracX = sx - sx0;
						float32x16 fracY = sy - sy0;
						const float32x16 sampleX[] = { sx0, sx1, sx0, sx1 };
						const float32x16 sampleY[] = { sy0, sy0, sy1, sy1 };
						float32x16 smapSamples[4];
						for (int i = 0; i < 4; ++i)
						{
							float32x16 ssx = sampleX[i];
							float32x16 ssy = sampleY[i];

							Mask16 inShadowMapBounds = xBoundsMask & ssx >= 0.f & ssy >= 0.f & ssx < float(this->drawCommands[1].renderW) & ssy < float(this->drawCommands[1].renderH);
							int32x16 gatherInd = int32x16(ssy.trunc()) * this->drawCommands[1].renderW + int32x16(ssx.trunc());
							float32x16 shadowMapDepths = _mm512_mask_i32gather_ps(_mm512_set1_ps(FLT_MAX), inShadowMapBounds, gatherInd, shadowMap_zBuffer, 4);
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
							smapSamples[i] = _mm512_maskz_mov_ps(~pointsInShadow, float32x16(1));
						}

						float32x16 lx1 = lerp(smapSamples[0], smapSamples[1], fracX);// fracX* smapSamples[0] + (-fracX + 1) * smapSamples[1];
						float32x16 lx2 = lerp(smapSamples[2], smapSamples[3], fracX);// fracX*smapSamples[2] + (-fracX + 1) * smapSamples[3];
						shadowMult += lerp(lx1, lx2, fracY);//fracY * lx1 + (-fracY + 1) * lx2;
					}
				}
				shadowMult /= 9.f;
			}

			float32x16 normalShadingMult;
			float32x16 normalDot;
			if (shadingMode == ShadingMode::SMOOTH)
			{
				Vec4_f32x16 normals = untransformedVerts[0].normal * alpha + untransformedVerts[1].normal * beta + untransformedVerts[2].normal * gamma;
				normals /= normals.len3d();
				Vec4f lightFrom = { 13.978434,1933.787476,117.000008 }, lightTo = { -874.297729,136.884766,0.909166 };
				Vec4_f32x16 lightDir = lightTo - lightFrom;
				lightDir /= lightDir.len3d();
				normalDot = -normals.dot3d(lightDir);
				normalShadingMult = _mm512_max_ps(float32x16(0.f), normalDot * this->lightIntesity);
			}
			else normalShadingMult = 1.f;

			for (int i = 0; i < 3; ++i)
			{
				texturePixels[i] = _mm512_mask_mul_ps(texturePixels[i], filledPixels, texturePixels[i], shadowMult * normalShadingMult + this->ambientLightIntensity);//unfilled pixels (sky) is invulnerable to lighting!
			}
			/*
			for (int i = 0; i < 3; ++i) totalLight[i] *= shadowMult;//_mm512_mask_mov_ps(totalLight[i], pointsInShadow, float32x16(this->ambientLightIntensity));
			for (int i = 0; i < 3; ++i) texturePixels[i] = _mm512_mask_mul_ps(texturePixels[i], filledPixels, totalLight[i], texturePixels[i]); //unfilled pixels (sky) is invulnerable to lighting!*/
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