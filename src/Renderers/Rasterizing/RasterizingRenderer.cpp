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
	if (false)
	{
		Model& m = this->sceneModels.emplace_back();
		Vec4f vertices[3] = {
			{-50, 0, 50},
			{50, 0, 50},
			{50, 20, 50},
		};
		for (int k = 0; k < 3; ++k)
		{
			uint32_t vertInd = this->original_verticeStore.insert(vertices[k].x, -vertices[k].y, vertices[k].z, 0.5, -0.5, 0, 0, 0);
			m.triangleStore.vertInd[k].push_back(vertInd);
			m.diffuseMapIndex = 0;
		}
		return;
	}

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
					uint32_t vertInd = this->original_verticeStore.insert(
						it.v[k].space.x, it.v[k].space.y, it.v[k].space.z, it.v[k].diffuseMapCoords.x, 1 - it.v[k].diffuseMapCoords.y, it.v[k].normal.x, it.v[k].normal.y, it.v[k].normal.z
					);

					//TODO: clean from degenerate triangles (i.e 2 or 3 vertices same or all 3 collinear)?
					m.triangleStore.vertInd[k].push_back(vertInd);
				}
			}
			std::cout << "Loaded " << m.triangleStore.size() << " triangles out of " << loadedModels[i].triangles.size() << " (" << discardedTriangles << " discarded) from " << path << "\n";
		}

		Threadpool::instance->waitForMultipleTasks(textureLoadingTasks);
		size_t lastModelInd = this->sceneModels.size() - 1;
		for (int i = 0; i < loadedModels.size(); ++i)
		{
			auto& currModel = this->sceneModels[firstModelInd + i];
			currModel.diffuseMapIndex = diffuseMapIndices[i];
			currModel.noBackfaceCulling = !this->textureManager.getTextureByHandle(diffuseMapIndices[i]).areAllPixelsOpaque();
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
	this->modelSlicesForThreads = this->makeModelSliceList();
	this->zBuffer.resize(settings.outputTextureParams.Width * settings.outputTextureParams.Height);
	int shadowMapW = 2560;//512*3;
	int shadowMapH = 1440;//288*3;
	this->shadowMap_zBuffer.resize(shadowMapW * shadowMapH);
	
	C_Input& inp = C_Input::getInstance();
	if (inp.wasCharPressedOnThisFrame('N')) this->shadingMode = EnumCycler::next(this->shadingMode);
	Threadpool* threadpool = settings.threadpool;
	int threadCount = threadpool->getThreadCount();


	DrawCommand mainDrawCmd;
	mainDrawCmd.ctr = { (int)settings.outputTextureParams.Width, (int)settings.outputTextureParams.Height }; 
	mainDrawCmd.ctr.prepare(settings.camPos, settings.camAng);
	mainDrawCmd.shadingMode = this->shadingMode;
	mainDrawCmd.renderW = settings.outputTextureParams.Width; //TODO: transformer has W and H already, infer it?
	mainDrawCmd.renderH = settings.outputTextureParams.Height;
	mainDrawCmd.needsUVs = true;
	mainDrawCmd.needsNormals = true;
	mainDrawCmd.faceCullingType = this->currGs->backfaceCullingEnabled ? FaceCullingType::BACKFACE : FaceCullingType::NONE;
	mainDrawCmd.outputStores = &this->mainRenderJobs;
	mainDrawCmd.threadCount = threadCount;
	mainDrawCmd.zBuffer = this->zBuffer.data();
	mainDrawCmd.frameBuffer = settings.graphicsOutputBuffer;
	this->drawCommands[0] = mainDrawCmd;

	DrawCommand shadowMapDrawCmd;
	shadowMapDrawCmd.ctr = { (int)shadowMapW, (int)shadowMapH };
	shadowMapDrawCmd.ctr.prepare(Vec4f(1281.845703, 2235.967773, 178.236572, 0.000000), Vec4f(0.000000, 4.523108, 0.797002, 0.000000));
	//shadowMapDrawCmd.ctr.prepare(Vec4f(-86.050537, 1644.088623, 710.859253, 0.000000), Vec4f(0.000000, -3.165947,0.366014,0.000000));
	//shadowMapDrawCmd.ctr.prepare(Vec4f(44.960358, 2656.120605,-223.813354, 0.000000), Vec4f(0.000000,1.054968,0.813000,0.000000));
	//shadowMapDrawCmd.ctr.prepare(Vec4f(44.960358, 2656.120605,-223.813354, 0.000000), Vec4f(0.000000,1.054968,0.813000,0.000000));
	//shadowMapDrawCmd.ctr.prepare(settings.camPos, settings.camAng);
	shadowMapDrawCmd.outputStores = &this->shadowMapRenderJobs;
	shadowMapDrawCmd.renderW = shadowMapW; //TODO: transformer has W and H already, infer it?
	shadowMapDrawCmd.renderH = shadowMapH;
	shadowMapDrawCmd.needsUVs = true;
	shadowMapDrawCmd.needsNormals = false;
	shadowMapDrawCmd.faceCullingType = FaceCullingType::FRONTFACE; //FaceCullingType::FRONT
	shadowMapDrawCmd.outputStores = &this->shadowMapRenderJobs;
	shadowMapDrawCmd.threadCount = threadCount;
	shadowMapDrawCmd.zBuffer = this->shadowMap_zBuffer.data();
	shadowMapDrawCmd.depthOnly = true;
	this->drawCommands[1] = shadowMapDrawCmd;

	for (auto& currSub : this->drawCommands)
	{
		if (currSub.outputStores->size() != threadCount) currSub.outputStores->resize(threadCount);
	}

	std::vector<task_id> transformTasks, drawTasks;
	for (int threadIndex = 0; threadIndex < threadCount; ++threadIndex)
	{
		transformTasks.emplace_back(threadpool->addTask(
			[&, threadIndex]() {
				this->doTransformationsAndClipping(threadIndex);
			}
		));
	}

	uint64_t bufCleanTicksBegin = SDL_GetTicksNS();
	for (auto& it : zBuffer) it = -INFINITY;
	for (auto& it : shadowMap_zBuffer) it = -INFINITY;
	uint64_t zBufCleanTicks = SDL_GetTicksNS();	
	
	int sz = settings.outputTextureParams.Width * settings.outputTextureParams.Height;
	uint64_t skyColor = _mm_extract_epi64(_mm_cvtps_ph(Vec4f(0.3, 0.7, 1, 1), _MM_FROUND_NO_EXC), 0);
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
				this->drawRenderJobs(threadIndex);
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
	//for (auto& it : renderJobsFromThreads) it.clear();
}

std::vector<std::vector<Rasterizing::ModelSlice>> RasterizingRenderer::makeModelSliceList() const
{
	//TODO: can frustrum culling be put here?
	std::vector<ModelSlice> availableSlices;
	size_t totalTriangleCount = 0;
	for (int modelIndex =0; modelIndex < sceneModels.size(); ++modelIndex)
	{
		totalTriangleCount += sceneModels[modelIndex].triangleStore.size();
		ModelSlice s;
		s.modelIndex = modelIndex;
		s.modelTriangleIndexBegin = 0;
		s.modelTriangleIndexEnd = sceneModels[modelIndex].triangleStore.size();
		availableSlices.push_back(s);
	}

	int threadCount = this->currGs->threadpool->getThreadCount();
	int trianglesPerThread = ceil(double(totalTriangleCount) / threadCount);
	std::vector<std::vector<ModelSlice>> ret(threadCount);
	for (int threadIndex = 0; threadIndex < threadCount-1; ++threadIndex)
	{
		size_t threadTrianglesRemaining = trianglesPerThread;
		while (threadTrianglesRemaining > 0 && !availableSlices.empty())
		{
			ModelSlice& s = availableSlices.back();
			size_t sliceTriangleCount = s.modelTriangleIndexEnd - s.modelTriangleIndexBegin;
			if (sliceTriangleCount <= threadTrianglesRemaining) //consume whole slice
			{
				ret[threadIndex].push_back(s);
				threadTrianglesRemaining -= sliceTriangleCount;
				availableSlices.pop_back();
			}
			else //decrement the slice by amount of triangles this thread takes
			{
				ModelSlice consumedSlice = s;
				consumedSlice.modelTriangleIndexBegin = s.modelTriangleIndexEnd - threadTrianglesRemaining;
				s.modelTriangleIndexEnd -= threadTrianglesRemaining;
				ret[threadIndex].push_back(consumedSlice);
				break;
				//threadTrianglesRemaining = 0;
			}
		}
	}
	ret[threadCount-1] = availableSlices; //last thread gets all remaining triangles. It should be at most triangleCount-1 bigger than the rest have, so OK

	size_t trianglesAfterDistribution = 0;
	for (auto& it : ret)
	{
		for (auto& s : it)
		{
			assert(s.modelTriangleIndexBegin < s.modelTriangleIndexEnd);
			assert(s.modelTriangleIndexBegin >= 0);
			assert(s.modelTriangleIndexEnd > 0);
			trianglesAfterDistribution += s.modelTriangleIndexEnd - s.modelTriangleIndexBegin;
		}
	}
	assert(trianglesAfterDistribution == totalTriangleCount);

	return ret;
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

void RasterizingRenderer::doTransformationsAndClipping(int threadIndex)
{
	uint64_t ticksBegin = SDL_GetTicksNS();
	int threadCount = this->currGs->threadpool->getThreadCount();
	if (this->mainRenderJobs[threadIndex].size() != threadCount) this->mainRenderJobs[threadIndex].resize(threadCount);
	if (this->shadowMapRenderJobs[threadIndex].size() != threadCount) this->shadowMapRenderJobs[threadIndex].resize(threadCount);
	assert(this->original_verticeStore.x.size() == this->original_verticeStore.y.size() && this->original_verticeStore.y.size() == this->original_verticeStore.z.size() && this->original_verticeStore.u.size() == this->original_verticeStore.v.size());

	for (auto& currSub : this->drawCommands)
	{
		for (auto& it : (*currSub.outputStores)[threadIndex]) it.clear();
	}
	
	float32x16 clippingZ = this->currGs->cameraPlane_zDist;
	size_t storedJobCount = 0;
	size_t seenTris = 0;

	for (auto& slice : this->modelSlicesForThreads[threadIndex])
	{
		//const float* xData = 
		seenTris += slice.modelTriangleIndexEnd - slice.modelTriangleIndexBegin;
		int myRenderJobCount = 0;
		//bool doBackfaceCulling = this->currGs->backfaceCullingEnabled && 
		for (int currModelTriangleIndex = slice.modelTriangleIndexBegin;
			currModelTriangleIndex < slice.modelTriangleIndexEnd; 
			currModelTriangleIndex += 16)
		{
			int32x16 triangleIndices = int32x16::sequence() + currModelTriangleIndex;
			Mask16 inBoundsTrianglesMask = triangleIndices < slice.modelTriangleIndexEnd;
			std::array<VertexPack16, 3> originalVertices;
			int32x16 verticeIndicesCache[3];
			const Model& model = sceneModels[slice.modelIndex];
			bool UVs_loaded = false, normalsLoaded = false;
			for (int i = 0; i < 3; ++i)
			{
				int32x16 verticeIndices = _mm512_maskz_loadu_epi32(inBoundsTrianglesMask, model.triangleStore.vertInd[i].data() + currModelTriangleIndex);
				originalVertices[i].space.x = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), inBoundsTrianglesMask, verticeIndices, this->original_verticeStore.x.data(), 4);
				originalVertices[i].space.y = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), inBoundsTrianglesMask, verticeIndices, this->original_verticeStore.y.data(), 4);
				originalVertices[i].space.z = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), inBoundsTrianglesMask, verticeIndices, this->original_verticeStore.z.data(), 4);
				originalVertices[i].space.w = 1;
				verticeIndicesCache[i] = verticeIndices;
			}

			
			for (int cmdIndex = 0; cmdIndex < this->drawCommands.size(); ++cmdIndex)
			{
				auto& currCmd = this->drawCommands[cmdIndex];
				float rcpScreenHeightPerThread = double(this->currGs->threadpool->getThreadCount()) / currCmd.renderH;
				float w = currCmd.renderW;
				float h = currCmd.renderH;


				int32x16 behindPlaneCount = 0;
				Mask16 behindPlaneMasks[3];
				std::array<VertexPack16, 6> output;
				for (int i = 0; i < 3; ++i)
				{
					Vec4_f32x16 rotatedTranslated = currCmd.ctr.rotateAndTranslate(originalVertices[i].space);
					Mask16 vertexBehindClippingPlane = rotatedTranslated.z < clippingZ;
					behindPlaneMasks[i] = vertexBehindClippingPlane;
					behindPlaneCount = _mm512_mask_add_epi32(behindPlaneCount, vertexBehindClippingPlane, behindPlaneCount, int32x16(1));
					output[i].space = rotatedTranslated;
				}

				if (Statsman::ENABLED)
				{
					int minVertInd = INT32_MAX, maxVertInd = INT32_MIN;
					for (int i = 0; i <= 3; ++i)
					{
						MyStatsman.triangles.verticesBehindNearPlane[i] += _mm512_mask_reduce_add_epi32(behindPlaneCount == i, _mm512_set1_epi32(1));
						if (i < 3)
						{
							minVertInd = std::min(minVertInd, _mm512_mask_reduce_min_epi32(inBoundsTrianglesMask, verticeIndicesCache[i]));
							maxVertInd = std::max(maxVertInd, _mm512_mask_reduce_max_epi32(inBoundsTrianglesMask, verticeIndicesCache[i]));
						}
						assert(maxVertInd >= minVertInd);
						assert(minVertInd >= 0);
						uint64_t delta = maxVertInd - minVertInd;
						MyStatsman.triangles.vertIndexDelta += delta;
						MyStatsman.triangles.vertIndexDeltaMin = std::min(MyStatsman.triangles.vertIndexDeltaMin.value_or(UINT64_MAX), delta);
						MyStatsman.triangles.vertIndexDeltaMax = std::max(MyStatsman.triangles.vertIndexDeltaMax.value_or(0), delta);
						MyStatsman.triangles.vertIndexDeltaCount += 1;
					}
				}
				if (!(behindPlaneCount != 3)) continue; //if all triangles have all vertices behind clipping plane, skip them

				Mask16 activeTrianglesMask = inBoundsTrianglesMask & behindPlaneCount != 3;

				if (currCmd.faceCullingType != FaceCullingType::NONE)
				{
					if (currCmd.faceCullingType == FaceCullingType::BACKFACE && !model.noBackfaceCulling)
					{
						Vec4_f32x16 transformedFaceNormals = getFaceNormalsForTriangles16(output[0].space, output[1].space, output[2].space);
						activeTrianglesMask &= output[0].space.dot3d(transformedFaceNormals) < 0.f;
					}
					if (currCmd.faceCullingType == FaceCullingType::FRONTFACE) //TODO: some flag to disable front face culling?
					{
						Vec4_f32x16 transformedFaceNormals = getFaceNormalsForTriangles16(output[0].space, output[1].space, output[2].space);
						activeTrianglesMask &= output[0].space.dot3d(transformedFaceNormals) >= 0.f;
					}
				}

				if (currCmd.needsUVs)
				{
					if (!UVs_loaded)
					{
						for (int i = 0; i < 3; ++i)
						{
							originalVertices[i].u = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), activeTrianglesMask, verticeIndicesCache[i], this->original_verticeStore.u.data(), 4);
							originalVertices[i].v = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), activeTrianglesMask, verticeIndicesCache[i], this->original_verticeStore.v.data(), 4);
						}
						UVs_loaded = true;
					}
					for (int i = 0; i < 3; ++i)
					{
						output[i].u = originalVertices[i].u;
						output[i].v = originalVertices[i].v;
					}
				}

				if (currCmd.needsNormals) //TODO: split this distinction into needs triangle normals and needs vertex normals?
				{
					if (!normalsLoaded)
					{
						for (int i = 0; i < 3; ++i)
						{
							originalVertices[i].normal.x = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), activeTrianglesMask, verticeIndicesCache[i], this->original_verticeStore.nx.data(), 4);
							originalVertices[i].normal.y = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), activeTrianglesMask, verticeIndicesCache[i], this->original_verticeStore.ny.data(), 4);
							originalVertices[i].normal.z = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), activeTrianglesMask, verticeIndicesCache[i], this->original_verticeStore.nz.data(), 4);
						}
						normalsLoaded = true;
					}
					for (int i = 0; i < 3; ++i)
					{
						switch (currCmd.shadingMode)
						{
						default:
							output[i].normal = originalVertices[i].normal;
							break;
						case ShadingMode::FLAT: //this mode is mostly an afterthought, so it's low priority to elude normal gathers in this case
							output[i].normal = getFaceNormalsForTriangles16(originalVertices[0].space, originalVertices[1].space, originalVertices[2].space); 
							break;
						case ShadingMode::NONE:
							break;
						}
					}
				}
				
				if (behindPlaneCount > 0) //near plane clipping
				{
					//continue;
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

					for (int i = 0; i < 3; ++i) clipOutput[i] = output[i];
					for (int frontVertex = 0; frontVertex < 3; ++frontVertex)
					{
						//these are behind
						int prevVertex = frontVertex == 0 ? 2 : frontVertex - 1;
						int nextVertex = frontVertex == 2 ? 0 : frontVertex + 1;
						Mask16 caseMask = ~behindPlaneMasks[frontVertex] & behindPlaneMasks[prevVertex] & behindPlaneMasks[nextVertex];
						clipOutput[0] = VertexPack16::maskMove(clipOutput[0], VertexPack16::lerpToClippingZ(output[prevVertex], output[frontVertex], clippingZ), caseMask);
						clipOutput[1] = VertexPack16::maskMove(clipOutput[1], output[frontVertex], caseMask);
						clipOutput[2] = VertexPack16::maskMove(clipOutput[2], VertexPack16::lerpToClippingZ(output[frontVertex], output[nextVertex], clippingZ), caseMask);
					}

					for (int behindVertex = 0; behindVertex < 3; ++behindVertex)
					{
						//these are ahead
						int prevVertex = behindVertex == 0 ? 2 : behindVertex - 1;
						int nextVertex = behindVertex == 2 ? 0 : behindVertex + 1;
						Mask16 caseMask = behindPlaneMasks[behindVertex] & ~behindPlaneMasks[prevVertex] & ~behindPlaneMasks[nextVertex];
						clipOutput[0] = VertexPack16::maskMove(clipOutput[0], output[nextVertex], caseMask);
						clipOutput[1] = VertexPack16::maskMove(clipOutput[1], output[prevVertex], caseMask);
						clipOutput[2] = VertexPack16::maskMove(clipOutput[2], VertexPack16::lerpToClippingZ(output[nextVertex], output[behindVertex], clippingZ), caseMask);
						clipOutput[3] = VertexPack16::maskMove(clipOutput[3], output[prevVertex], caseMask);
						clipOutput[4] = VertexPack16::maskMove(clipOutput[4], VertexPack16::lerpToClippingZ(output[prevVertex], output[behindVertex], clippingZ), caseMask);
						clipOutput[5] = VertexPack16::maskMove(clipOutput[5], VertexPack16::lerpToClippingZ(output[nextVertex], output[behindVertex], clippingZ), caseMask);
					}

					output = clipOutput;
				}

				//this stage needs to run again for new triangle created by clipping in 1 vertex behind plane case
				Mask16 oldActiveTriangles = activeTrianglesMask;
				int validOutputPacks = 3;
				for (int ouputStartIndex = 0; ouputStartIndex < 6; ouputStartIndex += 3)
				{
					if (ouputStartIndex == 3) //put it here since some of the continues may jump back to the beginning of the loop (like all triangles with 0 area)
					{
						if (behindPlaneCount == 1) //load new triangle if there is new
						{
							activeTrianglesMask = oldActiveTriangles & (behindPlaneCount == 1);
							validOutputPacks = 6;
						}
						else break;
					}

					float32x16 fovMult = 1; //TODO: adjustable game setting?
					float32x16 minX = INFINITY, maxX = -INFINITY, minY = INFINITY, maxY = -INFINITY;
					/*
					for (int j = 0; j < 3; ++j)
					{
						assert(bool(((float32x16(_mm512_abs_ps(transformedVertices[j].space.x)) > 5000.f) & activeTrianglesMask) == 0));
						assert(bool(((float32x16(_mm512_abs_ps(transformedVertices[j].space.y)) > 5000.f) & activeTrianglesMask) == 0));
						assert(bool(((float32x16(_mm512_abs_ps(transformedVertices[j].space.z)) > 5000.f) & activeTrianglesMask) == 0));
					}*/
					for (int i = 0; i < 3; ++i)
					{
						auto& currVertex = output[i + ouputStartIndex];
						float32x16 zInv = fovMult / currVertex.space.z;
						currVertex.u *= zInv;
						currVertex.v *= zInv;
						currVertex.normal.x *= zInv;
						currVertex.normal.y *= zInv;
						currVertex.normal.z *= zInv;

						for (int i = 0; i < 16; ++i)
						{

							/*for (int j = 0; j < 3; ++j)
							{
								//assert(0xFFFF == (float32x16(_mm512_abs_ps(transformedVertices[j].w)) < 5000.f));
								assert((float32x16(_mm512_abs_ps(transformedVertices[j].space.x)) < 5000.f).allOnes());
								assert((float32x16(_mm512_abs_ps(transformedVertices[j].space.y)) < 5000.f).allOnes());
								assert((float32x16(_mm512_abs_ps(transformedVertices[j].space.z)) < 5000.f).allOnes());
							}*/
						}
						currVertex.space = currCmd.ctr.screenSpaceToPixels(currVertex.space * zInv);
						currVertex.space.z = zInv;
						minX = _mm512_min_ps(minX, currVertex.space.x);
						minY = _mm512_min_ps(minY, currVertex.space.y);
						maxX = _mm512_max_ps(maxX, currVertex.space.x);
						maxY = _mm512_max_ps(maxY, currVertex.space.y);
					}

					const Vec4_f32x16& r1 = output[ouputStartIndex].space;
					const Vec4_f32x16& r2 = output[1 + ouputStartIndex].space;
					const Vec4_f32x16& r3 = output[2 + ouputStartIndex].space;
					//now transformedVertices hold screen coordinates (in pixels) and UVs are Z divided
					float32x16 signedArea = (r1 - r3).cross2d(r2 - r3);
					Mask16 nonZeroSignedAreaMask = signedArea != 0.f;
					activeTrianglesMask = (minX < w & maxX >= 0.f) & (minY < h & maxY >= 0.f) & (activeTrianglesMask & nonZeroSignedAreaMask);
					if (!activeTrianglesMask) continue;

					float32x16 rcpSignedArea = float32x16(1) / signedArea;

					minX = _mm512_floor_ps(minX);
					minY = _mm512_floor_ps(minY);
					maxX = _mm512_ceil_ps(maxX);
					maxY = _mm512_ceil_ps(maxY);
					for (int i = 0; i < 16; ++i)
					{
						if (!(activeTrianglesMask.mask & (1 << i))) continue;
						//assert(std::abs(maxX[i]) < 5000);
					}
					int32x16 vecFirstThread = _mm512_cvttps_epi32(minY * rcpScreenHeightPerThread);
					int32x16 vecLastThread = _mm512_cvttps_epi32(maxY * rcpScreenHeightPerThread);
					activeTrianglesMask &= (vecLastThread >= 0) & (vecFirstThread <= (threadCount - 1));

					//vecFirstThread = _mm512_mask_compress_epi32(int32x16(INT32_MAX), activeTrianglesMask, vecFirstThread);
					//vecLastThread = _mm512_mask_compress_epi32(int32x16(INT32_MIN), activeTrianglesMask, vecLastThread);
					vecFirstThread = _mm512_mask_mov_epi32(int32x16(INT32_MAX), activeTrianglesMask, vecFirstThread);
					vecLastThread = _mm512_mask_mov_epi32(int32x16(INT32_MIN), activeTrianglesMask, vecLastThread);
					int groupFirstThread = std::clamp(_mm512_reduce_min_epi32(vecFirstThread), 0, threadCount - 1);
					int groupLastThread = std::clamp(_mm512_reduce_max_epi32(vecLastThread), 0, threadCount - 1);

					int activeJobs = _mm_popcnt_u32(activeTrianglesMask);
					if (Statsman::ENABLED) MyStatsman.rendering.renderJobCountProducer += activeJobs;
					for (int currReceiverThread = groupFirstThread; currReceiverThread <= groupLastThread; ++currReceiverThread)
					{
						auto& targetStore = (*currCmd.outputStores)[threadIndex][currReceiverThread];
						Mask16 currMask = activeTrianglesMask & vecFirstThread <= currReceiverThread & vecLastThread >= currReceiverThread;
						targetStore.add(output.data() + ouputStartIndex, output.data() + ouputStartIndex + 3, rcpSignedArea, slice.modelIndex, currMask, currCmd);
					}
				}
			}

			//StatCount(threadIndex, trianges.ver)
			
		}
	}
	
	if (Statsman::ENABLED) {
		uint64_t ticksEnd = SDL_GetTicksNS();
		MyStatsman.time.transformMs = (ticksEnd - ticksBegin) / 1e6;
	}
}

__forceinline void calculateBarycentricCoordinates(const Vec4_f32x16& r, const Vec4_f32x16& r1, const Vec4_f32x16& r2, const Vec4_f32x16& r3, const float32x16& rcpSignedArea, float32x16& alpha, float32x16& beta, float32x16& gamma)
{
	alpha = (r - r3).cross2d(r2 - r3) * rcpSignedArea;
	beta = (r - r3).cross2d(r3 - r1) * rcpSignedArea;
	gamma = (r - r1).cross2d(r1 - r2) * rcpSignedArea; //do NOT change this to 1-alpha-beta or 1-(alpha+beta). That causes wonkiness in textures
}

void mask_store_vec4_f32x16_to_framebuffer(const Vec4_f32x16& pack, void* frameBuffer, int x, int y, int w, Mask16 mask)
{
	//we have px[0] == r0,r1,r2...,r15, px[1] == g0,..g15, ...
	//DX wants: r0,g0,b0,a0,r1,g1,b1,a1, etc
	//Meanings, that first 16-wide register to store should be r0,g0,b0,a0,...,r3,g3,b3,a3
	//Second - 4-7, third - 8-11, fourth - 12-15
	constexpr int DC = 0xDEADDEAD; //garbage value
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

#ifdef VS_CLANG
__m512i _mm512_setr_epi16(int16_t i0, int16_t i1, int16_t i2, int16_t i3, int16_t i4, int16_t i5, int16_t i6, int16_t i7, int16_t i8, int16_t i9, int16_t i10, int16_t i11, int16_t i12, int16_t i13, int16_t i14, int16_t i15, int16_t i16, int16_t i17, int16_t i18, int16_t i19, int16_t i20, int16_t i21, int16_t i22, int16_t i23, int16_t i24, int16_t i25, int16_t i26, int16_t i27, int16_t i28, int16_t i29, int16_t i30, int16_t i31)
{
	return _mm512_set_epi16(i31, i30, i29, i28, i27, i26, i25, i24, i23, i22, i21, i20, i19, i18, i17, i16, i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0);
}
#endif

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
void RasterizingRenderer::drawRenderJobs(int threadIndex)
{
	auto ticksBegin = SDL_GetTicksNS();
	int drawCmdInd = 0;
	for (auto& drawCmd : this->drawCommands)
	{
		auto [d_low, d_high] = this->currGs->threadpool->getLimitsForThread(threadIndex, 0, drawCmd.renderH);
		int threadCount = this->currGs->threadpool->getThreadCount();
		float my_yMin = floor(d_low);
		float my_yMax = std::min<float>(floor(d_high), drawCmd.renderH - 1);
		float my_xMin = 0;
		float my_xMax = drawCmd.renderW - 1;
		bool texturingEnabled = this->currGs->texturingEnabled;
		int w = drawCmd.renderW;
		for (int senderThreadIndex = 0; senderThreadIndex < threadCount; ++senderThreadIndex)
		{
			RenderJob_Store& storeForMe = (*drawCmd.outputStores)[senderThreadIndex][threadIndex];
			int jobsCountForMe = storeForMe.size();
			if (Statsman::ENABLED) MyStatsman.rendering.renderJobCountConsumer += jobsCountForMe;
			for (int myJobsPointerInt = 0; myJobsPointerInt < jobsCountForMe; myJobsPointerInt += 16)
			{
				Mask16 bounds = (int32x16::sequence() + myJobsPointerInt) < jobsCountForMe;
				float32x16 group_rcpSignedArea = _mm512_maskz_loadu_ps(bounds, storeForMe.rcpSignedArea.data() + myJobsPointerInt);
				int32x16 group_modelIndex = _mm512_maskz_loadu_epi32(bounds, storeForMe.modelIndex.data() + myJobsPointerInt);

				const std::array<VertexPack16, 3> vertices = storeForMe.loadVertices16(myJobsPointerInt, bounds);
				float32x16 group_xBeg = _mm512_max_ps(float32x16(my_xMin), _mm512_floor_ps(_mm512_min_ps(vertices[2].space.x, _mm512_min_ps(vertices[0].space.x, vertices[1].space.x))));
				float32x16 group_yBeg = _mm512_max_ps(float32x16(my_yMin), _mm512_floor_ps(_mm512_min_ps(vertices[2].space.y, _mm512_min_ps(vertices[0].space.y, vertices[1].space.y))));
				float32x16 group_xEnd = _mm512_min_ps(float32x16(my_xMax), _mm512_ceil_ps(_mm512_max_ps(vertices[2].space.x, _mm512_max_ps(vertices[0].space.x, vertices[1].space.x))));
				float32x16 group_yEnd = _mm512_min_ps(float32x16(my_yMax), _mm512_ceil_ps(_mm512_max_ps(vertices[2].space.y, _mm512_max_ps(vertices[0].space.y, vertices[1].space.y))));

				float32x16 group_initialAlpha, group_initialBeta, group_initialGamma;
				calculateBarycentricCoordinates({ group_xBeg, group_yBeg, 0.f,0.f }, vertices[0].space, vertices[1].space, vertices[2].space, group_rcpSignedArea, group_initialAlpha, group_initialBeta, group_initialGamma);
				float32x16 group_dAlpha_dx = (vertices[1].space.y - vertices[2].space.y) * group_rcpSignedArea;
				float32x16 group_dAlpha_dy = (vertices[2].space.x - vertices[1].space.x) * group_rcpSignedArea;
				float32x16 group_dBeta_dx = (vertices[2].space.y - vertices[0].space.y) * group_rcpSignedArea;
				float32x16 group_dBeta_dy = (vertices[0].space.x - vertices[2].space.x) * group_rcpSignedArea;
				float32x16 group_dGamma_dx = -group_dAlpha_dx - group_dBeta_dx; //TODO: replace with proper calculation, can cause precision issues! (actually, doesn't do now)
				float32x16 group_dGamma_dy = -group_dAlpha_dy - group_dBeta_dy; //and this too

				for (int i = 0; i < 16; ++i)
				{
					if ((bounds.mask & (1 << i)) == 0) continue;
					const auto& texture = this->textureManager.getTextureByHandle(this->sceneModels[group_modelIndex[i]].diffuseMapIndex);

					for (float y = group_yBeg[i]; y <= group_yEnd[i]; ++y)
					{
						size_t yInt = y;
						size_t xInt = group_xBeg[i];
						float32x16 dy = y - group_yBeg[i];
						for (float32x16 x = float32x16::sequence() + group_xBeg[i]; Mask16 xBoundsMask = (x <= group_xEnd[i]); x += 16, xInt += 16)
						{
							float32x16 dx = x - group_xBeg[i];
							float32x16 alpha = _mm512_fmadd_ps(_mm512_set1_ps(group_dAlpha_dy[i]), dy, _mm512_fmadd_ps(_mm512_set1_ps(group_dAlpha_dx[i]), dx, _mm512_set1_ps(group_initialAlpha[i])));
							float32x16 beta = _mm512_fmadd_ps(_mm512_set1_ps(group_dBeta_dy[i]), dy, _mm512_fmadd_ps(_mm512_set1_ps(group_dBeta_dx[i]), dx, _mm512_set1_ps(group_initialBeta[i])));
							float32x16 gamma = _mm512_fmadd_ps(_mm512_set1_ps(group_dGamma_dy[i]), dy, _mm512_fmadd_ps(_mm512_set1_ps(group_dGamma_dx[i]), dx, _mm512_set1_ps(group_initialGamma[i])));
							if (Statsman::ENABLED) MyStatsman.rendering.barycentricsCalculated += 16;

							Mask16 pointsInsideTriangleMask = (xBoundsMask & alpha >= 0.0) & (beta >= 0.0 & gamma >= 0.0);
							if (Statsman::ENABLED) MyStatsman.rendering.pointsInsideTriangles += _mm_popcnt_u32(pointsInsideTriangleMask.mask);
							if (!pointsInsideTriangleMask) continue;

							Vec4_f32x16 interpolatedDividedUv = Vec4_f32x16(vertices[0].u[i], vertices[0].v[i], vertices[0].space.z[i], 0.f) * alpha +
								Vec4_f32x16(vertices[1].u[i], vertices[1].v[i], vertices[1].space.z[i], 0.f) * beta +
								Vec4_f32x16(vertices[2].u[i], vertices[2].v[i], vertices[2].space.z[i], 0.f) * gamma;
							float32x16 currDepthValues = _mm512_maskz_loadu_ps(pointsInsideTriangleMask, drawCmd.zBuffer + yInt * w + xInt);
							if (Statsman::ENABLED)
							{
								MyStatsman.rendering.zBufferFetchLanes += 16;
								MyStatsman.rendering.zBufferFetchAliveLanes += _mm_popcnt_u32(pointsInsideTriangleMask.mask);
							}
							//depth test: bigger Z pre-divide = further. However, we have reciprocal Z stored in interpolatedDividedUv.z, and Z <= 1 are culled during clipping stage, thus 1/z < z at all times
							//example: Z post rotate and translate (but before divide) for 2 pixels are 2 and 3. After Z divide they become 0.5 and 0.333. 0.5 should win the depth test, since it's closer
							Mask16 notOccludedPoints = pointsInsideTriangleMask & currDepthValues < interpolatedDividedUv.z;
							if (Statsman::ENABLED) MyStatsman.rendering.notOccludedPoints += _mm_popcnt_u32(notOccludedPoints.mask);
							if (!notOccludedPoints) continue; //if all points are occluded, then skip

							Vec4_f32x16 uvCorrected = interpolatedDividedUv / interpolatedDividedUv.z;
							Vec4_f32x16 texturePixels;
							if (drawCmd.depthOnly)
							{
								/*
								auto accessor = texture.getGatherAccessor(uvCorrected.x, uvCorrected.y, notOccludedPoints);
								texturePixels.a = accessor.gatherA();*/
								texturePixels = texture.gatherLinearIntensities(uvCorrected.x, uvCorrected.y, notOccludedPoints);
							}
							else
							{
								if (texturingEnabled)
								{
									texturePixels = texture.gatherLinearIntensities(uvCorrected.x, uvCorrected.y, notOccludedPoints);
									if (drawCmd.shadingMode != ShadingMode::NONE)
									{
										Vec4_f32x16 interpolatedDividedNormals = Vec4_f32x16(vertices[0].normal.x[i], vertices[0].normal.y[i], vertices[0].normal.z[i], 0.f) * alpha +
											Vec4_f32x16(vertices[1].normal.x[i], vertices[1].normal.y[i], vertices[1].normal.z[i], 0.f) * beta +
											Vec4_f32x16(vertices[2].normal.x[i], vertices[2].normal.y[i], vertices[2].normal.z[i], 0.f) * gamma;
										Vec4_f32x16 correctedNormals = interpolatedDividedNormals / interpolatedDividedUv.z;
										correctedNormals /= correctedNormals.len3d();
										Vec4f lightFrom = { 13.978434,1933.787476,117.000008 }, lightTo = { -874.297729,136.884766,0.909166 };
										Vec4_f32x16 lightDir = lightTo - lightFrom;
										lightDir /= lightDir.len3d();
										float32x16 normalDot = -correctedNormals.dot3d(lightDir);
										texturePixels *= _mm512_max_ps(float32x16(0.05), normalDot);
									}
									if (Statsman::ENABLED)
									{
										MyStatsman.rendering.textureGatheredLanes += 16;
										MyStatsman.rendering.textureGatherAliveLanes += _mm_popcnt_u32(notOccludedPoints.mask);
									}
									texturePixels *= 3;
								}
								else
								{
									float32x16 dz = float32x16(1) / interpolatedDividedUv.z;
									float32x16 distIntensity = float32x16(1) - dz / (dz + 100.f);
									texturePixels.r = texturePixels.g = texturePixels.b = distIntensity;
									texturePixels.a = 1;
								}
							}
							Mask16 opaquePixelsMask = notOccludedPoints & (texturePixels.a > 0.0f);
							if (!opaquePixelsMask) continue;

							_mm512_mask_storeu_ps(drawCmd.zBuffer + yInt * w + xInt, opaquePixelsMask, interpolatedDividedUv.z);

							if (!drawCmd.depthOnly)
							{
								mask_store_vec4_f32x16_to_framebuffer(texturePixels, drawCmd.frameBuffer, xInt, yInt, w, opaquePixelsMask);
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
		drawCmdInd++;
	}
	if (Statsman::ENABLED) {
		auto ticksEnd = SDL_GetTicksNS();
		MyStatsman.time.drawMs = (ticksEnd - ticksBegin) / 1e6;
	}
}

void RasterizingRenderer::joinMainWithShadowMap(int threadIndex)
{
	auto [d_low, d_high] = this->currGs->threadpool->getLimitsForThread(threadIndex, 0, this->drawCommands[0].renderH);
	int threadCount = this->currGs->threadpool->getThreadCount();
	float my_yMin = floor(d_low);
	float my_yMax = std::min<float>(floor(d_high), this->drawCommands[0].renderH - 1);
	float my_xMin = 0;
	float my_xMax = this->drawCommands[0].renderW - 1;
	int w = this->drawCommands[0].renderW;
	for (float y = my_yMin; y < my_yMax; ++y)
	{
		size_t yInt = y;
		for (float32x16 x = float32x16::sequence() + my_xMin; Mask16 xBoundsMask = x <= my_xMax; x += 16)
		{
			size_t xInt = x[0];
			if (false) //debug draw shadow map to screen
			{
				float smw = this->drawCommands[1].renderW;
				float smh = this->drawCommands[1].renderH;
				Mask16 sm_xBounds = x < smw;
				if (!sm_xBounds) continue;
				if (y >= smh) continue;
				float32x16 z = _mm512_maskz_loadu_ps(sm_xBounds, this->drawCommands[1].zBuffer + yInt * this->drawCommands[1].renderW + xInt);
				float32x16 dz = float32x16(1) / z;
				float32x16 distIntensity = float32x16(1) - dz / (dz + 100.f);

				Vec4_f32x16 colOut;
				colOut.r = colOut.g = colOut.b = distIntensity;
				colOut.a = 1;
				mask_store_vec4_f32x16_to_framebuffer(colOut, this->drawCommands[0].frameBuffer, xInt, yInt, this->drawCommands[0].renderW, sm_xBounds);
				continue;
			}

			float32x16 zInvSrc = _mm512_maskz_loadu_ps(xBoundsMask, this->drawCommands[0].zBuffer + yInt * w + xInt);

			//float32x16 sy = y;
			//float32x16 sy = this->drawCommands[0].renderH - y;
			Vec4_f32x16 screenPos(x, y, 1, zInvSrc);
			Vec4_f32x16 worldCoords = this->drawCommands[0].ctr.inverseScreenPixelsToWorld(screenPos);
			const auto& currentShadowMap = this->drawCommands[1];
			Vec4_f32x16 sunWorldPositions = currentShadowMap.ctr.getCurrentTransformationMatrix() * worldCoords;
			float32x16 zInv = float32x16(1) / sunWorldPositions.z;
			Vec4_f32x16 sunScreenPositions = currentShadowMap.ctr.screenSpaceToPixels(sunWorldPositions * zInv);
			sunScreenPositions.z = zInv;
			//sunScreenPositions.y = float32x16(currentShadowMap.renderH) - sunScreenPositions.y;
			sunScreenPositions.y = sunScreenPositions.y;

			Mask16 inShadowMapBounds = xBoundsMask & (sunScreenPositions.x >= 0.f) & (sunScreenPositions.x < float(this->drawCommands[1].renderW)) & (sunScreenPositions.y >= 0.f) & (sunScreenPositions.y < float(this->drawCommands[1].renderH));
			int32x16 gatherInd = int32x16(sunScreenPositions.y.trunc()) * this->drawCommands[1].renderW + int32x16(sunScreenPositions.x.trunc());
			float32x16 shadowMapDepths = _mm512_mask_i32gather_ps(_mm512_set1_ps(INFINITY), inShadowMapBounds, gatherInd, this->drawCommands[1].zBuffer, 4);
			//float32x16 bias = 0.0001f;
			float32x16 bias = 0.f;
			Mask16 pointsInShadow = ~inShadowMapBounds | ((shadowMapDepths - bias) > sunScreenPositions.z);

			Vec4_f32x16 originalColors = mask_load_vec4_f32x16_from_framebuffer(this->drawCommands[0].frameBuffer, xInt, yInt, this->drawCommands[0].renderW, xBoundsMask);
			for (int i = 0; i < 3; ++i) originalColors[i] = _mm512_mask_mul_ps(originalColors[i], pointsInShadow, originalColors[i], float32x16(0.2));
			mask_store_vec4_f32x16_to_framebuffer(originalColors, this->drawCommands[0].frameBuffer, xInt, yInt, this->drawCommands[0].renderW, xBoundsMask);
		}
	}
}

uint32_t Rasterizing::Vertice_Store::insert(float x, float y, float z, float u, float v, float nx, float ny, float nz)
{
	assert(this->dedup.size() == this->x.size());
	auto t = std::make_tuple(x, y, z, u, v, nx, ny, nz);
	auto it = this->dedup.find(t);
	uint32_t ret;
	if (it == this->dedup.end())
	{
		this->dedup[t] = ret = this->dedup.size();
		this->x.push_back(x);
		this->y.push_back(y);
		this->z.push_back(z);
		this->u.push_back(u);
		this->v.push_back(v);
		this->nx.push_back(nx);
		this->ny.push_back(ny);
		this->nz.push_back(nz);
		return ret;
	}
	return it->second;
}


size_t Rasterizing::Vertice_Store::size() const
{
	return x.size();
}

size_t Rasterizing::Triangle_Store::size() const
{
	return vertInd[0].size();
}

std::array<VertexPack16,3> Rasterizing::RenderJob_Store::loadVertices16(size_t firstInd, Mask16 mask) const
{
	std::array<VertexPack16,3> ret;
	for (int i = 0; i < 3; ++i) 
	{
		ret[i].space.x = _mm512_maskz_loadu_ps(mask, this->x[i].data() + firstInd);
		ret[i].space.y = _mm512_maskz_loadu_ps(mask, this->y[i].data() + firstInd);
		ret[i].space.z = _mm512_maskz_loadu_ps(mask, this->z[i].data() + firstInd);
		ret[i].u = _mm512_maskz_loadu_ps(mask, this->u[i].data() + firstInd);
		ret[i].v = _mm512_maskz_loadu_ps(mask, this->v[i].data() + firstInd);
		ret[i].normal.x = _mm512_maskz_loadu_ps(mask, this->nx[i].data() + firstInd);
		ret[i].normal.y = _mm512_maskz_loadu_ps(mask, this->ny[i].data() + firstInd);
		ret[i].normal.z = _mm512_maskz_loadu_ps(mask, this->nz[i].data() + firstInd);
	}
	return ret;
}

size_t Rasterizing::RenderJob_Store::size() const
{
	return this->realSize;
}

void Rasterizing::RenderJob_Store::clear(bool forceClear)
{
	this->realSize = 0;
	if (forceClear)
	{
		for (int i = 0; i < 3; ++i)
		{
			x[i].clear();
			y[i].clear();
			z[i].clear();
			u[i].clear();
			v[i].clear();
			nx[i].clear();
			ny[i].clear();
			nz[i].clear();
		}
		modelIndex.clear();
		rcpSignedArea.clear();
		this->capacity = 0;
	}
}
void Rasterizing::RenderJob_Store::makeSpace(size_t newSize)
{
	if (newSize <= this->capacity) return;
	for (int i = 0; i < 3; ++i)
	{
		x[i].resize(newSize);
		y[i].resize(newSize);
		z[i].resize(newSize);
		u[i].resize(newSize);
		v[i].resize(newSize);
		nx[i].resize(newSize);
		ny[i].resize(newSize);
		nz[i].resize(newSize);
	}
	modelIndex.resize(newSize);
	rcpSignedArea.resize(newSize);
	this->capacity = newSize;
}
void Rasterizing::RenderJob_Store::add(const VertexPack16* pStart, const VertexPack16* pEnd, const float32x16& rcpSignedArea, const int32x16& modelIndex, Mask16 activeElementsMask, const DrawCommand& subInfo)
{
	if (pEnd - pStart != 3) throw std::runtime_error("Vertex pack sizes not equal to 3 are not yet supported in RenderJob_Store::add");
	//TODO: handle subInfo by eliding unused stores and resizes (normals/uvs/etc)
	if (!activeElementsMask) return;
	size_t oldSz = this->realSize;
	this->makeSpace(oldSz + 16);
	if (activeElementsMask == 0xFFFF)
	{
		for (int i = 0; i < 3; ++i)
		{
			_mm512_storeu_ps(this->x[i].data() + oldSz, pStart[i].space.x);
			_mm512_storeu_ps(this->y[i].data() + oldSz, pStart[i].space.y);
			_mm512_storeu_ps(this->z[i].data() + oldSz, pStart[i].space.z);
			_mm512_storeu_ps(this->u[i].data() + oldSz, pStart[i].u);
			_mm512_storeu_ps(this->v[i].data() + oldSz, pStart[i].v);
			_mm512_storeu_ps(this->nx[i].data() + oldSz, pStart[i].normal.x);
			_mm512_storeu_ps(this->ny[i].data() + oldSz, pStart[i].normal.y);
			_mm512_storeu_ps(this->nz[i].data() + oldSz, pStart[i].normal.z);
		}
		_mm512_storeu_epi32(this->modelIndex.data() + oldSz, modelIndex);
		_mm512_storeu_ps(this->rcpSignedArea.data() + oldSz, rcpSignedArea);
	}
	else
	{
		//since render job store resize does 16 overprovisioning on resize, it's OK not to mask these stores. Some architectures have awful compress store unaligned performance, so store compressed register
		for (int i = 0; i < 3; ++i)
		{
			_mm512_storeu_ps(this->x[i].data() + oldSz, _mm512_maskz_compress_ps(activeElementsMask, pStart[i].space.x));
			_mm512_storeu_ps(this->y[i].data() + oldSz, _mm512_maskz_compress_ps(activeElementsMask, pStart[i].space.y));
			_mm512_storeu_ps(this->z[i].data() + oldSz, _mm512_maskz_compress_ps(activeElementsMask, pStart[i].space.z));
			_mm512_storeu_ps(this->u[i].data() + oldSz, _mm512_maskz_compress_ps(activeElementsMask, pStart[i].u));
			_mm512_storeu_ps(this->v[i].data() + oldSz, _mm512_maskz_compress_ps(activeElementsMask, pStart[i].v));
			_mm512_storeu_ps(this->nx[i].data() + oldSz, _mm512_maskz_compress_ps(activeElementsMask, pStart[i].normal.x));
			_mm512_storeu_ps(this->ny[i].data() + oldSz, _mm512_maskz_compress_ps(activeElementsMask, pStart[i].normal.y));
			_mm512_storeu_ps(this->nz[i].data() + oldSz, _mm512_maskz_compress_ps(activeElementsMask, pStart[i].normal.z));
		}
		_mm512_storeu_epi32(this->modelIndex.data() + oldSz, _mm512_maskz_compress_epi32(activeElementsMask, modelIndex));
		_mm512_storeu_ps(this->rcpSignedArea.data() + oldSz, _mm512_maskz_compress_ps(activeElementsMask, rcpSignedArea));
	}
	this->realSize += _mm_popcnt_u32(activeElementsMask);
}
