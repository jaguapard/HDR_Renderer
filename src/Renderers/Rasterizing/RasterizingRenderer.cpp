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
			uint32_t vertInd = this->original_verticeStore.insert(vertices[k].x, vertices[k].y, vertices[k].z, 0.5, 0.5);
			m.triangleStore.vertInd[k].push_back(vertInd);
			m.diffuseMapIndex = 0;
		}
		return;
	}

	std::mutex mtx;
	std::vector<task_id> tasks;
	for (auto& [path, mode] : scd.files)
	{
		AssetLoader ldr;
		std::vector<AssetLoader::ImportedModel> loadedModels;
		if (mode == "obj") { loadedModels = ldr.loadObj(path, "H:\\Sponza goodies\\Old Sponza 2026.bmdl"); }
		else if (mode == "bmdl") { loadedModels = ldr.loadBmdl(path); }
		else throw std::runtime_error("Unsupported mode for RasterizingRenderer::loadScene: " + mode);

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
				Vec4f v1 = { it.vertices[0][0], it.vertices[0][1], it.vertices[0][2], 0 };
				Vec4f v2 = { it.vertices[1][0], it.vertices[1][1], it.vertices[1][2], 0 };
				Vec4f v3 = { it.vertices[2][0], it.vertices[2][1], it.vertices[2][2], 0 };
				Vec4f dv21 = v2 - v1;
				Vec4f dv32 = v3 - v2;
				if (dv21.lenSq() * dv32.lenSq() == 0) //degenerate triangles
				{
					++discardedTriangles; continue;
				}
				for (int k = 0; k < 3; ++k)
				{
					uint32_t vertInd = this->original_verticeStore.insert(
						it.vertices[k][0], it.vertices[k][1], it.vertices[k][2], it.u[k], it.v[k]
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
		}
	}

	Threadpool::instance->waitForMultipleTasks(tasks);
	Uint64 ticksEnd = SDL_GetTicksNS();
	std::cout << "Scene loaded in " << (ticksEnd - ticksBegin) / 1e9 << " sec.\n";
}

void RasterizingRenderer::renderFrame(const GameSettings& settings)
{
	this->currGs = &settings;
	this->modelSlicesForThreads = this->makeModelSliceList();
	this->zBuffer.resize(settings.outputTextureParams.Width * settings.outputTextureParams.Height);
	this->ctr = { (int)settings.outputTextureParams.Width, (int)settings.outputTextureParams.Height };
	this->ctr.prepare(settings.camPos, settings.camAng);
	
	Threadpool* threadpool = settings.threadpool;
	int threadCount = threadpool->getThreadCount();
	this->renderJobsFromThreads.resize(threadCount);
	this->renderJobForwardNetwork.resize(threadCount);

	std::vector<task_id> transformTasks, drawTasks;
	for (int threadIndex = 0; threadIndex < threadCount; ++threadIndex)
	{
		transformTasks.emplace_back(threadpool->addTask(
			[&, threadIndex]() {
				this->doTransformationsAndClipping(threadIndex);
			}
		));
	}
	for (auto& it : zBuffer) it = -INFINITY;
	int sz = settings.outputTextureParams.Width * settings.outputTextureParams.Height * 4;
	float* pp = (float*)(settings.graphicsOutputBuffer);
	for (int i = 0; i < sz; ++i) pp[i] = 0;
	threadpool->waitForMultipleTasks(transformTasks);

	size_t renderJobCount = 0;
	for (auto& it : this->renderJobsFromThreads) renderJobCount += it.size();
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
	for (auto& it : renderJobsFromThreads) it.clear();
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

struct VertexPack16
{
	Vec4_f32x16 space;
	float32x16 u, v;
	//VertexPack16(float32x16 x, )
	VertexPack16() {};
	static __forceinline VertexPack16 lerpVertices(const VertexPack16& from, const VertexPack16& to, const float32x16& alpha)
	{
		VertexPack16 ret;
		ret.space.x = lerp(from.space.x, to.space.x, alpha);
		ret.space.y = lerp(from.space.y, to.space.y, alpha);
		ret.space.z = lerp(from.space.z, to.space.z, alpha);
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
	this->renderJobForwardNetwork[threadIndex].clear();
	assert(this->original_verticeStore.x.size() == this->original_verticeStore.y.size() && this->original_verticeStore.y.size() == this->original_verticeStore.z.size() && this->original_verticeStore.u.size() == this->original_verticeStore.v.size());

	float rcpScreenHeightPerThread = double(this->currGs->threadpool->getThreadCount()) / this->currGs->outputTextureParams.Height;
	float32x16 clippingZ = this->currGs->cameraPlane_zDist;
	size_t rjRealSize = 0;
	size_t seenTris = 0;
	for (auto& slice : this->modelSlicesForThreads[threadIndex])
	{
		//const float* xData = 
		seenTris += slice.modelTriangleIndexEnd - slice.modelTriangleIndexBegin;
		int myRenderJobCount = 0;
		for (int currModelTriangleIndex = slice.modelTriangleIndexBegin;
			currModelTriangleIndex < slice.modelTriangleIndexEnd; 
			currModelTriangleIndex += 16)
		{
			int32x16 triangleIndices = int32x16::sequence() + currModelTriangleIndex;
			Mask16 inBoundsTrianglesMask = triangleIndices < slice.modelTriangleIndexEnd;
			VertexPack16 transformedVertices[3];

			int32x16 behindPlaneCount = 0;
			Mask16 behindPlaneMasks[3];
			int32x16 verticeIndicesCache[3];
			const Model& model = sceneModels[slice.modelIndex];
			for (int i = 0; i < 3; ++i)
			{
				int32x16 verticeIndices = _mm512_maskz_loadu_epi32(inBoundsTrianglesMask, model.triangleStore.vertInd[i].data() + currModelTriangleIndex);
				float32x16 x = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), inBoundsTrianglesMask, verticeIndices, this->original_verticeStore.x.data(), 4);
				float32x16 y = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), inBoundsTrianglesMask, verticeIndices, this->original_verticeStore.y.data(), 4);
				float32x16 z = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), inBoundsTrianglesMask, verticeIndices, this->original_verticeStore.z.data(), 4);
				float32x16 w = 1;

				Vec4_f32x16 rotatedTranslated = this->ctr.rotateAndTranslate(Vec4_f32x16(x, y, z, w));
				Mask16 vertexBehindClippingPlane = rotatedTranslated.z < clippingZ;
				behindPlaneMasks[i] = vertexBehindClippingPlane;
				behindPlaneCount = _mm512_mask_add_epi32(behindPlaneCount, vertexBehindClippingPlane, behindPlaneCount, int32x16(1));
				transformedVertices[i].space = rotatedTranslated;
				verticeIndicesCache[i] = verticeIndices;
			}
			//StatCount(threadIndex, trianges.ver)
			if (Statsman::ENABLED)
			{
				for (int i = 0; i <= 3; ++i)
				{
					Statsman::statsmenForThreads[threadIndex].triangles.verticesBehindNearPlane[i] += _mm512_mask_reduce_add_epi32(behindPlaneCount == i, _mm512_set1_epi32(1));
				}
			}
			if (!(behindPlaneCount != 3)) continue; //if all triangles have all vertices behind clipping plane, skip them

			Mask16 activeTrianglesMask = inBoundsTrianglesMask & behindPlaneCount != 3;

			/* 
			if (this->currGs.backfaceCullingEnabled && !this->sceneModels[slice.modelIndex].noBackfaceCulling)
			{
				Vec4_f32x16 normal = (transformedVertices[2].space - transformedVertices[0].space).cross3d(transformedVertices[1].space - transformedVertices[0].space);
				activeTrianglesMask &= transformedVertices[0].space.dot3d(normal) > 0.f;
			}*/
			
			for (int i = 0; i < 3; ++i)
			{
				transformedVertices[i].u = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), activeTrianglesMask, verticeIndicesCache[i], this->original_verticeStore.u.data(), 4);
				transformedVertices[i].v = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), activeTrianglesMask, verticeIndicesCache[i], this->original_verticeStore.v.data(), 4);
			}

			VertexPack16 clipOutput[6];
			if (behindPlaneCount > 0)
			{
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

				//TODO: this network can be adjusted to preserve input windings, it currently doesn't but we don't care for now.

				for (int i = 0; i < 3; ++i) clipOutput[i] = transformedVertices[i];
				for (int frontVertex = 0; frontVertex < 3; ++frontVertex)
				{
					//these are behind
					int prevVertex = frontVertex == 0 ? 2 : frontVertex - 1;
					int nextVertex = frontVertex == 2 ? 0 : frontVertex + 1;
					Mask16 caseMask = ~behindPlaneMasks[frontVertex] & behindPlaneMasks[prevVertex] & behindPlaneMasks[nextVertex];
					clipOutput[0] = VertexPack16::maskMove(clipOutput[0], VertexPack16::lerpToClippingZ(transformedVertices[prevVertex], transformedVertices[frontVertex], clippingZ), caseMask);
					clipOutput[1] = VertexPack16::maskMove(clipOutput[1], transformedVertices[frontVertex], caseMask);
					clipOutput[2] = VertexPack16::maskMove(clipOutput[2], VertexPack16::lerpToClippingZ(transformedVertices[frontVertex], transformedVertices[nextVertex], clippingZ), caseMask);
				}

				for (int behindVertex = 0; behindVertex < 3; ++behindVertex)
				{
					//these are ahead
					int prevVertex = behindVertex == 0 ? 2 : behindVertex - 1;
					int nextVertex = behindVertex == 2 ? 0 : behindVertex + 1;
					Mask16 caseMask = behindPlaneMasks[behindVertex] & ~behindPlaneMasks[prevVertex] & ~behindPlaneMasks[nextVertex];
					clipOutput[0] = VertexPack16::maskMove(clipOutput[0], transformedVertices[nextVertex], caseMask);
					clipOutput[1] = VertexPack16::maskMove(clipOutput[1], transformedVertices[prevVertex], caseMask);
					clipOutput[2] = VertexPack16::maskMove(clipOutput[2], VertexPack16::lerpToClippingZ(transformedVertices[nextVertex], transformedVertices[behindVertex], clippingZ), caseMask);
					clipOutput[3] = VertexPack16::maskMove(clipOutput[3], transformedVertices[prevVertex], caseMask);
					clipOutput[4] = VertexPack16::maskMove(clipOutput[4], VertexPack16::lerpToClippingZ(transformedVertices[prevVertex], transformedVertices[behindVertex], clippingZ), caseMask);
					clipOutput[5] = VertexPack16::maskMove(clipOutput[5], VertexPack16::lerpToClippingZ(transformedVertices[nextVertex], transformedVertices[behindVertex], clippingZ), caseMask);
				}

				for (int k = 0; k < 3; ++k) transformedVertices[k] = clipOutput[k];
			}

			//this stage needs to run again for new triangle created by clipping in 1 vertex behind plane case
			Mask16 oldActiveTriangles = activeTrianglesMask;
			for (int stageTrianglesProcessed = 0; stageTrianglesProcessed < 2; ++stageTrianglesProcessed)
			{
				if (stageTrianglesProcessed == 1) //put it here since some of the continues may jump back to the beginning of the loop (like all triangles with 0 area)
				{
					if (behindPlaneCount == 1) //load new triangle if there is new
					{
						for (int i = 0; i < 3; ++i)
						{
							transformedVertices[i] = clipOutput[i+3];
						}
						activeTrianglesMask = oldActiveTriangles & (behindPlaneCount == 1);
					}
					else break;
				}

				float32x16 fovMult = 1;
				float32x16 minX = INFINITY, maxX = -INFINITY, minY = INFINITY, maxY = -INFINITY;
				for (int j = 0; j < 3; ++j)
				{
					assert(bool(((float32x16(_mm512_abs_ps(transformedVertices[j].space.x)) > 5000.f) & activeTrianglesMask) == 0));
					assert(bool(((float32x16(_mm512_abs_ps(transformedVertices[j].space.y)) > 5000.f) & activeTrianglesMask) == 0));
					assert(bool(((float32x16(_mm512_abs_ps(transformedVertices[j].space.z)) > 5000.f) & activeTrianglesMask) == 0));
				}
				for (int i = 0; i < 3; ++i)
				{
					float32x16 zInv = fovMult / transformedVertices[i].space.z;
					transformedVertices[i].u *= zInv;
					transformedVertices[i].v *= zInv;
					
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
					Vec4_f32x16 screenSpaceVertice = this->ctr.screenSpaceToPixels(transformedVertices[i].space * zInv);
					screenSpaceVertice.z = zInv;
					transformedVertices[i].space = screenSpaceVertice;
					minX = _mm512_min_ps(minX, screenSpaceVertice.x);
					minY = _mm512_min_ps(minY, screenSpaceVertice.y);
					maxX = _mm512_max_ps(maxX, screenSpaceVertice.x);
					maxY = _mm512_max_ps(maxY, screenSpaceVertice.y);
				}

				const Vec4_f32x16& r1 = transformedVertices[0].space;
				const Vec4_f32x16& r2 = transformedVertices[1].space;
				const Vec4_f32x16& r3 = transformedVertices[2].space;
				//now transformedVertices hold screen coordinates (in pixels) and UVs are Z divided
				float32x16 signedArea = (r1 - r3).cross2d(r2 - r3);
				Mask16 zeroSignedAreaMask = signedArea == 0.f;
				activeTrianglesMask &= ~zeroSignedAreaMask;
				if (!activeTrianglesMask) continue;

				float32x16 rcpSignedArea = float32x16(1) / signedArea;
				int jobsToAdd = std::popcount(__mmask16(activeTrianglesMask));
				auto& myRjStore = this->renderJobsFromThreads[threadIndex];
				myRjStore.resize(rjRealSize + jobsToAdd);

				minX = _mm512_floor_ps(minX);
				minY = _mm512_floor_ps(minY);
				maxX = _mm512_ceil_ps(maxX);
				maxY = _mm512_ceil_ps(maxY);
				for (int i = 0; i < 16; ++i)
				{
					if (!(activeTrianglesMask.mask & (1 << i))) continue;
					//assert(std::abs(maxX[i]) < 5000);
				}
				int32x16 firstThread = _mm512_cvttps_epi32(minY * rcpScreenHeightPerThread);
				int32x16 lastThread = _mm512_cvttps_epi32(maxY * rcpScreenHeightPerThread);
				//since render job store resize does 16 overprovisioning on resize, it's OK not to mask these stores. Some architectures have awful compress store unaligned performance
				for (int i = 0; i < 3; ++i)
				{
					_mm512_storeu_ps(myRjStore.x[i].data() + rjRealSize, _mm512_maskz_compress_ps(activeTrianglesMask, transformedVertices[i].space.x));
					_mm512_storeu_ps(myRjStore.y[i].data() + rjRealSize, _mm512_maskz_compress_ps(activeTrianglesMask, transformedVertices[i].space.y));
					_mm512_storeu_ps(myRjStore.z[i].data() + rjRealSize, _mm512_maskz_compress_ps(activeTrianglesMask, transformedVertices[i].space.z));
					_mm512_storeu_ps(myRjStore.u[i].data() + rjRealSize, _mm512_maskz_compress_ps(activeTrianglesMask, transformedVertices[i].u));
					_mm512_storeu_ps(myRjStore.v[i].data() + rjRealSize, _mm512_maskz_compress_ps(activeTrianglesMask, transformedVertices[i].v));
				}
				_mm512_storeu_epi32(myRjStore.modelIndex.data() + rjRealSize, _mm512_maskz_compress_epi32(activeTrianglesMask, int32x16(slice.modelIndex)));
				_mm512_storeu_epi32(myRjStore.firstThread.data() + rjRealSize, _mm512_maskz_compress_epi32(activeTrianglesMask, firstThread));
				_mm512_storeu_epi32(myRjStore.lastThread.data() + rjRealSize, _mm512_maskz_compress_epi32(activeTrianglesMask, lastThread));
				_mm512_storeu_ps(myRjStore.minX.data() + rjRealSize, _mm512_maskz_compress_ps(activeTrianglesMask, minX));
				_mm512_storeu_ps(myRjStore.maxX.data() + rjRealSize, _mm512_maskz_compress_ps(activeTrianglesMask, maxX));
				_mm512_storeu_ps(myRjStore.minY.data() + rjRealSize, _mm512_maskz_compress_ps(activeTrianglesMask, minY));
				_mm512_storeu_ps(myRjStore.maxY.data() + rjRealSize, _mm512_maskz_compress_ps(activeTrianglesMask, maxY));
				_mm512_storeu_ps(myRjStore.rcpSignedArea.data() + rjRealSize, _mm512_maskz_compress_ps(activeTrianglesMask, rcpSignedArea));

				for (int i = 0; i < jobsToAdd; ++i)
				{
					int ind = rjRealSize + i;
					assert(myRjStore.rcpSignedArea[ind] != 0);
				}
				rjRealSize += jobsToAdd;
			}
		}
	}

	/*
	size_t trisInSlices = 0;
	for (auto& m : )
	{
		for ()
	}
	assert(seenTris == trisInSlices);*/
	this->renderJobsFromThreads[threadIndex].resize(rjRealSize, false);

	int maxThread = this->currGs->threadpool->getThreadCount() - 1;
	this->renderJobForwardNetwork[threadIndex].resize(maxThread+1);
	for (int i = 0; i < rjRealSize; ++i)
	{
		int firstThread = this->renderJobsFromThreads[threadIndex].firstThread[i];
		int lastThread = this->renderJobsFromThreads[threadIndex].lastThread[i];
		if (firstThread > maxThread || lastThread < 0) continue;
		firstThread = std::clamp(firstThread, 0, maxThread);
		lastThread = std::clamp(lastThread, 0, maxThread);
		for (int j = firstThread; j <= lastThread; ++j)
		{
			this->renderJobForwardNetwork[threadIndex][j].push_back(i);
		}
	}
	uint64_t ticksEnd = SDL_GetTicksNS();
	if (Statsman::ENABLED) MyStatsman.time.transformMs = (ticksEnd-ticksBegin)/1e6;
}

__forceinline void calculateBarycentricCoordinates(const Vec4_f32x16& r, const Vec4_f32x16& r1, const Vec4_f32x16& r2, const Vec4_f32x16& r3, const float32x16& rcpSignedArea, float32x16& alpha, float32x16& beta, float32x16& gamma)
{
	alpha = (r - r3).cross2d(r2 - r3) * rcpSignedArea;
	beta = (r - r3).cross2d(r3 - r1) * rcpSignedArea;
	gamma = (r - r1).cross2d(r1 - r2) * rcpSignedArea; //do NOT change this to 1-alpha-beta or 1-(alpha+beta). That causes wonkiness in textures
}

void RasterizingRenderer::drawRenderJobs(int threadIndex)
{
	auto ticksBegin = SDL_GetTicksNS();
	auto [d_low, d_high] = this->currGs->threadpool->getLimitsForThread(threadIndex, 0, this->currGs->outputTextureParams.Height);
	int threadCount = this->currGs->threadpool->getThreadCount();
	float my_yMin = floor(d_low);
	float my_yMax = std::min<float>(floor(d_high), this->currGs->outputTextureParams.Height - 1);
	float my_xMin = 0;
	float my_xMax = this->currGs->outputTextureParams.Width - 1;
	int w = this->currGs->outputTextureParams.Width;

	for (int senderThreadIndex = 0; senderThreadIndex < threadCount; ++senderThreadIndex)
	{
		const RenderJob_Store& creatorThreadJobStore = this->renderJobsFromThreads[senderThreadIndex];
		int jobsCountForMe = this->renderJobForwardNetwork[senderThreadIndex][threadIndex].size();
		int* pData = this->renderJobForwardNetwork[senderThreadIndex][threadIndex].data();
		for (int myJobsPointerInt = 0; myJobsPointerInt < jobsCountForMe; myJobsPointerInt += 16)
		{
			Mask16 bounds = (int32x16::sequence() + myJobsPointerInt) < jobsCountForMe;
			int32x16 jobIndices = _mm512_maskz_loadu_epi32(bounds, pData + myJobsPointerInt);
			float32x16 group_xBeg = _mm512_max_ps(float32x16(my_xMin), _mm512_mask_i32gather_ps(_mm512_set1_ps(0), bounds, jobIndices, creatorThreadJobStore.minX.data(), 4));
			float32x16 group_xEnd = _mm512_min_ps(float32x16(my_xMax), _mm512_mask_i32gather_ps(_mm512_set1_ps(0), bounds, jobIndices, creatorThreadJobStore.maxX.data(), 4));
			float32x16 group_yBeg = _mm512_max_ps(float32x16(my_yMin), _mm512_mask_i32gather_ps(_mm512_set1_ps(0), bounds, jobIndices, creatorThreadJobStore.minY.data(), 4));
			float32x16 group_yEnd = _mm512_min_ps(float32x16(my_yMax), _mm512_mask_i32gather_ps(_mm512_set1_ps(0), bounds, jobIndices, creatorThreadJobStore.maxY.data(), 4));
			float32x16 group_rcpSignedArea = _mm512_mask_i32gather_ps(_mm512_set1_ps(0), bounds, jobIndices, creatorThreadJobStore.rcpSignedArea.data(), 4);
			int32x16 group_modelIndex = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(0), bounds, jobIndices, creatorThreadJobStore.modelIndex.data(), 4);
			VertexPack16 group_verts[3];

			for (int i = 0; i < 3; ++i)
			{
				group_verts[i].space.x = _mm512_mask_i32gather_ps(_mm512_set1_ps(0), bounds, jobIndices, creatorThreadJobStore.x[i].data(), 4);
				group_verts[i].space.y = _mm512_mask_i32gather_ps(_mm512_set1_ps(0), bounds, jobIndices, creatorThreadJobStore.y[i].data(), 4);
				group_verts[i].space.z = _mm512_mask_i32gather_ps(_mm512_set1_ps(0), bounds, jobIndices, creatorThreadJobStore.z[i].data(), 4);
				group_verts[i].u = _mm512_mask_i32gather_ps(_mm512_set1_ps(0), bounds, jobIndices, creatorThreadJobStore.u[i].data(), 4);
				group_verts[i].v = _mm512_mask_i32gather_ps(_mm512_set1_ps(0), bounds, jobIndices, creatorThreadJobStore.v[i].data(), 4);
			}
			for (int i = 0; i < 16; ++i)
			{
				if ((bounds.mask & (1 << i)) == 0) continue;
				int32x16 permInd = i;
				float32x16 xBeg = _mm512_permutexvar_ps(permInd, group_xBeg);
				float32x16 xEnd = _mm512_permutexvar_ps(permInd, group_xEnd);
				float32x16 yBeg = _mm512_permutexvar_ps(permInd, group_yBeg);
				float32x16 yEnd = _mm512_permutexvar_ps(permInd, group_yEnd);
				float32x16 rcpSignedArea = _mm512_permutexvar_ps(permInd, group_rcpSignedArea);
				const auto& texture = this->textureManager.getTextureByHandle(this->sceneModels[group_modelIndex[i]].diffuseMapIndex);
				VertexPack16 verts[3];
				for (int j = 0; j < 3; ++j)
				{
					verts[j].space.x = _mm512_permutexvar_ps(permInd, group_verts[j].space.x);
					verts[j].space.y = _mm512_permutexvar_ps(permInd, group_verts[j].space.y);
					verts[j].space.z = _mm512_permutexvar_ps(permInd, group_verts[j].space.z);
					verts[j].u = _mm512_permutexvar_ps(permInd, group_verts[j].u);
					verts[j].v = _mm512_permutexvar_ps(permInd, group_verts[j].v);
				}
				
				for (float y = yBeg[0]; y <= yEnd[0]; ++y)
				{
					size_t yInt = y;
					size_t xInt = xBeg[0];
					for (float32x16 x = float32x16::sequence() + xBeg; Mask16 xBoundsMask = (x <= xEnd); x += 16, xInt += 16)
					{
						Vec4_f32x16 r = Vec4_f32x16(x, y, 0.0, 0.0);
						float32x16 alpha, beta, gamma;
						calculateBarycentricCoordinates(r, verts[0].space, verts[1].space, verts[2].space, rcpSignedArea, alpha, beta, gamma);
						if (Statsman::ENABLED) MyStatsman.rendering.barycentricsCalculated += 16;

						Mask16 pointsInsideTriangleMask = xBoundsMask & (alpha >= 0.0) & (beta >= 0.0) & (gamma >= 0.0);
						if (Statsman::ENABLED) MyStatsman.rendering.pointsInsideTriangles += std::popcount(pointsInsideTriangleMask.mask);
						if (!pointsInsideTriangleMask) continue;

						Vec4_f32x16 interpolatedDividedUv = 
							Vec4_f32x16(verts[0].u, verts[0].v, verts[0].space.z, 0.f) * alpha + 
							Vec4_f32x16(verts[1].u, verts[1].v, verts[1].space.z, 0.f) * beta +
							Vec4_f32x16(verts[2].u, verts[2].v, verts[2].space.z, 0.f) * gamma;

						//float32x16 currDepthValues = this->zBuffer.getPixels16(xInt, yInt);
						float32x16 currDepthValues = _mm512_maskz_loadu_ps(pointsInsideTriangleMask, this->zBuffer.data() + yInt * w + xInt);
						if (Statsman::ENABLED)
						{
							MyStatsman.rendering.zBufferFetchLanes += 16;
							MyStatsman.rendering.zBufferFetchAliveLanes += std::popcount(pointsInsideTriangleMask.mask);
						}
						//depth test: bigger Z pre-divide = further. However, we have reciprocal Z stored in interpolatedDividedUv.z, and Z <= 1 are culled during clipping stage, thus 1/z < z at all times
						//example: Z post rotate and translate (but before divide) for 2 pixels are 2 and 3. After Z divide they become 0.5 and 0.333. 0.5 should win the depth test, since it's closer
						Mask16 visiblePointsMask = pointsInsideTriangleMask & currDepthValues < interpolatedDividedUv.z;
						if (Statsman::ENABLED) MyStatsman.rendering.visiblePoints += std::popcount(visiblePointsMask.mask);
						if (!visiblePointsMask) continue; //if all points are occluded, then skip

						Vec4_f32x16 uvCorrected = interpolatedDividedUv / interpolatedDividedUv.z;
						//Vec4_f32x16 texturePixels = texture.gatherPixels512(uvCorrected.x, uvCorrected.y, visiblePointsMask);
						Vec4_f32x16 texturePixels = texture.gatherLinearIntensities(uvCorrected.x, uvCorrected.y, visiblePointsMask);
						Mask16 opaquePixelsMask = visiblePointsMask & (texturePixels.a > 0.0f);
						if (Statsman::ENABLED)
						{
							MyStatsman.rendering.opaquePixels += std::popcount(opaquePixelsMask.mask);
							MyStatsman.rendering.textureGatheredLanes += 16;
							MyStatsman.rendering.textureGatherAliveLanes += std::popcount(visiblePointsMask.mask);
						}
						if (!opaquePixelsMask) continue;

						_mm512_mask_storeu_ps(this->zBuffer.data() + yInt * w + xInt, opaquePixelsMask, interpolatedDividedUv.z);
						/*float32x16 dz = float32x16(1) / interpolatedDividedUv.z;
						float32x16 distIntensity = float32x16(1) - dz / (dz + 100.f);
						texturePixels.r = texturePixels.g = texturePixels.b = distIntensity;*/

						//we have px[0] == r0,r1,r2...,r15, px[1] == g0,..g15, ...
						//DX wants: r0,g0,b0,a0,r1,g1,b1,a1, etc
						//Meanings, that first 16-wide register to store should be r0,g0,b0,a0,...,r3,g3,b3,a3
						//Second - 4-7, third - 8-11, fourth - 12-15
						constexpr int DC = 0xDEADDEAD; //garbage value
						//duplicate each opaquePixelsMask bit 4 times, i.e: 0123 -> 0000111122223333, 16 bits -> 64
						__m512i expanded = _mm512_maskz_mov_epi32(opaquePixelsMask, _mm512_set1_epi32(-1));
						__mmask64 duplicated = _mm512_cmpneq_epi8_mask(expanded, _mm512_set1_epi32(0));
						for (int i = 0; i < 16; i += 4)
						{
							__m512i rg_ind = _mm512_add_epi32(_mm512_set1_epi32(i), _mm512_setr_epi32(0, 16, DC, DC, 1, 17, DC, DC, 2, 18, DC, DC, 3, 19, DC, DC));
							__m512i ba_ind = _mm512_add_epi32(_mm512_set1_epi32(i), _mm512_setr_epi32(DC, DC, 0, 16, DC, DC, 1, 17, DC, DC, 2, 18, DC, DC, 3, 19));
							//__m512i gr_ind = _mm512_add_epi32(_mm512_set1_epi32(i), _mm512_setr_epi32(DC, DC, 0, 16, DC, DC, 1, 17, DC, DC, 2, 18, DC, DC, 3, 19));
							//__m512i ab_ind = _mm512_add_epi32(_mm512_set1_epi32(i), _mm512_setr_epi32(0, 16, DC, DC, 1, 17, DC, DC, 2, 18, DC, DC, 3, 19, DC, DC));
							float32x16 rgxx = _mm512_permutex2var_ps(texturePixels.x, rg_ind, texturePixels.y);
							float32x16 xxba = _mm512_permutex2var_ps(texturePixels.z, ba_ind, _mm512_set1_ps(1));
							float32x16 rgba = _mm512_mask_mov_ps(rgxx, 0b1100110011001100, xxba);
							int storeInd = (yInt * w + xInt + i) * 4;
							_mm512_mask_storeu_ps((float*)this->currGs->graphicsOutputBuffer + storeInd, duplicated >> (i * 4), rgba);
						}

						if (Statsman::ENABLED)
						{
							MyStatsman.rendering.zBufferWriteLanes += 16;
							MyStatsman.rendering.zBufferWriteAliveLanes += std::popcount(opaquePixelsMask.mask);
							MyStatsman.rendering.frameBufWriteLanes += 16;
							MyStatsman.rendering.frameBufWriteAliveLanes += std::popcount(opaquePixelsMask.mask);
						}
					}
				}
			}
		}
		
	
	}
	auto ticksEnd = SDL_GetTicksNS();
	if (Statsman::ENABLED) MyStatsman.time.drawMs = (ticksEnd - ticksBegin) / 1e6;
}

uint32_t Rasterizing::Vertice_Store::insert(float x, float y, float z, float u, float v)
{
	assert(this->dedup.size() == this->x.size());
	auto t = std::make_tuple(x, y, z, u, v);
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

size_t Rasterizing::RenderJob_Store::size() const
{
	return modelIndex.size();
}

void Rasterizing::RenderJob_Store::clear()
{
	for (int i = 0; i < 3; ++i)
	{
		x[i].clear();
		y[i].clear();
		z[i].clear();
		u[i].clear();
		v[i].clear();
	}
	minX.clear();
	minY.clear();
	maxX.clear();
	maxY.clear();
	modelIndex.clear();
	firstThread.clear();
	lastThread.clear();
	rcpSignedArea.clear();
}
void Rasterizing::RenderJob_Store::resize(size_t ind, bool overprovision)
{
	if (overprovision) ind += 16;
	for (int i = 0; i < 3; ++i)
	{
		x[i].resize(ind);
		y[i].resize(ind);
		z[i].resize(ind);
		u[i].resize(ind);
		v[i].resize(ind);
	}
	minX.resize(ind);
	minY.resize(ind);
	maxX.resize(ind);
	maxY.resize(ind);
	modelIndex.resize(ind);
	firstThread.resize(ind);
	lastThread.resize(ind);
	rcpSignedArea.resize(ind);
}