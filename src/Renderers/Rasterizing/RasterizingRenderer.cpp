#include "RasterizingRenderer.h"
#include "../../AssetLoader.h"
#include <stdexcept>
#include <unordered_map>
#include "../../Vec.h"
#include "../../GameSettings.h"
#include "../../Threadpool.h"
#include <map>
#include <iostream>
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
void RasterizingRenderer::loadScene(std::string path, std::string mode)
{
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
		}
		return;
	}
	AssetLoader ldr;
	std::vector<AssetLoader::ImportedModel> loadedModels;
	if (mode == "obj") { loadedModels = ldr.loadObj(path, "H:\\Sponza goodies\\Old Sponza 2026.bmdl"); }
	else if (mode == "bmdl") { loadedModels = ldr.loadBmdl(path); }
	else throw std::runtime_error("Unsupported mode for RasterizingRenderer::loadScene: " + mode);

	//uint32_t vertCount = 0, uvCount = 0;
	//TODO: load textures
	for (int i = 0; i < loadedModels.size(); ++i)
	{
		Model& m = this->sceneModels.emplace_back();
		if (loadedModels[i].diffuseMapPath) m.diffuseMapIndex = this->textureManager.addTextureByPath(*loadedModels[i].diffuseMapPath);
		//Need to track the ownership of indices from global vertice stores. I.e. mapping of
		std::vector<int> modelXyzIndices, modelUvIndices;
		for (auto& it : loadedModels[i].triangles)
		{
			for (int k = 0; k < 3; ++k)
			{
				uint32_t vertInd = this->original_verticeStore.insert(
					it.vertices[k][0], it.vertices[k][1], it.vertices[k][2], it.u[k], it.v[k]
				);

				//TODO: clean from degenerate triangles (i.e 2 or 3 vertices same or all 3 collinear)?
				//this->original_triangleStore.vertInd[k].push_back(vertInd);
				m.triangleStore.vertInd[k].push_back(vertInd);
			}
			//this->original_triangleStore.modelInd.push_back(i);
		}
		
	}

	int a = 0;
	//assert(originalX.size() == originalY.size() && originalY.size() == originalZ.size());
	//assert(originalU.size() == originalV.size());
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

float32x16 inverse_lerp(float32x16 from, float32x16 to, float32x16 value)
{
	return (value - from) / (to - from);
}
void RasterizingRenderer::doTransformationsAndClipping(int threadIndex)
{
	assert(this->original_verticeStore.x.size() == this->original_verticeStore.y.size() && this->original_verticeStore.y.size() == this->original_verticeStore.z.size() && this->original_verticeStore.u.size() == this->original_verticeStore.v.size());

	float rcpScreenHeightPerThread = double(this->currGs->threadpool->getThreadCount()) / this->currGs->outputTextureParams.Height;
	float clippingZ = this->currGs->cameraPlane_zDist;
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
			Vec4_f32x16 transformedVertices[3];

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
				transformedVertices[i] = rotatedTranslated;
				verticeIndicesCache[i] = verticeIndices;
			}
			Mask16 allBehindMask = behindPlaneCount == 3;
			if (behindPlaneCount > 0) continue; //TODO: remove this after implementing plane clipping
			//if (!(behindPlaneCount != 3)) continue; //if all vertices behind the plane, continue

			

			//backface culling
			/*
			if (currFrameGameSettings.backfaceCullingEnabled && !model.noBackfaceCulling)
			{
			Vec4 normal = rotated.getNormalVector();
			if (rotated.tv[0].spaceCoords.dot(normal) >= 0) return 0;
			}
			*/


			Mask16 activeTrianglesMask = inBoundsTrianglesMask & (behindPlaneCount == 0); //TODO: change this after implementing clipping to behindPlaneCount != 3
			float32x16 u[3], v[3];
			for (int i = 0; i < 3; ++i)
			{
				u[i] = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), activeTrianglesMask, verticeIndicesCache[i], this->original_verticeStore.u.data(), 4);
				v[i] = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), activeTrianglesMask, verticeIndicesCache[i], this->original_verticeStore.v.data(), 4);
			}

			//TODO: implement clipping by camera plane!
			if (behindPlaneCount > 0) //skip clipping attempts for non-plane-bordering triangles
			{
				//if 0 vertices behind - don't touch them
				//if 1 - this triangle turns into two
				//if 2 - can change the original triangle


			}
			
			float32x16 fovMult = 1;
			float32x16 minX = INFINITY, maxX = -INFINITY, minY = INFINITY, maxY = -INFINITY;
			for (int i = 0; i < 3; ++i)
			{
				float32x16 zInv = fovMult / transformedVertices[i].z;
				u[i] *= zInv;
				v[i] *= zInv;
				Vec4_f32x16 screenSpaceVertice = this->ctr.screenSpaceToPixels(transformedVertices[i] * zInv);
				screenSpaceVertice.z = zInv;
				transformedVertices[i] = screenSpaceVertice;
				minX = _mm512_min_ps(minX, screenSpaceVertice.x);
				minY = _mm512_min_ps(minY, screenSpaceVertice.y);
				maxX = _mm512_max_ps(maxX, screenSpaceVertice.x);
				maxY = _mm512_max_ps(maxY, screenSpaceVertice.y);
			}

			const Vec4_f32x16& r1 = transformedVertices[0];
			const Vec4_f32x16& r2 = transformedVertices[1];
			const Vec4_f32x16& r3 = transformedVertices[2];
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
			int32x16 firstThread = _mm512_cvttps_epi32(minY * rcpScreenHeightPerThread);
			int32x16 lastThread = _mm512_cvttps_epi32(maxY * rcpScreenHeightPerThread);
			//since render job store resize does 16 overprovisioning on resize, it's OK not to mask these stores. Some architectures have awful compress store unaligned performance
			for (int i = 0; i < 3; ++i)
			{
				_mm512_storeu_ps(myRjStore.x[i].data() + rjRealSize, _mm512_maskz_compress_ps(activeTrianglesMask, transformedVertices[i].x));
				_mm512_storeu_ps(myRjStore.y[i].data() + rjRealSize, _mm512_maskz_compress_ps(activeTrianglesMask, transformedVertices[i].y));
				_mm512_storeu_ps(myRjStore.z[i].data() + rjRealSize, _mm512_maskz_compress_ps(activeTrianglesMask, transformedVertices[i].z));
				_mm512_storeu_ps(myRjStore.u[i].data() + rjRealSize, _mm512_maskz_compress_ps(activeTrianglesMask, u[i]));
				_mm512_storeu_ps(myRjStore.v[i].data() + rjRealSize, _mm512_maskz_compress_ps(activeTrianglesMask, v[i]));
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

	/*
	size_t trisInSlices = 0;
	for (auto& m : )
	{
		for ()
	}
	assert(seenTris == trisInSlices);*/
	this->renderJobsFromThreads[threadIndex].resize(rjRealSize, false);
}

std::tuple<float32x16, float32x16, float32x16> calculateBarycentricCoordinates(const Vec4_f32x16& r, const Vec4f& r1, const Vec4f& r2, const Vec4f& r3, const float rcpSignedArea)
{
	return {
		(r - r3).cross2d(r2 - r3) * rcpSignedArea,
		(r - r3).cross2d(r3 - r1) * rcpSignedArea,
		(r - r1).cross2d(r1 - r2) * rcpSignedArea //do NOT change this to 1-alpha-beta or 1-(alpha+beta). That causes wonkiness in textures
	};
}

void RasterizingRenderer::drawRenderJobs(int threadIndex)
{
	auto [d_low, d_high] = this->currGs->threadpool->getLimitsForThread(threadIndex, 0, this->currGs->outputTextureParams.Height);
	float my_yMin = floor(d_low);
	float my_yMax = std::min<float>(floor(d_high), this->currGs->outputTextureParams.Height-1);
	float my_xMin = 0;
	float my_xMax = this->currGs->outputTextureParams.Width - 1;
	int w = this->currGs->outputTextureParams.Width;

	for (auto& fromThread : this->renderJobsFromThreads)
	{
		size_t jobCount = fromThread.size();
		for (size_t jobIndex = 0; jobIndex < jobCount; ++jobIndex)
		{
			//NOTE: this logic caused trouble before (threads skipped jobs too eagerly), flipping signs helped. Keep a look for it failing
			if (fromThread.firstThread[jobIndex] > threadIndex && fromThread.lastThread[jobIndex] < threadIndex) continue;

			float xBeg = std::max(my_xMin, fromThread.minX[jobIndex]);
			float xEnd = std::min(my_xMax, fromThread.maxX[jobIndex]);
			float yBeg = std::max(my_yMin, fromThread.minY[jobIndex]);
			float yEnd = std::min(my_yMax, fromThread.maxY[jobIndex]);
			const auto& texture = this->textureManager.getTextureByHandle(this->sceneModels[fromThread.modelIndex[jobIndex]].diffuseMapIndex);

			for (float y = yBeg; y <= yEnd; ++y)
			{
				size_t yInt = y;
				size_t xInt = xBeg;
				for (float32x16 x = float32x16::sequence() + xBeg; Mask16 xBoundsMask = (x <= xEnd); x += 16, xInt += 16)
				{
					Vec4_f32x16 r = Vec4_f32x16(x, y, 0.0, 0.0);
					Vec4f v1_screen = Vec4f(fromThread.x[0][jobIndex], fromThread.y[0][jobIndex], 0, 0);
					Vec4f v2_screen = Vec4f(fromThread.x[1][jobIndex], fromThread.y[1][jobIndex], 0, 0);
					Vec4f v3_screen = Vec4f(fromThread.x[2][jobIndex], fromThread.y[2][jobIndex], 0, 0);
					auto [alpha, beta, gamma] = calculateBarycentricCoordinates(r, v1_screen, v2_screen, v3_screen, fromThread.rcpSignedArea[jobIndex]);

					Mask16 pointsInsideTriangleMask = xBoundsMask & (alpha >= 0.0) & (beta >= 0.0) & (gamma >= 0.0);
					if (!pointsInsideTriangleMask) continue;

					Vec4_f32x16 interpolatedDividedUv = Vec4_f32x16(fromThread.u[0][jobIndex], fromThread.v[0][jobIndex], fromThread.z[0][jobIndex], 0.f) * alpha + Vec4_f32x16(fromThread.u[1][jobIndex], fromThread.v[1][jobIndex], fromThread.z[1][jobIndex], 0.f) * beta + Vec4_f32x16(fromThread.u[2][jobIndex], fromThread.v[2][jobIndex], fromThread.z[2][jobIndex], 0.f) * gamma;

					//float32x16 currDepthValues = this->zBuffer.getPixels16(xInt, yInt);
					float32x16 currDepthValues = this->zBuffer.data() + yInt * w + xInt;
					//depth test: bigger Z pre-divide = further. However, we have reciprocal Z stored in interpolatedDividedUv.z, and Z <= 1 are culled during clipping stage, thus 1/z < z at all times
					//example: Z post rotate and translate (but before divide) for 2 pixels are 2 and 3. After Z divide they become 0.5 and 0.333. 0.5 should win the depth test, since it's closer
					Mask16 visiblePointsMask = pointsInsideTriangleMask & currDepthValues < interpolatedDividedUv.z;
					if (!visiblePointsMask) continue; //if all points are occluded, then skip

					Vec4_f32x16 uvCorrected = interpolatedDividedUv / interpolatedDividedUv.z;
					//Vec4_f32x16 texturePixels = texture.gatherPixels512(uvCorrected.x, uvCorrected.y, visiblePointsMask);
					Vec4_f32x16 texturePixels = texture.gatherLinearIntensities(uvCorrected.x, uvCorrected.y, visiblePointsMask);
					Mask16 opaquePixelsMask = visiblePointsMask & (texturePixels.a > 0.0f);
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
				}
			}
		}
	}
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