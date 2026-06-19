#include "RayCastingRenderer.h"
#include <optional>
#include "../../Vec.h"
#include "../../GameSettings.h"
#include "../../AssetLoader.h"
#include "../../Threadpool.h"
#include <iostream>
#include "Octree.h"

using namespace RayCasting;

//https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
float rayTriangleIntersectionT(Vec4f rayOrigin, Vec4f rayDir, Vec4f triA, Vec4f triB, Vec4f triC)
{
	//TODO: change dot products to 3d!
	constexpr float epsilon = std::numeric_limits<float>::epsilon();
	constexpr float eps = std::numeric_limits<float>::epsilon();

	Vec4f edge1 = triB - triA;
	Vec4f edge2 = triC - triA;

	// Backface culling, assuming CCW-wound triangles.
	//const Vec4f normal = edge1.cross3d(edge2); // No need to normalize
	//if (normal.dot(rayDir) > 0) return FLT_MAX;

	Vec4f ray_cross_e2 = rayDir.cross3d(edge2);
	float det = edge1.dot(ray_cross_e2);

	if (abs(det) < epsilon) return FLT_MAX; // Ray is parallel to triangle

	float inv_det = 1.0 / det;
	Vec4f s = rayOrigin - triA;
	float u = inv_det * s.dot(ray_cross_e2);

	if (u < -eps || u - 1 > eps) return FLT_MAX; // Ray passes outside edge2's bounds

	Vec4f s_cross_e1 = s.cross3d(edge1);
	float v = inv_det * rayDir.dot(s_cross_e1);

	if (v < -eps || u + v - 1 > eps) return FLT_MAX; // Ray passes outside edge1's bounds

	// The ray line intersects with the triangle.
	// We compute t to find where on the ray the intersection is.
	float t = inv_det * edge2.dot(s_cross_e1);

	if (t > epsilon) // Ray intersection
	{
		return t;
	}
	else // This means that there is a line intersection but not a ray intersection.
		return FLT_MAX;
}

//Checks 16 rays for intersection with 1 triangle, returning mask of rays hitting the triangle.
//Intersection T is written out retT. The values of T are undefined for non-intersecting rays
//https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
Mask16 raysTriangleIntersectionTs(Vec4_f32x16 rayOrigins, Vec4_f32x16 rayDirs, Vec4f triA, Vec4f triB, Vec4f triC, float32x16& retT)
{
	constexpr float epsilon = std::numeric_limits<float>::epsilon();
	constexpr float eps = std::numeric_limits<float>::epsilon();

	Vec4_f32x16 edge1 = triB - triA;
	Vec4_f32x16 edge2 = triC - triA;

	Mask16 activeRays = 0xFFFF;
	// Backface culling, assuming CW-wound triangles.
	/*
	const Vec4_f32x16 normal = edge1.cross3d(edge2); // No need to normalize
	activeRays &= normal.dot3d(rayDirs) > 0.f;
	if (!activeRays) return 0;*/

	Vec4_f32x16 ray_cross_e2 = rayDirs.cross3d(edge2);
	float32x16 det = edge1.dot3d(ray_cross_e2);

	activeRays &= abs(det) >= eps;
	if (!activeRays) return 0; // Ray is parallel to triangle

	float32x16 inv_det = float32x16(1.f) / det;
	Vec4_f32x16 s = rayOrigins - triA;
	float32x16 u = inv_det * s.dot3d(ray_cross_e2);

	activeRays &= u >= -eps & (u - 1) <= eps;
	if (!activeRays) return 0; // Ray passes outside edge2's bounds

	Vec4_f32x16 s_cross_e1 = s.cross3d(edge1);
	float32x16 v = inv_det * rayDirs.dot3d(s_cross_e1);
	activeRays &= (v >= -eps) & (u + v - 1) <= eps; 
	if (!activeRays) return 0; // Ray passes outside edge1's bounds

	// The ray line intersects with the triangle.
	// We compute t to find where on the ray the intersection is.
	// t < epsilon means that there is a line intersection but not a ray intersection.
	float32x16 t = inv_det * edge2.dot3d(s_cross_e1);
	retT = t;
	return activeRays & t > epsilon; // Ray intersection
}

//TODO: verify that it works
//Checks 8 rays for intersection with 1 triangle, returning mask of rays hitting the triangle.
//Intersection T is written out retT. The values of T are undefined for non-intersecting rays
//https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
Mask8 raysTriangleIntersectionTs(Vec4_f32x8 rayOrigins, Vec4_f32x8 rayDirs, Vec4f triA, Vec4f triB, Vec4f triC, float32x8& retT)
{
	constexpr float epsilon = std::numeric_limits<float>::epsilon();
	const f32x8 eps = std::numeric_limits<float>::epsilon();

	Vec4_f32x8 edge1 = triB - triA;
	Vec4_f32x8 edge2 = triC - triA;

	Mask8 activeRays = 0xFF;
	// Backface culling, assuming CW-wound triangles.
	/*
	const Vec4_f32x8 normal = edge1.cross3d(edge2); // No need to normalize
	activeRays &= normal.dot3d(rayDirs) > 0.f;
	if (!activeRays) return 0;*/

	Vec4_f32x8 ray_cross_e2 = rayDirs.cross3d(edge2);
	float32x8 det = edge1.dot3d(ray_cross_e2);
	
	activeRays &= abs(det) >= eps;
	if (!activeRays) return 0; // Ray is parallel to triangle

	float32x8 inv_det = float32x8(1.f) / det;
	Vec4_f32x8 s = rayOrigins - triA;
	float32x8 u = inv_det * s.dot3d(ray_cross_e2);

	activeRays &= u >= -eps & ((u - 1) <= eps);
	if (!activeRays) return 0; // Ray passes outside edge2's bounds

	Vec4_f32x8 s_cross_e1 = s.cross3d(edge1);
	float32x8 v = inv_det * rayDirs.dot3d(s_cross_e1);
	activeRays &= (v >= -eps) & ((u + v - 1) <= eps);
	if (!activeRays) return 0; // Ray passes outside edge1's bounds

	// The ray line intersects with the triangle.
	// We compute t to find where on the ray the intersection is.
	// t < epsilon means that there is a line intersection but not a ray intersection.
	float32x8 t = inv_det * edge2.dot3d(s_cross_e1);
	retT = t;
	return activeRays & (t > epsilon); // Ray intersection
}
void RayCastingRenderer::loadScene(RendererLoadSceneData scd)
{
	this->octree.root.reset();
	this->textureManager.clear();
	this->sceneModels.clear();

	AssetLoader ldr;
	std::vector<AssetLoader::ImportedModel> loadedModels;
	for (auto [path, mode] : scd.files)
	{
		if (mode == "obj") { loadedModels = ldr.loadObj(path, ""); }
		else if (mode == "bmdl") { loadedModels = ldr.loadBmdl(path); }
		else throw std::runtime_error("Unsupported mode for RayCastingRenderer::loadScene: " + mode);

		//TODO: load textures
		size_t importModelCount = loadedModels.size();
		std::vector<int> diffuseMapIndices(importModelCount, -1);
		std::vector<Threadpool::TaskHandle> textureLoadingTasks(importModelCount);

		Threadpool::Task tsk;
		for (int i = 0; i < importModelCount; ++i)
		{
			tsk.func = [&, this, i]() {
				if (loadedModels[i].diffuseMapPath) diffuseMapIndices[i] = this->textureManager.addTextureByPath(*loadedModels[i].diffuseMapPath);
				else diffuseMapIndices[i] = 0;
				};
			textureLoadingTasks[i] = Threadpool::instance->addTask(tsk);
		}

		for (int i = 0; i < loadedModels.size(); ++i)
		{
			std::vector<Triangle> tris;
			for (auto& it : loadedModels[i].triangles)
			{
				auto& t = tris.emplace_back();
				for (int k = 0; k < 3; ++k)
				{
					t.tv[k].space = { it.v[k].space.x, it.v[k].space.y, it.v[k].space.z, 0.f };
					t.tv[k].diffuse = { it.v[k].diffuseMapCoords.x, -it.v[k].diffuseMapCoords.y, 0.f,0.f };
					t.tv[k].normal = { it.v[k].normal.x, it.v[k].normal.y, it.v[k].normal.z, 0.f };
				}
			}
			Model m;
			m.triangles = tris;
			this->sceneModels.emplace_back(m);
		}

		Threadpool::instance->blockUntilComplete(textureLoadingTasks);
		for (int i = 0; i < loadedModels.size(); ++i)
			this->sceneModels[i].textureIndex = diffuseMapIndices[i];

		if (this->discardUntexturedTriangles)
		{
			for (int i = 0; i < sceneModels.size(); ++i)
			{
				if (this->sceneModels[i].textureIndex == 0) {
					this->sceneModels[i--] = std::move(this->sceneModels.back());
					this->sceneModels.pop_back();
				}
			}
		}
	}
	this->octree = RayCasting::Octree(*this);
}

void RayCastingRenderer::renderFrame(const GameSettings& settings)
{
	int bufW = settings.outputTextureW, bufH = settings.outputTextureH;
	//coordinate check
	/*	first:	second:
		___		
		\  |	|\		
		 \ |	| \
		  \|	|__\
	*/
	float size = 20;
	//first 2 show direction of X, second 2 - Y, last 2 - Z. Ideally first goes right, second down, third away from camera
	//expected result - red-green bar goes right, blue-yellow down, magenta-white away from the screen
	float d = 10;
	Vec4f vertices[] = {
		{0, 0, 0},
		{size, 0, 0},
		{size, 0, size/d},
		{size, 0, size/d},
		{0, 0, size/d},
		{0, 0, 0},

		{0,0,0},
		{0,size,0},
		{size/d,size,0},
		{size/d,size,0},
		{size/d,0,0},
		{0,0,0},

		{0,0,0},
		{0,0,size},
		{size/d,0,size},
		{size/d,0,size},
		{size/d,0,0},
		{0,0,0},
	};

	Vec4f colors[] = {
		{1,0,0,1},
		{0,1,0,1},
		{0,0,1,1},
		{1,1,0,1},
		{1,0,1,1},
		{1,1,1,1},
	};

	Vec4f forward = settings.forward;
	Vec4f right = settings.right;
	Vec4f down = settings.down;
	Vec4f camPos = settings.camPos;

	//camPos = { 0,0, -20 };
	
	float widthToHeightRatio = double(bufW) / bufH;
	uint64_t* pixels = (uint64_t*)settings.graphicsOutputBuffer;

	std::vector<Threadpool::TaskHandle> tasks;
	Threadpool::Task tsk;
	Vec4_f32x16 rayOrigins = camPos;
	Vec4f lightDir = { 0.4, 0.5, 0.2, 0 };
	lightDir /= lightDir.len();
	float32x16 ambientLightIntensity = 0.1;
	int threadCount = Threadpool::instance->getWorkerCount();
	for (int yStart = 0; yStart < bufH; yStart += 4)
	{
		tsk.func = [&, this, yStart]() 
		#ifdef VS_CLANG 
			__attribute__((noinline)) //Prevent inlining of lambda on Clang. Without it, profiling results are total garbage. MSVC doesn't work with this, but it has useful profiling without it.
		#endif
			{
			int threadIndexFake = (yStart/4) % threadCount;
			float32x16 y = float32x16(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3) + yStart;
			for (float32x16 x = float32x16(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3); Mask16 bounds = x < bufW & y < bufH; x += 4)
			{
				float32x16 progressX = x / float(bufH);
				float32x16 progressY = y / float(bufH);
				Vec4_f32x16 rayDirs = Vec4_f32x16(forward) * settings.cameraPlane_zDist + Vec4_f32x16(down) * (progressY - 0.5) + Vec4_f32x16(right) * (progressX - widthToHeightRatio * 0.5);
				rayDirs /= rayDirs.len3d();

				TraceResults hits = this->traceRays(rayOrigins, rayDirs, bounds, false, threadIndexFake);

				Vec4_f32x16 textureColors(0.f, 0.f, 0.f, 1.f);
				if (hits.raysHit)
				{
					int32x16 modelTextureIndOffset = hits.modelIndices * sizeof(RayCasting::Model) + offsetof(RayCasting::Model, textureIndex);
					int32x16 diffuseMapIndices = gather<i32x16, 1>(this->sceneModels.data(), modelTextureIndOffset, hits.raysHit);
					textureColors = this->textureManager.gatherLinearIntesitiesFromMultipleTextures(diffuseMapIndices, hits.textureCoords[0], hits.textureCoords[1], hits.raysHit);
					
					float32x16 normalShadingMult = max(f32x16(0), hits.normals.dot3d(lightDir));
					Vec4_f32x16 shadowTraceRayOrigins = rayOrigins + rayDirs * hits.t + hits.normals * 1;
					TraceResults shadowTrace = this->traceRays(shadowTraceRayOrigins, lightDir, hits.raysHit, true, threadIndexFake);
					float32x16 totalMult = ambientLightIntensity + maskz_mov(~shadowTrace.raysHit, normalShadingMult);
					for (int i = 0; i < 3; ++i)
					{
						textureColors[i] *= totalMult;
					}
				}
				size_t xInt = x[0];

				u16x16 fp16_r = vcvt_fp32_fp16(textureColors.r);
				u16x16 fp16_g = vcvt_fp32_fp16(textureColors.g);
				u16x16 fp16_b = vcvt_fp32_fp16(textureColors.b);
				u16x16 fp16_a = vcvt_fp32_fp16(textureColors.a); //TODO: can be forced to 1 and moved later
				for (int packY = 0; packY < 4; ++packY)
				{
					u16x16 fp16_rg = permx2(fp16_r, fp16_g, u16x16(0, 16, 0, 0, 1, 17, 0, 0, 2, 18, 0, 0, 3, 19, 0, 0) + packY * 4);
					u16x16 fp16_ba = permx2(fp16_b, fp16_a, u16x16(0, 0, 0, 16, 0, 0, 1, 17, 0, 0, 2, 18, 0, 0, 3, 19) + packY * 4);
					u16x16 toStore = mask_mov(fp16_rg, 0b1100110011001100, fp16_ba);
					store(vcast<u64x4>(toStore), (uint64_t*)(settings.graphicsOutputBuffer) + (yStart + packY) * bufW + xInt, bounds >> 4 * packY);
				}
			}
		};
		tasks.emplace_back(settings.threadpool->addTask(tsk));
	}
	
	settings.threadpool->blockUntilComplete(tasks);
}

RayCasting::TraceResults RayCastingRenderer::traceRays(Vec4_f32x16 rayOrigins, Vec4_f32x16 rayDirs, Mask16 activeRays, bool shadowRays, uint32_t threadIndex)
{
	Vec4_f32x16 rcpRayDirs = Vec4_f32x16(1.f, 1.f, 1.f, 0.f) / rayDirs;
	std::array<OctreeNode*, 2048> stack;
	TraceResults ret;

	int stackTopIndex = 1;
	stack[0] = this->octree.root.get();
	uint64_t nodesInspected = 0, trianglesInspected = 0, triangleIntersectionTests = 0, triangleIntersectionTestsLive = 0, rayNodeIntersections = 0, rayNodeTests = 0;
	while (stackTopIndex > 0)
	{
		++nodesInspected;
		OctreeNode* currNode = stack[--stackTopIndex];
		float32x16 bboxTmin, bboxTmax; //not used
		Mask16 raysIntersectingNodeBoundingBox = activeRays & currNode->bbox.getMinAndMaxIntestionsFor(rayOrigins, rcpRayDirs, bboxTmin, bboxTmax) & ret.t > bboxTmin;
		rayNodeIntersections += std::popcount((uint32_t)raysIntersectingNodeBoundingBox);
		rayNodeTests += 16;
		if (!raysIntersectingNodeBoundingBox) continue;

		for (int contentInd = 0; const OctreeContent* content = currNode->getContentOrNull(contentInd); ++contentInd)
		{
			++trianglesInspected;
			uint32_t triangleIndex = content->triangleIndex;
			uint32_t modelIndex = content->modelIndex;
			const Triangle& triangle = this->sceneModels[modelIndex].triangles[triangleIndex];
			float32x16 t;
			Mask16 raysHittingThisTriangle = activeRays & raysTriangleIntersectionTs(rayOrigins, rayDirs, triangle.tv[0].space, triangle.tv[1].space, triangle.tv[2].space, t);
			triangleIntersectionTestsLive += std::popcount((uint32_t)raysHittingThisTriangle);
			triangleIntersectionTests += 16;
			if (!raysHittingThisTriangle) continue;

			std::array<float32x16, 3> worldBarycentrics;
			calculateBarycentricCoordinates3D<Vec4_f32x16>(rayOrigins + rayDirs * t, triangle.tv[0].space, triangle.tv[1].space, triangle.tv[2].space, worldBarycentrics);
			Vec4_f32x16 uv(0.f, 0.f, 0.f, 0.f);
			for (int i = 0; i < 3; ++i) uv += Vec4_f32x16(triangle.tv[i].diffuse) * worldBarycentrics[i];

			const auto& texture = this->textureManager.getTextureByHandle(this->sceneModels[modelIndex].textureIndex);
			//auto accessor = texture.getGatherAccessor(uv.x, uv.y, raysHittingThisTriangle); //todo: add t < ret.t?
			float32x16 textureAlpha = texture.gatherA(uv.x, uv.y, raysHittingThisTriangle);

			Mask16 toOverride = raysHittingThisTriangle & t < ret.t & textureAlpha >= 1.f;
			ret.raysHit |= toOverride;
			if (!shadowRays)
			{
				ret.t = mask_mov(ret.t, toOverride, t);

				ret.modelIndices = mask_mov(ret.modelIndices, toOverride, int32x16(modelIndex));
				ret.triangleIndices = mask_mov(ret.triangleIndices, toOverride, int32x16(triangleIndex));
				Vec4_f32x16 normals(0.f, 0.f, 0.f, 0.f);
				for (int i = 0; i < 3; ++i) normals += Vec4_f32x16(triangle.tv[i].normal) * worldBarycentrics[i];
				normals /= normals.len3d();

				for (int k = 0; k < 3; ++k)
				{
					ret.worldBarycentrics[k] = mask_mov(ret.worldBarycentrics[k], toOverride, worldBarycentrics[k]);
					ret.normals[k] = mask_mov(ret.normals[k], toOverride, normals[k]);
				}
				for (int k = 0; k < 2; ++k)
				{
					ret.textureCoords[k] = mask_mov(ret.textureCoords[k], toOverride, uv[k]);
				}
			}
			else
			{
				activeRays &= ~toOverride;
				if (!activeRays) goto end;
			}
		}

		for (int i = 0; i < currNode->CHILD_COUNT; ++i)
		{
			OctreeNode* child = currNode->getChild(i);
			if (child) stack[stackTopIndex++] = child;
		}
	}
	end:
	if (Statsman::ENABLED)
	{
		MyStatsman.rayCasting.nodesInspected += nodesInspected;
		MyStatsman.rayCasting.trianglesInspected += trianglesInspected;
		//MyStatsman.rayCasting.rayCount += 16;
		MyStatsman.rayCasting.triangleIntersectionTests += triangleIntersectionTests;
		MyStatsman.rayCasting.triangleIntersectionTestsLive += triangleIntersectionTestsLive;
		MyStatsman.rayCasting.rayNodeIntersectionTests += rayNodeTests;
		MyStatsman.rayCasting.rayNodeIntersections += rayNodeIntersections;

	}
	return ret;
}
