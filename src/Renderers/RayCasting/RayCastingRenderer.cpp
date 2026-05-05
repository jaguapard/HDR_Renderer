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
	//if (normal.dot(rayDir) > 0) return INFINITY;

	Vec4f ray_cross_e2 = rayDir.cross3d(edge2);
	float det = edge1.dot(ray_cross_e2);

	if (abs(det) < epsilon) return INFINITY; // Ray is parallel to triangle

	float inv_det = 1.0 / det;
	Vec4f s = rayOrigin - triA;
	float u = inv_det * s.dot(ray_cross_e2);

	if (u < -eps || u - 1 > eps) return INFINITY; // Ray passes outside edge2's bounds

	Vec4f s_cross_e1 = s.cross3d(edge1);
	float v = inv_det * rayDir.dot(s_cross_e1);

	if (v < -eps || u + v - 1 > eps) return INFINITY; // Ray passes outside edge1's bounds

	// The ray line intersects with the triangle.
	// We compute t to find where on the ray the intersection is.
	float t = inv_det * edge2.dot(s_cross_e1);

	if (t > epsilon) // Ray intersection
	{
		return t;
	}
	else // This means that there is a line intersection but not a ray intersection.
		return INFINITY;
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

	// Backface culling, assuming CCW-wound triangles.
	//const Vec4f normal = edge1.cross3d(edge2); // No need to normalize
	//if (normal.dot3d(rayDir) > 0) return INFINITY;

	Vec4_f32x16 ray_cross_e2 = rayDirs.cross3d(edge2);
	float32x16 det = edge1.dot3d(ray_cross_e2);

	Mask16 intersectingTriangles = float32x16(_mm512_abs_ps(det)) >= eps;
	if (!intersectingTriangles) return 0; // Ray is parallel to triangle

	float32x16 inv_det = float32x16(1.f) / det;
	Vec4_f32x16 s = rayOrigins - triA;
	float32x16 u = inv_det * s.dot3d(ray_cross_e2);

	intersectingTriangles &= u >= -eps & (u - 1) <= eps;
	if (!intersectingTriangles) return 0; // Ray passes outside edge2's bounds

	Vec4_f32x16 s_cross_e1 = s.cross3d(edge1);
	float32x16 v = inv_det * rayDirs.dot3d(s_cross_e1);
	intersectingTriangles &= (v >= -eps) & (u + v - 1) <= eps; 
	if (!intersectingTriangles) return 0; // Ray passes outside edge1's bounds

	// The ray line intersects with the triangle.
	// We compute t to find where on the ray the intersection is.
	// t < epsilon means that there is a line intersection but not a ray intersection.
	float32x16 t = inv_det * edge2.dot3d(s_cross_e1);
	retT = t;
	return intersectingTriangles & t > epsilon; // Ray intersection
}
void RayCastingRenderer::loadScene(RendererLoadSceneData scd)
{
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
				}
			}
			Model m;
			m.triangles = tris;
			this->sceneModels.emplace_back(m);
		}

		Threadpool::instance->blockUntilComplete(textureLoadingTasks);
		for (int i = 0; i < loadedModels.size(); ++i)
			this->sceneModels[i].textureIndex = diffuseMapIndices[i];
	}
	this->octree = RayCasting::Octree(*this);
}

void RayCastingRenderer::renderFrame(const GameSettings& settings)
{
	int bufW = settings.outputTextureParams.Width, bufH = settings.outputTextureParams.Height;
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
	std::atomic<uint64_t> nodesInspectedTotal = 0, triangleIntersectionChecksTotal = 0;
	Vec4f lightDir = { 0.4, 0.5, 0.2, 0 };
	lightDir /= lightDir.len();
	for (int y = 0; y < bufH; ++y)
	{
		tsk.func = [&, this, y]() 
		#ifdef VS_CLANG 
			__attribute__((noinline)) //Prevent inlining of lambda on Clang. Without it, profiling results are total garbage. MSVC doesn't work with this, but it has useful profiling without it.
		#endif
			{
			for (float32x16 x = float32x16::sequence(); Mask16 bounds = x < bufW; x += 16)
			{
				float32x16 progressX = x / float(bufH);
				float progressY = y / float(bufH);
				Vec4_f32x16 rayDirs = Vec4_f32x16(forward) * settings.cameraPlane_zDist + Vec4_f32x16(down) * (progressY - 0.5) + Vec4_f32x16(right) * (progressX - widthToHeightRatio * 0.5);
				rayDirs /= rayDirs.len3d();

				TraceResults results = this->traceRays(rayOrigins, rayDirs, bounds, false);
				
				//nodesInspectedTotal += nodesInspected;
				//triangleIntersectionChecksTotal += triangleIntersectionChecks;

				Vec4_f32x16 textureColors(0.f, 0.f, 0.f, 1.f);
				if (results.raysHit)
				{
					for (int i = 0; i < 16; ++i)
					{
						if (!(results.raysHit.mask & (1 << i))) continue;
						int diffuseMapIndex = this->sceneModels[results.hitModelIndices[i]].textureIndex;
						Vec4f texturePixel = this->textureManager.getTextureByHandle(diffuseMapIndex).getLinearIntensity(results.hitTextureCords[0][i], results.hitTextureCords[1][i]);
						textureColors.x[i] = texturePixel.x;
						textureColors.y[i] = texturePixel.y;
						textureColors.z[i] = texturePixel.z;
						textureColors.w[i] = texturePixel.w;
					}

					Vec4_f32x16 shadowTraceRayOrigins = rayOrigins + rayDirs * results.minT + results.normals * 1;
					TraceResults shadowTrace = this->traceRays(shadowTraceRayOrigins, lightDir, results.raysHit, true);
					for (int i = 0; i < 3; ++i)
					{
						textureColors[i] = _mm512_mask_mul_ps(textureColors[i], shadowTrace.raysHit, textureColors[i], float32x16(0.1));
					}
				}
				size_t xInt = x[0];
				mask_store_vec4_f32x16_to_framebuffer(textureColors, settings.graphicsOutputBuffer, xInt, y, settings.outputTextureParams.Width, bounds);
			}
		};
		tasks.emplace_back(settings.threadpool->addTask(tsk));
	}
	
	settings.threadpool->blockUntilComplete(tasks);
	//TODO: make statsman for this renderer
	std::cout << "Nodes inspected: " << nodesInspectedTotal << " (" << nodesInspectedTotal / double(bufW * bufH) << " per pixel)\n";
	std::cout << "Triangles inspected: " << triangleIntersectionChecksTotal << " (" << triangleIntersectionChecksTotal / double(bufW * bufH) << " per pixel)\n";
}

RayCasting::TraceResults RayCastingRenderer::traceRays(Vec4_f32x16 rayOrigins, Vec4_f32x16 rayDirs, Mask16 mask, bool shadowRays)
{
	Vec4_f32x16 rcpRayDirs = Vec4_f32x16(1.f, 1.f, 1.f, 0.f) / rayDirs;
	std::array<OctreeNode*, 2048> stack;
	TraceResults ret;

	int stackTopIndex = 1;
	stack[0] = this->octree.root.get();
	uint64_t nodesInspected = 0, triangleIntersectionChecks = 0;
	while (stackTopIndex > 0)
	{
		++nodesInspected;
		OctreeNode* currNode = stack[--stackTopIndex];
		float32x16 bboxTmin, bboxTmax; //not used
		Mask16 raysIntersectingNodeBoundingBox = mask & currNode->bbox.getMinAndMaxIntestionsFor(rayOrigins, rcpRayDirs, bboxTmin, bboxTmax);
		if (!raysIntersectingNodeBoundingBox) continue;

		for (auto& content : currNode->contents)
		{
			//++triangleIntersectionChecks;
			uint32_t triangleIndex = content.triangleIndex;
			uint32_t modelIndex = content.modelIndex;
			const Triangle& triangle = this->sceneModels[modelIndex].triangles[triangleIndex];
			float32x16 t;
			Mask16 raysHittingThisTriangle = mask & raysTriangleIntersectionTs(rayOrigins, rayDirs, triangle.tv[0].space, triangle.tv[1].space, triangle.tv[2].space, t);
			if (!raysHittingThisTriangle) continue;

			Mask16 toOverride = raysHittingThisTriangle & t < ret.minT;
			ret.raysHit |= toOverride;
			ret.minT = _mm512_mask_mov_ps(ret.minT, toOverride, t);
			//TODO: all traces will need textures, since they can be fully or semi-transparent. For now, shadows skip all this
			if (!shadowRays)
			{
				ret.hitModelIndices = _mm512_mask_mov_epi32(ret.hitModelIndices, toOverride, int32x16(modelIndex));
				ret.hitTriangleIndices = _mm512_mask_mov_epi32(ret.hitTriangleIndices, toOverride, int32x16(triangleIndex));

				std::array<float32x16, 3> barycentrics;
				calculateBarycentricCoordinates3D(rayOrigins + rayDirs * ret.minT, triangle.tv[0].space, triangle.tv[1].space, triangle.tv[2].space, barycentrics);
				Vec4f faceNormal = getFaceNormalForTriangle(triangle.tv[0].space, triangle.tv[1].space, triangle.tv[2].space);
				Vec4_f32x16 uv(0.f, 0.f, 0.f, 0.f);
				for (int k = 0; k < 3; ++k)
				{
					ret.hitBarycentrics[k] = _mm512_mask_mov_ps(ret.hitBarycentrics[k], toOverride, barycentrics[k]);
					ret.normals[k] = _mm512_mask_mov_ps(ret.normals[k], toOverride, float32x16(faceNormal[k]));
					uv += Vec4_f32x16(triangle.tv[k].diffuse) * barycentrics[k];
				}
				for (int k = 0; k < 2; ++k)
				{
					ret.hitTextureCords[k] = _mm512_mask_mov_ps(ret.hitTextureCords[k], toOverride, uv[k]);
				}
			}
		}

		for (int i = 0; i < currNode->CHILD_COUNT; ++i)
		{
			if (currNode->children[i]) stack[stackTopIndex++] = currNode->children[i].get();
		}
	}
	return ret;
}
