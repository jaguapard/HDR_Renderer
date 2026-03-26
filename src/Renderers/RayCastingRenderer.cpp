#include "RayCastingRenderer.h"
#include <optional>
#include "../Vec.h"
#include "../GameSettings.h"
#include "../AssetLoader.h"
#include "../Threadpool.h"

void RayCastingRenderer::loadScene(std::string path, std::string mode)
{
	AssetLoader ldr;
	std::vector<AssetLoader::ImportedModel> loadedModels;
	if (mode == "obj") { loadedModels = ldr.loadObj(path, "H:\\Sponza goodies\\Old Sponza 2026.bmdl"); }
	else if (mode == "bmdl") { loadedModels = ldr.loadBmdl(path); }
	else throw std::runtime_error("Unsupported mode for RayCastingRenderer::loadScene: " + mode);

	//TODO: load textures
	for (int i = 0; i < loadedModels.size(); ++i)
	{
		std::vector<Triangle> tris;
		for (auto& it : loadedModels[i].triangles)
		{
			auto& t = tris.emplace_back();
			for (int k = 0; k < 3; ++k)
			{
				t.tv[k].space = { it.vertices[k][0], it.vertices[k][1], it.vertices[k][2], 0 };
				t.tv[k].diffuse = { it.u[k], it.v[k], 0,0 };
			}
		}
		Model m;
		m.triangles = tris;
		this->sceneModels.emplace_back(m);
	}
}

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
	Vec4f* pixels = (Vec4f*)settings.graphicsOutputBuffer;

	std::vector<task_id> tasks;
	for (int y = 0; y < bufH; ++y)
	{
		tasks.emplace_back(settings.threadpool->addTask([&, y]() {
			for (int x = 0; x < bufW; ++x)
			{
				float progressX = x / float(bufH);
				float progressY = 1 - y / float(bufH);
				Vec4f rayDir = forward * settings.cameraPlane_zDist + down * (progressY - 0.5) + right * (progressX - widthToHeightRatio * 0.5);

				float minT = INFINITY;
				bool hit = false;

				size_t triangleCounter = 0;
				for (auto& model : this->sceneModels)
				{
					for (auto& triangle : model.triangles)
					{
						//if (triangleCounter++ % 128 != 0) continue;
						float t = rayTriangleIntersectionT(camPos, rayDir, triangle.tv[0].space, triangle.tv[1].space, triangle.tv[2].space);
						if (t < minT)
						{
							minT = t;
						}
					}
				}

				if (minT != INFINITY)
				{
					float distScalar = minT / 1000;
					float intensity = std::max(0.1f, 1 - distScalar);
					pixels[y * bufW + x] = { intensity,intensity,intensity,1 };
				}
				else pixels[y * bufW + x] = { 0,0,0,1 };
				/*
				for (int i = 0; i < 6; ++i)
				{
					float t = rayTriangleIntersectionT(camPos, rayDir, vertices[i*3], vertices[i*3+1], vertices[i*3+2]);
					if (t < minT)
					{
						float intensity = std::min(1.f / t, 1.f);
						pixels[y * bufW + x] = colors[i];
						hit = true;
					}
				}*/
			}
		}));
	}
	settings.threadpool->waitForMultipleTasks(tasks);
}
