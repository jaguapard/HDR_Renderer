#include "RayCastingRenderer.h"
#include <optional>
#include "../../Vec.h"
#include "../../GameSettings.h"
#include "../../AssetLoader.h"
#include "../../Threadpool.h"

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
		for (int i = 0; i < loadedModels.size(); ++i)
		{
			std::vector<Triangle> tris;
			for (auto& it : loadedModels[i].triangles)
			{
				auto& t = tris.emplace_back();
				for (int k = 0; k < 3; ++k)
				{
					t.tv[k].space = { it.v[k].space.x, it.v[k].space.y, it.v[k].space.z, 0.f };
					t.tv[k].diffuse = { it.v[k].diffuseMapCoords.x, it.v[k].diffuseMapCoords.y, 0.f,0.f };
				}
			}
			Model m;
			m.triangles = tris;
			this->sceneModels.emplace_back(m);
		}
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
	for (int y = 0; y < bufH; ++y)
	{
		tsk.func = [&, this, y]() {
			std::array<OctreeNode*, 2048> stack;
			for (int x = 0; x < bufW; ++x)
			{
				float progressX = x / float(bufH);
				float progressY = y / float(bufH);
				Vec4f rayDir = forward * settings.cameraPlane_zDist + down * (progressY - 0.5) + right * (progressX - widthToHeightRatio * 0.5);
				Vec4f rcpRayDir = Vec4f(1, 1, 1, 0) / rayDir;

				float minT = INFINITY;
				bool hit = false;

				int stackTopIndex = 1;
				stack[0] = this->octree.root.get();
				while (stackTopIndex > 0)
				{
					OctreeNode* currNode = stack[--stackTopIndex];
					if (currNode->bbox.getMinAndMaxIntestionsFor(camPos, rcpRayDir).first == INFINITY) continue;

					for (auto& content : currNode->contents)
					{
						Triangle triangle = this->sceneModels[content.modelIndex].triangles[content.triangleIndex];
						float t = rayTriangleIntersectionT(camPos, rayDir, triangle.tv[0].space, triangle.tv[1].space, triangle.tv[2].space);
						if (t < minT)
						{
							minT = t;
						}
					}

					for (int i = 0; i < currNode->CHILD_COUNT; ++i)
					{
						if (currNode->children[i]) stack[stackTopIndex++] = currNode->children[i].get();
					}
				}

				Vec4f pixelColor;
				if (minT != INFINITY)
				{
					float distScalar = minT / 1000;
					float intensity = std::max(0.1f, 1 - distScalar);
					pixelColor = { intensity,intensity,intensity,1 };
				}
				else pixelColor = { 0,0,0,1 };

				pixels[y*bufW+x] = _mm_extract_epi64(_mm_cvtps_ph(pixelColor, _MM_FROUND_NO_EXC), 0);
			}
		};
		tasks.emplace_back(settings.threadpool->addTask(tsk));
	}
	
	settings.threadpool->blockUntilComplete(tasks);
}

BoundingBox RayCasting::OctreeNode::getBoundingBoxForChildIndex(int i) const
{
	int takeStepX = i & 1;
	int takeStepY = i & 2;
	int takeStepZ = i & 4;

	BoundingBox bb;
	float xStep = (this->bbox.xmax - this->bbox.xmin) * 0.5;
	float yStep = (this->bbox.ymax - this->bbox.ymin) * 0.5;
	float zStep = (this->bbox.zmax - this->bbox.zmin) * 0.5;
	bb.xmin = this->bbox.xmin + (takeStepX ? xStep : 0);
	bb.ymin = this->bbox.ymin + (takeStepY ? yStep : 0);
	bb.zmin = this->bbox.zmin + (takeStepZ ? zStep : 0);
	bb.xmax = bb.xmin + xStep;
	bb.ymax = bb.ymin + yStep;
	bb.zmax = bb.zmin + zStep;
	return bb;
}

bool RayCasting::OctreeNode::tryAddTriangle(int modelIndex, int triangleIndex, const RayCastingRenderer& rend)
{
	const Triangle& t = rend.sceneModels[modelIndex].triangles[triangleIndex];
	BoundingBox tbb = t;
	if (!this->bbox.containsFully(tbb)) return false;

	//scan children for the ones that can be used to insert the triangle it. Only subdivide is big enough
	float xSize = bbox.xmax - bbox.xmin;
	float ySize = bbox.ymax - bbox.ymin;
	float zSize = bbox.zmax - bbox.zmin;
	if (xSize > 4 || ySize > 4 || zSize > 4)
	{
		for (int i = 0; i < CHILD_COUNT; ++i)
		{
			BoundingBox childBox = getBoundingBoxForChildIndex(i);
			if (childBox.containsFully(tbb))
			{
				if (!children[i])
				{
					children[i] = std::make_unique<OctreeNode>();
					children[i]->bbox = childBox;
				}
				if (children[i]->tryAddTriangle(modelIndex, triangleIndex, rend)) return true;
			}
		}
	}

	//no children containing fully, but this one does, so put here
	OctreeContent c;
	c.modelIndex = modelIndex;
	c.triangleIndex = triangleIndex;
	this->contents.push_back(c);
	return true;
}

RayCasting::BoundingBox::BoundingBox(const Triangle& t)
{
	xmin = ymin = zmin = INFINITY;
	xmax = ymax = zmax = -INFINITY;
	for (const auto& v : t.tv)
	{
		xmin = std::min(xmin, v.space.x);
		ymin = std::min(ymin, v.space.y);
		zmin = std::min(zmin, v.space.z);
		xmax = std::max(xmax, v.space.x);
		ymax = std::max(ymax, v.space.y);
		zmax = std::max(zmax, v.space.z);
	}
}

BoundingBox RayCasting::BoundingBox::unionWith(const BoundingBox& other) const
{
	BoundingBox bb;
	bb.xmin = std::min(xmin, other.xmin);
	bb.ymin = std::min(ymin, other.ymin);
	bb.zmin = std::min(zmin, other.zmin);
	bb.xmax = std::max(xmax, other.xmax);
	bb.ymax = std::max(ymax, other.ymax);
	bb.zmax = std::max(zmax, other.zmax);
	return bb;
}

bool RayCasting::BoundingBox::intersectsWith(const BoundingBox& other) const
{
	return xmin <= other.xmax && xmax >= other.xmin
		&& ymin <= other.ymax && ymax >= other.ymin
		&& zmin <= other.zmax && zmax >= other.zmin;
}
bool RayCasting::BoundingBox::containsFully(const BoundingBox& other) const
{
	return xmin <= other.xmin && xmax >= other.xmax
		&& ymin <= other.ymin && ymax >= other.ymax
		&& zmin <= other.zmin && zmax >= other.zmax;
}
BoundingBox RayCasting::BoundingBox::infinite()
{
	BoundingBox bb;
	bb.xmin = bb.ymin = bb.zmin = -INFINITY;
	bb.xmax = bb.ymax = bb.zmax = INFINITY;
	return bb;
}
RayCasting::Octree::Octree(RayCastingRenderer& rend)
{
	this->rend = &rend;
	BoundingBox globalAABB;
	globalAABB.xmin = globalAABB.ymin = globalAABB.zmin = INFINITY;
	globalAABB.xmax = globalAABB.ymax = globalAABB.zmax = -INFINITY;
	for (const auto& model : rend.sceneModels)
		for (const auto& triangle : model.triangles)
			globalAABB = globalAABB.unionWith(triangle);

	this->root = std::make_unique<OctreeNode>();
	this->root->bbox = globalAABB;
	
	for (int modelIndex = 0; modelIndex < rend.sceneModels.size(); ++modelIndex)
	{
		for (int triangleIndex = 0; triangleIndex < rend.sceneModels[modelIndex].triangles.size(); ++triangleIndex)
		{
			if (!this->root->tryAddTriangle(modelIndex, triangleIndex, rend)) throw std::runtime_error("Failed to add triangle into Octree! Model index: " + std::to_string(modelIndex) + ", triangle index " + std::to_string(triangleIndex));
		}
	}
}

std::pair<float, float> RayCasting::BoundingBox::getMinAndMaxIntestionsFor(Vec4f rayOrigin, Vec4f rcpRayDir) const
{
	float tx1 = (xmin - rayOrigin.x) * rcpRayDir.x;
	float ty1 = (ymin - rayOrigin.y) * rcpRayDir.y;
	float tz1 = (zmin - rayOrigin.z) * rcpRayDir.z;

	float tx2 = (xmax - rayOrigin.x) * rcpRayDir.x;
	float ty2 = (ymax - rayOrigin.y) * rcpRayDir.y;
	float tz2 = (zmax - rayOrigin.z) * rcpRayDir.z;

	float tmin_x = std::min(tx1, tx2);
	float tmin_y = std::min(ty1, ty2);
	float tmin_z = std::min(tz1, tz2);

	float tmax_x = std::max(tx1, tx2);
	float tmax_y = std::max(ty1, ty2);
	float tmax_z = std::max(tz1, tz2);

	float tmin_total = std::max(std::max(0.f, tmin_z), std::max(tmin_x, tmin_y));
	float tmax_total = std::min(tmax_z, std::min(tmax_x, tmax_y));
	if (tmin_total > tmax_total) return { INFINITY, -INFINITY };
	return { tmin_total, tmax_total };
}