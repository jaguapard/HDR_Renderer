#include "Octree.h"
#include "RayCastingRenderer.h"

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
	BoundingBox tbb = { t.tv[0].space,t.tv[1].space,t.tv[2].space };
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

RayCasting::Octree::Octree(RayCastingRenderer& rend)
{
	this->rend = &rend;
	BoundingBox globalAABB;
	globalAABB.xmin = globalAABB.ymin = globalAABB.zmin = INFINITY;
	globalAABB.xmax = globalAABB.ymax = globalAABB.zmax = -INFINITY;
	for (const auto& model : rend.sceneModels)
		for (const auto& triangle : model.triangles)
			globalAABB = globalAABB.unionWith({ triangle.tv[0].space, triangle.tv[1].space, triangle.tv[2].space });

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