#pragma once
#include "../../BoundingBox.h"
#include <memory>
#include <vector>
class RayCastingRenderer;
namespace RayCasting
{
	struct OctreeContent
	{
		int modelIndex = -1, triangleIndex = -1;
		bool isEmpty() const;
	};
	struct OctreeNode
	{
		static constexpr int CHILD_COUNT = 8;
		BoundingBox bbox;

		bool isLeafNode() const;
		//std::vector<OctreeContent> contents;

		BoundingBox getBoundingBoxForChildIndex(int i) const;
		bool tryAddTriangle(int modelIndex, int triangleIndex, const RayCastingRenderer& rend);

		//Returns a pointer to non-empty content of this node at index i, or nullptr if that element is empty or doesn't exist
		const OctreeContent* getContentOrNull(size_t i) const;

		//Appends an empty content element to this node and returns a reference to it. 
		OctreeContent& appendContent();

		//Gets index'th child of this node. Does not perform bounds checks, index >= CHILD_COUNT is undefined behavior.
		OctreeNode* getChild(size_t index);

		std::pair<OctreeNode&, bool> getOrCreateChild(size_t index);
	private:
		std::array<OctreeContent, 7> content;
		std::vector<OctreeContent>* extendedContent = nullptr;
		std::unique_ptr<OctreeNode> children[CHILD_COUNT] = { nullptr };
	};

	class Octree
	{
	public:
		Octree() = default;
		Octree(RayCastingRenderer& rend);
		std::unique_ptr<OctreeNode> root = nullptr;
	private:
		RayCastingRenderer* rend;
	};
}