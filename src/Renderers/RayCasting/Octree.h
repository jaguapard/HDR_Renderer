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
		std::unique_ptr<OctreeNode> children[CHILD_COUNT] = { nullptr };

		bool isLeafNode() const;
		//std::vector<OctreeContent> contents;

		BoundingBox getBoundingBoxForChildIndex(int i) const;
		bool tryAddTriangle(int modelIndex, int triangleIndex, const RayCastingRenderer& rend);


		struct ContentIterator
		{
			friend struct OctreeNode;
			ContentIterator(OctreeNode* p, bool readOnly = true) 
			{ 
				parent = p;
				this->readOnly = readOnly; 
			}
			OctreeContent* getAndIncrement()
			{
				if (ind >= parent->content.size())
				{
					size_t vi = ind - parent->content.size();
					if (!parent->extendedContent || vi >= parent->extendedContent->size()) return nullptr;
					++ind;
					return std::addressof((*parent->extendedContent)[vi]);
				}
				if (parent->content[ind].isEmpty()) return nullptr;
				return &parent->content[ind++];
			}
		private:
			size_t ind = 0;
			OctreeNode* parent;
			bool readOnly;
		};
	private:
		std::array<OctreeContent, 7> content;
		std::vector<OctreeContent>* extendedContent = nullptr;
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