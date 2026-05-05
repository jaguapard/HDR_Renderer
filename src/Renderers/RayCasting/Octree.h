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

		//Returns a pointer to non-empty content of this node at index i, or nullptr if that element is empty or doesn't exist
		const OctreeContent* getContentOrNull(size_t i) const
		{
			if (i >= content.size())
			{
				if (!extendedContent) return nullptr;
				size_t vi = i - content.size();
				if (vi >= this->extendedContent->size()) return nullptr;
				return std::addressof((*extendedContent)[vi]);
			}
			const OctreeContent* c = &content[i];
			if (c->isEmpty()) return nullptr;
			return c;
		}

		//Appends an empty content element to this node and returns a reference to it. 
		OctreeContent& appendContent()
		{
			for (int i = 0; i < content.size(); ++i)
			{
				if (content[i].isEmpty()) return content[i];
			}
			if (!extendedContent) extendedContent = new std::vector<OctreeContent>;
			return extendedContent->emplace_back();
		}
		OctreeContent* getContentElement(size_t i, bool returnNullForEmpty = true)
		{
			if (i >= content.size())
			{
				if (!extendedContent)
				{
					if (returnNullForEmpty) return nullptr;
					extendedContent = new std::vector<OctreeContent>;
				}
				size_t vi = i - content.size();
				if (extendedContent->size() <= vi)
				{
					if (returnNullForEmpty) return nullptr;
					this->extendedContent->resize(vi + 1);
				}
				return std::addressof((*extendedContent)[vi]);
			}
			
			OctreeContent* c = &content[i];
			if (c->isEmpty()) return nullptr;
			return c;
		}
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