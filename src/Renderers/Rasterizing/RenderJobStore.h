#pragma once
#include <memory>
#include <vector>
#include "../../Vec.h"
namespace Rasterizing
{
	struct RenderJob
	{
		float x[3], y[3], z[3], u[3], v[3], nx[3], ny[3], nz[3], rcpSignedArea;
		int diffuseMapIndex;

		RenderJob() {};
	};

	struct VertexPack16;
	struct DrawCommand;
	//uses block list to store render jobs with stable pointers
	struct RenderJobStore
	{
		static constexpr size_t ELEMENTS_PER_BLOCK = 8192;
		std::vector<std::unique_ptr<RenderJob[]>> blocks;
		std::vector<size_t> elementCountInBlock;
		//size_t totalElementCount = 0;

		//std::array<VertexPack16, 3> loadVertices16(size_t firstInd, Mask16 mask) const;
		//VertexPack16 gatherVertices16(int32x16 indices) const;

		size_t size() const;
		void clear(bool forceClear = false); //sets the realSize to 0. If forceClear is true also cleans all blocks, freeing their allocated memory.
		std::pair<std::unique_ptr<RenderJob[]>&, size_t&> getInsertTarget(size_t countToInsert); //returns modifiable block reference AND modifiable size reference. Caller must adjust size if changing the block!

		//DOES NOT perform range or validity checks. May return garbage element even if index is above current total size (i.e. if cleared without deallocating, blocks will still have old data) 
		//Example: 100 elements pushed into empty store. The store is then cleared without forceClear flag. Accessing store[50] will still return old element at index 50.
		//Due to uneven size of the blocks, this operator is much slower than that of a randomly accessible container. Use with caution and prefer to use forward iterator where applicable.
		RenderJob& operator[](size_t i); 
		//void makeSpace(size_t newSize);
		void add(const VertexPack16* pStart, const VertexPack16* pEnd, const float32x16& rcpSignedArea, const int32x16& diffuseMapIndex, Mask16 activeElementsMask, const DrawCommand& subInfo);
	};
}