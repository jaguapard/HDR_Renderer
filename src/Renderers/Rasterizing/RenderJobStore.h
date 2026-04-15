#pragma once
#include <memory>
#include <vector>
#include <optional>
#include "../../Vec.h"
#include "../../BlockStore.h"
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
	struct RenderJobStore : BlockStore<RenderJob, 8192>
	{
		//std::array<VertexPack16, 3> loadVertices16(size_t firstInd, Mask16 mask) const;
		//VertexPack16 gatherVertices16(int32x16 indices) const;

		void addMany(const VertexPack16* pStart, const VertexPack16* pEnd, const float32x16& rcpSignedArea, const int32x16& diffuseMapIndex, Mask16 activeElementsMask, const DrawCommand& subInfo);
		void addOne(const RenderJob& rj);
		
		/*
		class RenderJobStoreFrozenForwardIterator
		{
		public:
			RenderJobStoreFrozenForwardIterator(RenderJobStore& parent, size_t startBlockIndex, size_t startElementIndex) : parent(parent), currBlockIndex(startBlockIndex), currElementIndex(startElementIndex) {};
			//std::optional<RenderJob&> get();
			RenderJob* getAndIncrement();
			//void increment();
		private:
			size_t currBlockIndex, currElementIndex;
			RenderJobStore& parent;
			std::vector<RenderJob*> snapshottedBlocks;
			friend struct RenderJobStore;
		};

		//This iterator snapshots the parent's store state at the moment of it's creation and will only iterate over elements inside the store at that time (i.e. newly added elements after it's creation WILL NOT be given by this iterator)
		RenderJobStoreFrozenForwardIterator getIteratorFromStart();*/

	private:
		
	};
}