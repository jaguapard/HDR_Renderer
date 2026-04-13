#include "RenderJobStore.h"
#include "primitives.h"

using namespace Rasterizing;
std::pair<std::unique_ptr<RenderJob[]>&, size_t&> Rasterizing::RenderJobStore::getInsertTarget(size_t countToInsert)
{
	assert(countToInsert <= ELEMENTS_PER_BLOCK);
	if (this->elementCountInBlock.empty()) this->elementCountInBlock.push_back(0);
	
	size_t currBlockOccupiedCount = this->elementCountInBlock.back();
	if (currBlockOccupiedCount + countToInsert > ELEMENTS_PER_BLOCK) this->elementCountInBlock.push_back(0);
	
	size_t blockIndexToReturn = this->elementCountInBlock.size() - 1;
	while (this->blocks.size() <= blockIndexToReturn) this->blocks.emplace_back(std::make_unique<RenderJob[]>(ELEMENTS_PER_BLOCK));
	return std::make_pair(std::ref(this->blocks[blockIndexToReturn]), std::ref(this->elementCountInBlock.back()));
}

RenderJob& Rasterizing::RenderJobStore::operator[](size_t i)
{
	assert(i < this->size());
	size_t passed = 0, j = 0;
	while (true)
	{
		if (j >= this->elementCountInBlock.size()) break;
		size_t s = this->elementCountInBlock[j];
		if (passed + s <= i)
		{
			passed += s;
			++j;
		}
		else break;
	}
	return this->blocks[j][i - passed];
}

size_t Rasterizing::RenderJobStore::size() const
{
	size_t acc = 0;
	for (auto& it : this->elementCountInBlock) acc += it;
	return acc;
}

void Rasterizing::RenderJobStore::clear(bool forceClear)
{
	this->elementCountInBlock.clear();
	if (forceClear) this->blocks.clear();
}


void Rasterizing::RenderJobStore::add(const VertexPack16* pStart, const VertexPack16* pEnd, const float32x16& rcpSignedArea, const int32x16& diffuseMapIndex, Mask16 activeElementsMask, const DrawCommand& subInfo)
{
	if (pEnd - pStart != 3) throw std::runtime_error("Vertex pack sizes not equal to 3 are not yet supported in RenderJob_Store::add");
	//TODO: handle subInfo by eliding unused stores and resizes (normals/uvs/etc) (actually, maybe better not with AoS layout, since we won't avoid much anyway)
	if (!activeElementsMask) return;
	//The next code expects exactly this layout and may break if it changes, so have strict checks for it! You'll have to tweak it when changing stuff!
	{
		static_assert(sizeof(RenderJob) == 104);
		static_assert(offsetof(RenderJob, x[0]) == 0);
		static_assert(offsetof(RenderJob, x[1]) == 4);
		static_assert(offsetof(RenderJob, x[2]) == 8);
		static_assert(offsetof(RenderJob, y[0]) == 12);
		static_assert(offsetof(RenderJob, y[1]) == 16);
		static_assert(offsetof(RenderJob, y[2]) == 20);
		static_assert(offsetof(RenderJob, z[0]) == 24);
		static_assert(offsetof(RenderJob, z[1]) == 28);
		static_assert(offsetof(RenderJob, z[2]) == 32);
		static_assert(offsetof(RenderJob, u[0]) == 36);
		static_assert(offsetof(RenderJob, u[1]) == 40);
		static_assert(offsetof(RenderJob, u[2]) == 44);
		static_assert(offsetof(RenderJob, v[0]) == 48);
		static_assert(offsetof(RenderJob, v[1]) == 52);
		static_assert(offsetof(RenderJob, v[2]) == 56);
		static_assert(offsetof(RenderJob, nx[0]) == 60);
		static_assert(offsetof(RenderJob, nx[1]) == 64);
		static_assert(offsetof(RenderJob, nx[2]) == 68);
		static_assert(offsetof(RenderJob, ny[0]) == 72);
		static_assert(offsetof(RenderJob, ny[1]) == 76);
		static_assert(offsetof(RenderJob, ny[2]) == 80);
		static_assert(offsetof(RenderJob, nz[0]) == 84);
		static_assert(offsetof(RenderJob, nz[1]) == 88);
		static_assert(offsetof(RenderJob, nz[2]) == 92);
		static_assert(offsetof(RenderJob, rcpSignedArea) == 96);
		static_assert(offsetof(RenderJob, diffuseMapIndex) == 100);
	}

	int toAddCount = _mm_popcnt_u32(activeElementsMask);
	auto [blockToInsert, pushIndex] = this->getInsertTarget(toAddCount);
	
	for (int j = 0; j < 16; ++j)
	{
		if ((activeElementsMask.mask & (1 << j)) == 0) continue;
		RenderJob& rj = blockToInsert[pushIndex++];

		int32x16 gatherInd_first = _mm512_setr_epi32(
			sizeof(VertexPack16) * 0 + offsetof(VertexPack16, space.x),
			sizeof(VertexPack16) * 1 + offsetof(VertexPack16, space.x),
			sizeof(VertexPack16) * 2 + offsetof(VertexPack16, space.x),
			sizeof(VertexPack16) * 0 + offsetof(VertexPack16, space.y),
			sizeof(VertexPack16) * 1 + offsetof(VertexPack16, space.y),
			sizeof(VertexPack16) * 2 + offsetof(VertexPack16, space.y),
			sizeof(VertexPack16) * 0 + offsetof(VertexPack16, space.z),
			sizeof(VertexPack16) * 1 + offsetof(VertexPack16, space.z),
			sizeof(VertexPack16) * 2 + offsetof(VertexPack16, space.z),
			sizeof(VertexPack16) * 0 + offsetof(VertexPack16, u),
			sizeof(VertexPack16) * 1 + offsetof(VertexPack16, u),
			sizeof(VertexPack16) * 2 + offsetof(VertexPack16, u),
			sizeof(VertexPack16) * 0 + offsetof(VertexPack16, v),
			sizeof(VertexPack16) * 1 + offsetof(VertexPack16, v),
			sizeof(VertexPack16) * 2 + offsetof(VertexPack16, v),
			sizeof(VertexPack16) * 0 + offsetof(VertexPack16, normal.x)
		);
		int32x16 gatherInd_second = _mm512_setr_epi32(
			sizeof(VertexPack16) * 1 + offsetof(VertexPack16, normal.x),
			sizeof(VertexPack16) * 2 + offsetof(VertexPack16, normal.x),
			sizeof(VertexPack16) * 0 + offsetof(VertexPack16, normal.y),
			sizeof(VertexPack16) * 1 + offsetof(VertexPack16, normal.y),
			sizeof(VertexPack16) * 2 + offsetof(VertexPack16, normal.y),
			sizeof(VertexPack16) * 0 + offsetof(VertexPack16, normal.z),
			sizeof(VertexPack16) * 1 + offsetof(VertexPack16, normal.z),
			sizeof(VertexPack16) * 2 + offsetof(VertexPack16, normal.z),
			0, 0, 0, 0, 0, 0, 0, 0
		);
		_mm512_storeu_ps(&rj, _mm512_i32gather_ps(gatherInd_first + j * 4, pStart, 1));
		_mm512_mask_storeu_ps(reinterpret_cast<__m512*>(&rj) + 1, 0xFF, _mm512_mask_i32gather_ps(_mm512_setzero_ps(), 0xFF, gatherInd_second + j * 4, pStart, 1));
		rj.diffuseMapIndex = diffuseMapIndex[j];
		rj.rcpSignedArea = rcpSignedArea[j];
	}
	//assert(this->realSize - oldSz == std::popcount(activeElementsMask.mask));
}

RenderJobStore::RenderJobStoreForwardIterator Rasterizing::RenderJobStore::getIteratorFromStart()
{
	return RenderJobStoreForwardIterator(*this, 0, 0);
}

RenderJob* Rasterizing::RenderJobStore::RenderJobStoreForwardIterator::getAndIncrement()
{
	while (this->currBlockIndex < this->parent.elementCountInBlock.size())
	{
		size_t occupiedElementsInBlock = this->parent.elementCountInBlock[this->currBlockIndex];
		if (occupiedElementsInBlock > this->currElementIndex) return &this->parent.blocks[currBlockIndex][currElementIndex++];
		this->currBlockIndex++;
		this->currElementIndex = 0;
	}
	return nullptr;
}
