#include "RenderJobStore.h"
#include "primitives.h"

using namespace Rasterizing;

Rasterizing::RenderJobStore::RenderJobStore(const Vertice_Store* vertStore)
{
	this->verticeStore = vertStore;
}

void Rasterizing::RenderJobStore::reset()
{
	this->lightRenderJobs.reset();
	this->heavyRenderJobs.reset();
}
void Rasterizing::RenderJobStore::addMany(const VertexPack16* pStart, const VertexPack16* pEnd, const float32x16& rcpSignedArea, const int32x16& diffuseMapIndex, Mask16 activeElementsMask, const DrawCommand& subInfo, const int32x16* vertexIndexStart, const int32x16* vertexIndexEnd, bool areClipped)
{
	if (pEnd - pStart != 3) throw std::runtime_error("Vertex pack sizes not equal to 3 are not yet supported in RenderJob_Store::add");
	//TODO: handle subInfo by eliding unused stores and resizes (normals/uvs/etc) (actually, maybe better not with AoS layout, since we won't avoid much anyway)
	if (!activeElementsMask) return;
	//The next code expects exactly this layout and may break if it changes, so have strict checks for it! You'll have to tweak it when changing stuff!
	if (areClipped)
	{
		/*
		for (int j = 0; j < 16; ++j)
		{
			if ((activeElementsMask.mask & (1 << j)) == 0) continue;
			RenderJob rj;
			for (int i = 0; i < 3; ++i)
			{
				rj.x[i] = pStart[i].space.x[j];
				rj.y[i] = pStart[i].space.y[j];
				rj.z[i] = pStart[i].space.z[j];
				rj.u[i] = pStart[i].u[j];
				rj.v[i] = pStart[i].v[j];
				rj.nx[i] = pStart[i].normal.x[j];
				rj.ny[i] = pStart[i].normal.y[j];
				rj.nz[i] = pStart[i].normal.z[j];
			}
			rj.diffuseMapIndex = diffuseMapIndex[j];
			rj.rcpSignedArea = rcpSignedArea[j];
			this->heavyRenderJobs.append(rj);
		}
		*/
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

		int toAddCount = _mm_popcnt_u32(activeElementsMask);

		for (int j = 0; j < 16; ++j)
		{
			if ((activeElementsMask.mask & (1 << j)) == 0) continue;
			RenderJob rj;

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
			this->heavyRenderJobs.append(rj);
		}
	}
	else
	{
		for (int j = 0; j < 16; ++j)
		{
			if ((activeElementsMask.mask & (1 << j)) == 0) continue;
			RenderJobLight rjLight;
			for (int k = 0; k < 3; ++k)
			{
				rjLight.screenX[k] = pStart[k].space.x[j];
				rjLight.screenY[k] = pStart[k].space.y[j];
				rjLight.rcpZ[k] = pStart[k].space.z[j];
				rjLight.vertexIndex[k] = vertexIndexStart[k][j];
			}
			rjLight.diffuseMapIndex = diffuseMapIndex[j];
			rjLight.rcpSignedArea = rcpSignedArea[j];
			this->lightRenderJobs.append(rjLight);
		}
	}
	//assert(this->realSize - oldSz == std::popcount(activeElementsMask.mask));
}

size_t Rasterizing::RenderJobStore::size() const
{
	return this->lightRenderJobs.size() + this->heavyRenderJobs.size();
}

RenderJob Rasterizing::RenderJobStore::operator[](size_t i)
{
	size_t lightCount = this->lightRenderJobs.size();
	if (i < lightCount)
	{
		RenderJob ret;
		RenderJobLight& light = this->lightRenderJobs[i];

		for (int j = 0; j < 3; ++j)
		{
			ret.x[j] = light.screenX[j];
			ret.y[j] = light.screenY[j];
			float rcpZ = light.rcpZ[j];
			ret.z[j] = rcpZ;

			uint32_t vi = light.vertexIndex[j];
			ret.u[j] = this->verticeStore->u[vi] * rcpZ;
			ret.v[j] = this->verticeStore->v[vi] * rcpZ;
			ret.nx[j] = this->verticeStore->nx[vi] * rcpZ;
			ret.ny[j] = this->verticeStore->ny[vi] * rcpZ;
			ret.nz[j] = this->verticeStore->nz[vi] * rcpZ;
		}
		ret.diffuseMapIndex = light.diffuseMapIndex;
		ret.rcpSignedArea = light.rcpSignedArea;
		return ret;
	}
	return this->heavyRenderJobs[i - lightCount];
}

/*

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
*/