#include "primitives.h"

uint32_t Rasterizing::VertexStore::insert(float x, float y, float z, float u, float v, float nx, float ny, float nz)
{
	//assert(this->dedup.size() == this->x.size());
	auto t = std::make_tuple(x, y, z, u, v, nx, ny, nz);
	auto it = this->dedup.find(t);
	uint32_t ret;
	if (it == this->dedup.end())
	{
		this->dedup[t] = ret = this->dedup.size();
		this->xyzp.push_back(x);
		this->xyzp.push_back(y);
		this->xyzp.push_back(z);
		__m128 uv_f32 = _mm_setr_ps(u, v, u, u);
		__m128i uv_f16 = _mm_cvtps_ph(uv_f32, 0);
		this->xyzp.push_back(std::bit_cast<float>(_mm_extract_epi32(uv_f16, 0)));
		this->nx.push_back(nx);
		this->ny.push_back(ny);
		this->nz.push_back(nz);
		size_t sz = this->xyzp.capacity() / 4;
		if (sz % 16 != 0) this->reserve(sz + 16 - sz % 16);
		return ret;
	}
	return it->second;
}


size_t Rasterizing::VertexStore::size() const
{
	assert(xyzp.size() % 4 == 0);
	return this->xyzp.size() / 4;
}

void Rasterizing::VertexStore::reserve(size_t newSize)
{
	this->xyzp.reserve(newSize * 4);
	this->nx.reserve(newSize);
	this->ny.reserve(newSize);
	this->nz.reserve(newSize);
}

void Rasterizing::VertexStore::clear()
{
	this->dedup.clear();
	this->xyzp.clear();
	this->nx.clear();
	this->ny.clear();
	this->nz.clear();
}

size_t Rasterizing::TriangleStore::size() const
{
	return diffuseMapIndex.size();
}

void Rasterizing::TriangleStore::clear()
{
	for (auto& it : this->vertInd) it.clear();
	this->diffuseMapIndex.clear();
	this->modelFlags.clear();
	this->modelIndex.clear();
}