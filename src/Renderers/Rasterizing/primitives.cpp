#include "primitives.h"

uint32_t Rasterizing::Vertice_Store::insert(float x, float y, float z, float u, float v, float nx, float ny, float nz)
{
	assert(this->dedup.size() == this->x.size());
	auto t = std::make_tuple(x, y, z, u, v, nx, ny, nz);
	auto it = this->dedup.find(t);
	uint32_t ret;
	if (it == this->dedup.end())
	{
		this->dedup[t] = ret = this->dedup.size();
		this->x.push_back(x);
		this->y.push_back(y);
		this->z.push_back(z);
		__m128 uv_f32 = _mm_setr_ps(u, v, u, u);
		__m128i uv_f16 = _mm_cvtps_ph(uv_f32, 0);
		this->uvPacked.push_back(_mm_extract_epi32(uv_f16, 0));
		this->nx.push_back(nx);
		this->ny.push_back(ny);
		this->nz.push_back(nz);
		return ret;
	}
	return it->second;
}


size_t Rasterizing::Vertice_Store::size() const
{
	return x.size();
}

void Rasterizing::Vertice_Store::clear()
{
	this->dedup.clear();
	this->x.clear();
	this->y.clear();
	this->z.clear();
	this->uvPacked.clear();
	this->nx.clear();
	this->ny.clear();
	this->nz.clear();
}

size_t Rasterizing::Triangle_Store::size() const
{
	return diffuseMapIndex.size();
}

void Rasterizing::Triangle_Store::clear()
{
	for (auto& it : this->vertInd) it.clear();
	this->diffuseMapIndex.clear();
	this->modelFlags.clear();
	this->modelIndex.clear();
}