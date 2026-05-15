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
		this->xyz.push_back(FixedPoint::encode_3pack(x, y, z));
		__m128 f32 = _mm_setr_ps(u, v, nx, ny);
		__m128i f16 = _mm_cvtps_ph(f32, _MM_FROUND_TO_NEAREST_INT);
		int32_t nx_fp16 = _mm_extract_epi16(f16, 2);
		int32_t ny_fp16 = _mm_extract_epi16(f16, 3);
		nx_fp16 &= 0xFFFE; //steal lowest mantissa bit for z sign
		if (nz < 0) nx_fp16 |= 1;
		this->uvs.push_back(std::bit_cast<float>(_mm_extract_epi32(f16, 0)));
		this->normals.push_back(std::bit_cast<float>(nx_fp16 | (ny_fp16 << 16)));

		__m512i bcst = _mm512_set1_epi64(this->xyz.back());
		std::array<__m512,3> a = FixedPoint::decode_3pack(bcst, bcst);
		float recoveredX = float32x16(a[0])[0];
		float recoveredY = float32x16(a[1])[0];
		float recoveredZ = float32x16(a[2])[0];
		assert(std::max(x, recoveredX) - std::min(x, recoveredX) <= 1);
		assert(std::max(y, recoveredY) - std::min(y, recoveredY) <= 1);
		assert(std::max(z, recoveredZ) - std::min(z, recoveredZ) <= 1);
		return ret;
	}
	return it->second;
}


size_t Rasterizing::VertexStore::size() const
{
	//assert(xyzp.size() % 4 == 0);
	return this->xyz.size();
}

void Rasterizing::VertexStore::reserve(size_t newSize)
{
	this->xyz.reserve(newSize);
	this->uvs.reserve(newSize);
	this->normals.reserve(newSize);
}

void Rasterizing::VertexStore::clear()
{
	this->dedup.clear();
	this->xyz.clear();
	this->uvs.clear();
	this->normals.clear();
}

void Rasterizing::TriangleStore::insert(uint32_t v0, uint32_t v1, uint32_t v2, uint32_t diffuseMapIndex, uint32_t modelIndex, ModelFlags modelFlags)
{
	this->insert(v0, v1, v2, diffuseMapIndex);
	this->modelFlags.push_back(modelFlags);
	this->modelIndex.push_back(modelIndex);
}

void Rasterizing::TriangleStore::insert(uint32_t v0, uint32_t v1, uint32_t v2, uint32_t diffuseMapIndex)
{
	this->vind_diffuseInd.push_back(v0);
	this->vind_diffuseInd.push_back(v1);
	this->vind_diffuseInd.push_back(v2);
	this->vind_diffuseInd.push_back(diffuseMapIndex);
}

void Rasterizing::TriangleStore::setDiffuseMapIndex(uint32_t triangleIndex, uint32_t diffuseMapIndex)
{
	this->vind_diffuseInd[triangleIndex * 4 + 3] = diffuseMapIndex;
}

uint32_t Rasterizing::TriangleStore::getDiffuseMapIndex(uint32_t triangleIndex)
{
	return this->vind_diffuseInd[triangleIndex * 4 + 3];
}

size_t Rasterizing::TriangleStore::size() const
{
	return this->vind_diffuseInd.size() / 4;
}

void Rasterizing::TriangleStore::clear()
{
	this->vind_diffuseInd.clear();
	this->modelFlags.clear();
	this->modelIndex.clear();
}