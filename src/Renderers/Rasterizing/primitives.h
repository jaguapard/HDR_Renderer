#pragma once
#include "../../helpers.h"
#include "../../Vec.h"
#include <vector>
#include "CoordinateTransformer.h"
#include "RenderJobStore.h"
#include <map>
#include "BufferZoneManager.h"

namespace Rasterizing
{
	enum class ShadingMode
	{
		SMOOTH,
		FLAT,
		NONE,
		COUNT,
	};

	struct VertexPack16
	{
		Vec4_f32x16 space, normal;
		float32x16 u, v;
		//VertexPack16(float32x16 x, )
		VertexPack16() {};
		static __forceinline VertexPack16 lerpVertices(const VertexPack16& from, const VertexPack16& to, const float32x16& alpha)
		{
			VertexPack16 ret;
			ret.space.x = lerp(from.space.x, to.space.x, alpha);
			ret.space.y = lerp(from.space.y, to.space.y, alpha);
			ret.space.z = lerp(from.space.z, to.space.z, alpha);
			ret.normal.x = lerp(from.normal.x, to.normal.x, alpha);
			ret.normal.y = lerp(from.normal.y, to.normal.y, alpha);
			ret.normal.z = lerp(from.normal.z, to.normal.z, alpha);
			ret.normal /= ret.normal.len3d();
			ret.u = lerp(from.u, to.u, alpha);
			ret.v = lerp(from.v, to.v, alpha);
			return ret;
		}

		static __forceinline VertexPack16 maskMove(const VertexPack16& zero, const VertexPack16& one, Mask16 mask)
		{
			VertexPack16 ret;
			ret.space.x = _mm512_mask_mov_ps(zero.space.x, mask, one.space.x);
			ret.space.y = _mm512_mask_mov_ps(zero.space.y, mask, one.space.y);
			ret.space.z = _mm512_mask_mov_ps(zero.space.z, mask, one.space.z);
			ret.normal.x = _mm512_mask_mov_ps(zero.normal.x, mask, one.normal.x);
			ret.normal.y = _mm512_mask_mov_ps(zero.normal.y, mask, one.normal.y);
			ret.normal.z = _mm512_mask_mov_ps(zero.normal.z, mask, one.normal.z);
			ret.u = _mm512_mask_mov_ps(zero.u, mask, one.u);
			ret.v = _mm512_mask_mov_ps(zero.v, mask, one.v);
			return ret;
		}

		static __forceinline VertexPack16 lerpToClippingZ(const VertexPack16& from, const VertexPack16& to, const float32x16& clippingZ)
		{
			float32x16 alpha = (clippingZ - from.space.z) / (to.space.z - from.space.z);
			return VertexPack16::lerpVertices(from, to, alpha);
		}
	};

	struct DrawCommand;

	enum class FaceCullingType
	{
		NONE,
		BACKFACE,
		FRONTFACE,
		COUNT,
	};

	struct GenericBuffer
	{
		void* data = nullptr;
		int w, h;
		GenericBuffer(void* p, int w, int h) : data(p), w(w), h(h) {};
	};

	enum class DrawRecipe
	{
		MAIN_DEPTH_PREPASS,
		SHADOW_MAP_DEPTH,
	};

	typedef BlockStore<int, 65536> TriangleIndexStore;
	struct DrawCommand
	{
		CoordinateTransformer ctr;
		bool needsUVs = true, needsNormals = true;
		FaceCullingType faceCullingType = FaceCullingType::NONE;
		Rasterizing::ShadingMode shadingMode = Rasterizing::ShadingMode::SMOOTH;

		std::vector<TriangleIndexStore>* trianglesToZones = nullptr;
		std::vector<GenericBuffer> buffers;
		uint32_t renderW, renderH;
		uint32_t threadCount = -1;
		DrawRecipe recipe;
		BufferZoneManager zoneManager;
	};

	struct VertexStore
	{
		std::vector<float> x, y, z, nx, ny, nz;
		std::vector<uint32_t> uvPacked;
		uint32_t insert(float x, float y, float z, float u, float v, float nx, float ny, float nz);
		size_t size() const;
		void clear();

		//Gathers world XYZ positions for vertex indices using mask. Corresponding value in src is returned for masked out elements.
		__forceinline void gatherWorldXYZ(int32x16 ind, Mask16 mask, float32x16& retX, float32x16& retY, float32x16& retZ, float32x16 src = 0.f) const
		{
			retX = _mm512_mask_i32gather_ps(src, mask, ind, this->x.data(), 4);
			retY = _mm512_mask_i32gather_ps(src, mask, ind, this->y.data(), 4);
			retZ = _mm512_mask_i32gather_ps(src, mask, ind, this->z.data(), 4);
		}

		__forceinline void gatherWorldXYZ(int32x16 ind, Mask16 mask, Vec4_f32x16& ret, float32x16 src = 0.f) const
		{
			this->gatherWorldXYZ(ind, mask, ret.x, ret.y, ret.z, src);
		}

		__forceinline void gatherUV(int32x16 ind, Mask16 mask, float32x16& retU, float32x16& retV) const
		{
			int32x16 packedUv = _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), mask, ind, this->uvPacked.data(), 4);
			interleaved_ph_to_ps(packedUv, retU, retV);
		}


	private:
		//TODO: if gonna make this dynamic, make it cleanable and check
		std::map<std::tuple<float, float, float, float, float, float, float, float>, uint32_t> dedup;
	};

	enum ModelFlags : uint32_t
	{
		NONE = 0,
		NO_BACKFACE_CULLING = 1 << 0,
		NO_FRONTFACE_CULLING = 1 << 1,
	};
	struct TriangleStore
	{
		std::vector<uint32_t> vertInd[3];
		std::vector<int> diffuseMapIndex, modelIndex;
		std::vector<ModelFlags> modelFlags;
		size_t size() const;
		void clear();
		//std::vector<uint32_t> modelInd;
	};
}