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
	template <typename T>
	struct ZonedBuffer
	{
		BufferZoneManager manager;
		T* data = nullptr;
	};
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
		float* zBuffer = nullptr;
		uint64_t* frameBuffer = nullptr;
		uint32_t* triangleIndexBuffer = nullptr;
		uint32_t renderW, renderH;
		uint32_t threadCount = -1;
		DrawRecipe recipe;
		std::atomic<size_t> lastClaimedTriangle = 0;
	};

	struct Vertice_Store
	{
		std::vector<float> x, y, z, u, v, nx, ny, nz;
		uint32_t insert(float x, float y, float z, float u, float v, float nx, float ny, float nz);
		size_t size() const;
		void clear();
		void resize(size_t newSize)
		{
			this->x.resize(newSize);
			this->y.resize(newSize);
			this->z.resize(newSize);
			this->u.resize(newSize);
			this->v.resize(newSize);
			this->nx.resize(newSize);
			this->ny.resize(newSize);
			this->nz.resize(newSize);
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
	struct Triangle_Store
	{
		std::vector<uint32_t> vertInd[3];
		std::vector<int> diffuseMapIndex, modelIndex;
		std::vector<ModelFlags> modelFlags;
		size_t size() const;
		void clear();
		//std::vector<uint32_t> modelInd;
	};
}