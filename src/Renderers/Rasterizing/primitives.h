#pragma once
#include "../../helpers.h"
#include "../../Vec.h"
#include <vector>
#include "CoordinateTransformer.h"
#include "RenderJobStore.h"
#include <map>
#include "BufferZoneManager.h"
#include "../../aos2soa.h"
#include "../../Threadpool.h"

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
			ret.normal /= ret.normal.len<3>();
			ret.u = lerp(from.u, to.u, alpha);
			ret.v = lerp(from.v, to.v, alpha);
			return ret;
		}

		static __forceinline VertexPack16 maskMove(const VertexPack16& zero, const VertexPack16& one, mask16d mask)
		{
			VertexPack16 ret;
			ret.space.x = mask_mov(zero.space.x, mask, one.space.x);
			ret.space.y = mask_mov(zero.space.y, mask, one.space.y);
			ret.space.z = mask_mov(zero.space.z, mask, one.space.z);
			ret.normal.x = mask_mov(zero.normal.x, mask, one.normal.x);
			ret.normal.y = mask_mov(zero.normal.y, mask, one.normal.y);
			ret.normal.z = mask_mov(zero.normal.z, mask, one.normal.z);
			ret.u = mask_mov(zero.u, mask, one.u);
			ret.v = mask_mov(zero.v, mask, one.v);
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
		uint32_t renderW, renderH;
		uint32_t threadCount = -1;
		DrawRecipe recipe;
		BufferZoneManager zoneManager;
	};

	class VertexStore
	{
	public:
		uint32_t insert(float x, float y, float z, float u, float v, float nx, float ny, float nz);
		size_t size() const;
		void clear();
		void reserve(size_t newSize);

		__forceinline void gatherXYZUV(int32x16 ind, mask16d mask, Vec4_f32x16& retXYZ, float32x16& retU, float32x16& retV) const
		{
			if (!mask) return;
			std::array<float32x16, 4> a = aos2soa_gather_and_transpose<float32x16, 4>(this->xyzp.data(), ind, mask);
			for (int i = 0; i < 3; ++i) retXYZ[i] = a[i];
			interleaved_ph_to_ps(vcast<u32x16>(a[3]), retU, retV);
		}

		__forceinline void gatherNormals(int32x16 ind, mask16d mask, Vec4_f32x16& ret) const
		{
			int32x16 f = gather<i32x16>(this->normals.data(), ind, mask);
			int32x16 xy = f & 0xFFFFFFFE; //LSB of x (LSB of mantissa) holds the sign bit of z, so discard it before conversion
			interleaved_ph_to_ps(xy, ret.x, ret.y);
			float32x16 zsq = max(float32x16(0.f), float32x16(1) - (ret.x * ret.x) - (ret.y * ret.y));
			float32x16 signless_z = sqrtf(zsq);
			ret.z = mask_mov(signless_z, (f & ~0xFFFFFFFE) != 0, -signless_z);
		}
	private:
		std::vector<float> xyzp, normals; //xyzp = world coords + packed uv's
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
		void insert(uint32_t v0, uint32_t v1, uint32_t v2, uint32_t diffuseMapIndex, uint32_t modelIndex, ModelFlags modelFlags);
		void insert(uint32_t v0, uint32_t v1, uint32_t v2, uint32_t diffuseMapIndex);
		void setDiffuseMapIndex(uint32_t triangleIndex, uint32_t diffuseMapIndex);
		uint32_t getDiffuseMapIndex(uint32_t triangleIndex);
		std::vector<int> modelIndex;
		std::vector<ModelFlags> modelFlags;
		size_t size() const;
		void clear();
		//std::vector<uint32_t> modelInd;

		template<size_t N>
		__forceinline void gatherVertexAndDiffuseMapIndices(const SIMD_Vector<int32_t, N>& ind, const mask_t<int32_t, N>& mask, SIMD_Vector<int32_t, N>& retVind0, SIMD_Vector<int32_t, N>& retVind1, SIMD_Vector<int32_t, N>& retVind2, SIMD_Vector<int32_t, N>& retDiffMapInd) const
		{
			auto a = aos2soa_gather_and_transpose<SIMD_Vector<int32_t, N>, 4>(this->vind_diffuseInd.data(), ind, mask);
			retVind0 = a[0];
			retVind1 = a[1];
			retVind2 = a[2];
			retDiffMapInd = a[3];
		}
		//loads and returns vertex indices for 16 sequential triangles, starting from startInd. Masked off elements values are not defined
		__forceinline void loadVertexAndDiffuseMapIndices16(uint32_t startInd, mask16d mask, int32x16& retVind0, int32x16& retVind1, int32x16& retVind2, int32x16& retDiffMapInd) const
		{
			const uint32_t* p = this->vind_diffuseInd.data() + startInd * 4;

			__mmask64 m = duplicate_mmask_bits_16_to_64(mask);
			int32x16 r0 = load<i32x16>(p, m); //v0v1v2di for triangles 0,1,2,3
			int32x16 r1 = load<i32x16>(p + 16, m >> 16); //v0v1v2di for triangles 4,5,6,7
			int32x16 r2 = load<i32x16>(p + 32, m >> 32);  //v0v1v2di for triangles 8,9,10,11
			int32x16 r3 = load<i32x16>(p + 48, m >> 48);  //v0v1v2di for triangles 12,13,14,15

			u64x8 v0v1_v0v1_x = vcast<u64x8>(unpacklo(r0, r1)); //v0_0, v0_4, v1_0, v1_4 | v0_1, v0_5, v1_1, v1_5 | v0_2, v0_6, v1_2, v1_6 | v0_3, v0_7, v1_3, v1_3 | 
			u64x8 v0v1_v0v1_y = vcast<u64x8>(unpacklo(r2, r3)); //same as above, but +8
			u64x8 v2di_v2di_x = vcast<u64x8>(unpackhi(r0, r1)); //v2di for 8 triangles: 0,4,1,5,2,6,3,7
			u64x8 v2di_v2di_y = vcast<u64x8>(unpackhi(r2, r3)); //same as above, but +8
			int32x16 v0_a = vcast<i32x16>(unpacklo(v0v1_v0v1_x, v0v1_v0v1_y)); //v0: 0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15
			int32x16 v1_a = vcast<i32x16>(unpackhi(v0v1_v0v1_x, v0v1_v0v1_y)); //v1: 0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15
			int32x16 v2_a = vcast<i32x16>(unpacklo(v2di_v2di_x, v2di_v2di_y));
			int32x16 di_a = vcast<i32x16>(unpackhi(v2di_v2di_x, v2di_v2di_y));
			retVind0 = permx(v0_a, i32x16(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15));
			retVind1 = permx(v1_a, i32x16(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15));
			retVind2 = permx(v2_a, i32x16(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15));
			retDiffMapInd = permx(di_a, i32x16(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15));
		}
	private:
		std::vector<uint32_t> vind_diffuseInd;
	};
}