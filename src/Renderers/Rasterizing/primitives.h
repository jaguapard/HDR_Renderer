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

		__forceinline void gatherXYZUV(int32x16 ind, Mask16 mask, Vec4_f32x16& retXYZ, float32x16& retU, float32x16& retV) const
		{
			if (!mask) return;
			std::array<float32x16, 4> a = aos2soa_gather_and_transpose<float32x16, 4>(this->xyzp.data(), ind, mask);
			for (int i = 0; i < 3; ++i) retXYZ[i] = a[i];
			interleaved_ph_to_ps(_mm512_castps_si512(a[3]), retU, retV);
			/*
			for (int i = 0; i < 4; ++i)
			{
				float32x16 a = _mm512_permutex2var_ps(_mm512_loadu_ps(r0), _mm512_add_epi32(_mm512_set1_epi32(i), _mm512_setr_epi32(0, 16, 0, 0, 4, 20, 0, 0, 8, 24, 0, 0, 12, 28, 0, 0)), _mm512_loadu_ps(r1));
				float32x16 b = _mm512_permutex2var_ps(_mm512_loadu_ps(r2), _mm512_add_epi32(_mm512_set1_epi32(i), _mm512_setr_epi32(0, 0, 0, 16, 0, 0, 4, 20, 0, 0, 8, 24, 0, 0, 12, 28)), _mm512_loadu_ps(r3));
				float32x16 c = _mm512_mask_mov_ps(a, 0b1100110011001100, b);
				if (i < 3) retXYZ[i] = c;
				else interleaved_ph_to_ps(_mm512_castps_si512(c), retU, retV);
			}*/
		}
		//Gathers world XYZ positions for vertex indices using mask. Corresponding value in src is returned for masked out elements.
		/*
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

		__forceinline void gatherUV(int32x16 ind, Mask16 mask, float32x16& retU, float32x16& retV, float32x16 src = 0.f) const
		{
			int32x16 packedUv = _mm512_mask_i32gather_epi32(src, mask, ind, this->uvPacked.data(), 4);
			interleaved_ph_to_ps(packedUv, retU, retV);
		}*/

		__forceinline void gatherNormals(int32x16 ind, Mask16 mask, Vec4_f32x16& ret) const
		{
			int32x16 f = int32x16::gather(this->normals.data(), ind, mask);
			int32x16 xy = f & 0xFFFFFFFE; //LSB of x (LSB of mantissa) holds the sign bit of z, so discard it before conversion
			interleaved_ph_to_ps(xy, ret.x, ret.y);
			float32x16 zsq = _mm512_max_ps(float32x16(0.f), float32x16(1) - (ret.x * ret.x) - (ret.y * ret.y));
			float32x16 signless_z = zsq.sqrt();
			ret.z = _mm512_mask_mov_ps(signless_z, (f & ~0xFFFFFFFE) != 0, -signless_z);
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
		
		__forceinline void gatherVertexAndDiffuseMapIndices(int32x16 ind, Mask16 mask, int32x16& retVind0, int32x16& retVind1, int32x16& retVind2, int32x16& retDiffMapInd) const
		{
			auto a = aos2soa_gather_and_transpose<int32x16, 4>(this->vind_diffuseInd.data(), ind, mask);
			retVind0 = a[0];
			retVind1 = a[1];
			retVind2 = a[2];
			retDiffMapInd = a[3];
		}
		//loads and returns vertex indices for 16 sequential triangles, starting from startInd. Masked off elements values are not defined
		__forceinline void loadVertexAndDiffuseMapIndices16(uint32_t startInd, Mask16 mask, int32x16& retVind0, int32x16& retVind1, int32x16& retVind2, int32x16& retDiffMapInd) const
		{
			const uint32_t* p = this->vind_diffuseInd.data() + startInd * 4;

			__mmask64 m = duplicate_mmask_bits_16_to_64(mask);
			int32x16 r0 = _mm512_maskz_loadu_epi32(m, p); //v0v1v2di for triangles 0,1,2,3
			int32x16 r1 = _mm512_maskz_loadu_epi32(m >> 16, p+16); //v0v1v2di for triangles 4,5,6,7
			int32x16 r2 = _mm512_maskz_loadu_epi32(m >> 32, p+32);  //v0v1v2di for triangles 8,9,10,11
			int32x16 r3 = _mm512_maskz_loadu_epi32(m >> 48, p+48);  //v0v1v2di for triangles 12,13,14,15

			int32x16 v0v1_v0v1_x = _mm512_unpacklo_epi32(r0, r1); //v0_0, v0_4, v1_0, v1_4 | v0_1, v0_5, v1_1, v1_5 | v0_2, v0_6, v1_2, v1_6 | v0_3, v0_7, v1_3, v1_3 | 
			int32x16 v0v1_v0v1_y = _mm512_unpacklo_epi32(r2, r3); //same as above, but +8
			int32x16 v2di_v2di_x = _mm512_unpackhi_epi32(r0, r1); //v2di for 8 triangles: 0,4,1,5,2,6,3,7
			int32x16 v2di_v2di_y = _mm512_unpackhi_epi32(r2, r3); //same as above, but +8
			int32x16 v0_a = _mm512_unpacklo_epi64(v0v1_v0v1_x, v0v1_v0v1_y); //v0: 0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15
			int32x16 v1_a = _mm512_unpackhi_epi64(v0v1_v0v1_x, v0v1_v0v1_y); //v1: 0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15
			int32x16 v2_a = _mm512_unpacklo_epi64(v2di_v2di_x, v2di_v2di_y);
			int32x16 di_a = _mm512_unpackhi_epi64(v2di_v2di_x, v2di_v2di_y);
			retVind0 = _mm512_permutexvar_epi32(int32x16(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15), v0_a);
			retVind1 = _mm512_permutexvar_epi32(int32x16(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15), v1_a);
			retVind2 = _mm512_permutexvar_epi32(int32x16(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15), v2_a);
			retDiffMapInd = _mm512_permutexvar_epi32(int32x16(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15), di_a);
		}
	private:
		std::vector<uint32_t> vind_diffuseInd;
	};
}