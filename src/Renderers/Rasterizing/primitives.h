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

	class VertexStore
	{
	public:
		uint32_t insert(float x, float y, float z, float u, float v, float nx, float ny, float nz);
		size_t size() const;
		void clear();
		void reserve(size_t newSize);

		//Loads 16 16-byte elements from base using mask and ind, then transposes and stores them into 4 SoA vectors.
		//If an element is masked off, it is not loaded and it's returned value is undefined.
		//A, B, C and D must be 64-byte types.
		//for i in [0,15]:
		//    if mask[i]:
		//        tmp = load contigious 16 bytes starting at byte size_t(base) + ind[i]*16
		//        ret1[i*4..i*4+3] = tmp[0..3]
		//        ret2[i*4..i*4+3] = tmp[4..7]
		//        ret3[i*4..i*4+3] = tmp[8..11]
		//        ret4[i*4..i*4+3] = tmp[12..15]
		template <typename A, typename B, typename C, typename D>
		requires (sizeof(A) == 64 && sizeof(B) == 64 && sizeof(C) == 64 && sizeof(D) == 64)
		__forceinline void masked_16x4aos_to_4x16soa_gather_and_transpose(int32x16 ind, Mask16 mask, const void* base, A& ret1, B& ret2, C& ret3, D& ret4) const
		{
			ind *= 4;
			uint32_t* uind = (uint32_t*)&ind;
			const float* fp = (const float*)base;
			double* r0 = reinterpret_cast<double*>(&ret1), * r1 = reinterpret_cast<double*>(&ret2), * r2 = reinterpret_cast<double*>(&ret3), * r3 = reinterpret_cast<double*>(&ret4);
			__mmask32 m = duplicate_mmask_bits_16_to_32(mask);
			for (int i = 0; i < 16; i += 4)
			{
				__m128 v0 = _mm_castpd_ps(_mm_maskz_loadu_pd(m >> (i * 2), fp + uind[i])); //abcd0
				__m128 v1 = _mm_castpd_ps(_mm_maskz_loadu_pd(m >> (i * 2 + 2), fp + uind[i + 1])); //abcd1
				__m128 v2 = _mm_castpd_ps(_mm_maskz_loadu_pd(m >> (i * 2 + 4), fp + uind[i + 2])); //abcd2
				__m128 v3 = _mm_castpd_ps(_mm_maskz_loadu_pd(m >> (i * 2 + 6), fp + uind[i + 3])); //abcd3

				__m128 aabb01 = _mm_unpacklo_ps(v0, v1);
				__m128 aabb23 = _mm_unpacklo_ps(v2, v3);
				__m128 ccdd01 = _mm_unpackhi_ps(v0, v1);
				__m128 ccdd23 = _mm_unpackhi_ps(v2, v3);
				_mm_storeu_pd(r0+i/2, _mm_unpacklo_pd(_mm_castps_pd(aabb01), _mm_castps_pd(aabb23)));
				_mm_storeu_pd(r1+i/2, _mm_unpackhi_pd(_mm_castps_pd(aabb01), _mm_castps_pd(aabb23)));
				_mm_storeu_pd(r2+i/2, _mm_unpacklo_pd(_mm_castps_pd(ccdd01), _mm_castps_pd(ccdd23)));
				_mm_storeu_pd(r3+i/2, _mm_unpackhi_pd(_mm_castps_pd(ccdd01), _mm_castps_pd(ccdd23)));
			}
		}
		__forceinline void gatherXYZUV(int32x16 ind, Mask16 mask, Vec4_f32x16& retXYZ, float32x16& retU, float32x16& retV) const
		{
			__m512 pppp;
			masked_16x4aos_to_4x16soa_gather_and_transpose(ind, mask, this->xyzp.data(), retXYZ.x, retXYZ.y, retXYZ.z, pppp);
			interleaved_ph_to_ps(_mm512_castps_si512(pppp), retU, retV);
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

		__forceinline void gatherNormals(int32x16 ind, Mask16 mask, Vec4_f32x16& ret, float32x16 src = 0.f) const
		{
			ret.x = _mm512_mask_i32gather_ps(src, mask, ind, this->nx.data(), 4);
			ret.y = _mm512_mask_i32gather_ps(src, mask, ind, this->ny.data(), 4);
			ret.z = _mm512_mask_i32gather_ps(src, mask, ind, this->nz.data(), 4);
		}
	private:
		std::vector<float> xyzp, nx, ny, nz; //xyzp = world coords + packed uv's
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