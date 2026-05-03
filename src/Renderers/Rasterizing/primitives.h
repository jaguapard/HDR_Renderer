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

		//Gathers 16 packed 4-tuples of attributes from p using mask, and transposes them into 4 SoA vectors.
		//ret[i][j] = mask[j] ? (float*)(p)[ind[j]*4+i]
		//Example: p stores 4x4 byte attributes x,y,z,w in 16-byte packs. Packs at index ind[i] are fetched and transposed into output vectors:
		//retFirst[i] = x[i], retSecond[i] = y[i], retThird[i] = z[i], retFourth[i] = w[i]
		//Requirements: each output element must be 64-bytes wide. All masked elements are in bounds of p (sizeof(*p) >= 16*max(ind))
		//Values of unmasked lanes are undefined, but they are guaranteed to not be accessed in memory
		template <typename A, typename B, typename C, typename D>
		requires (sizeof(A) == 64 && sizeof(B) == 64 && sizeof(C) == 64 && sizeof(D) == 64)
		__forceinline void masked_16x4aos_to_4x16soa_gather_and_transpose(int32x16 ind, Mask16 mask, const void* p, A& retFirst, B& retSecond, C& retThird, D& retFourth) const
		{
			ind *= 4;
			float r0[16], r1[16], r2[16], r3[16];
			uint32_t* rawIndUnsigned = (uint32_t*)&ind;
			const float* fp = (const float*)p;
			__mmask32 m = duplicate_mmask_bits_16_to_32(mask);
			for (int i = 0; i < 16; i += 4)
			{
				__m128 v0 = _mm_castpd_ps(_mm_maskz_loadu_pd(m >> (i * 2), fp + rawIndUnsigned[i]));
				__m128 v1 = _mm_castpd_ps(_mm_maskz_loadu_pd(m >> (i * 2 + 2), fp + rawIndUnsigned[i + 1]));
				__m128 v2 = _mm_castpd_ps(_mm_maskz_loadu_pd(m >> (i * 2 + 4), fp + rawIndUnsigned[i + 2]));
				__m128 v3 = _mm_castpd_ps(_mm_maskz_loadu_pd(m >> (i * 2 + 6), fp + rawIndUnsigned[i + 3]));

				//__m128 x0x1y0y1 = _mm_unpacklo_ps(v0, v1);
				//__m128 x2x3y2y3 = _mm_unpacklo_ps(v2, v3);
				//__m128 x0_4 = _mm_shuffle_ps(x0x1y0y1, x2x3y2y3, _MM_SHUFFLE(1, 0, 1, 0));
				_mm_storeu_ps(&r0[i], v0); //r0 = xyzp0,xyzp4,xyzp8,xyzp12
				_mm_storeu_ps(&r1[i], v1); //r1 = xyzp1,xyzp5,xyzp9,xyzp13
				_mm_storeu_ps(&r2[i], v2); //r2 = xyzp2,xyzp6,xyzp10,xyzp14
				_mm_storeu_ps(&r3[i], v3); //r3 = xyzp3,xyzp7,xyzp11,xyzp15
			}

			__m512 xxyy01 = _mm512_unpacklo_ps(_mm512_loadu_ps(r0), _mm512_loadu_ps(r1));
			__m512 xxyy23 = _mm512_unpacklo_ps(_mm512_loadu_ps(r2), _mm512_loadu_ps(r3));
			__m512 zzpp01 = _mm512_unpackhi_ps(_mm512_loadu_ps(r0), _mm512_loadu_ps(r1));
			__m512 zzpp23 = _mm512_unpackhi_ps(_mm512_loadu_ps(r2), _mm512_loadu_ps(r3));
			_mm512_storeu_pd(&retFirst, _mm512_unpacklo_pd(_mm512_castps_pd(xxyy01), _mm512_castps_pd(xxyy23)));
			_mm512_storeu_pd(&retSecond, _mm512_unpackhi_pd(_mm512_castps_pd(xxyy01), _mm512_castps_pd(xxyy23)));
			_mm512_storeu_pd(&retThird, _mm512_unpacklo_pd(_mm512_castps_pd(zzpp01), _mm512_castps_pd(zzpp23)));
			_mm512_storeu_pd(&retFourth, _mm512_unpackhi_pd(_mm512_castps_pd(zzpp01), _mm512_castps_pd(zzpp23)));
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