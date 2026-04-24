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

		__forceinline void gatherXYZUV(int32x16 ind, Mask16 mask, Vec4_f32x16& retXYZ, float32x16& retU, float32x16& retV) const
		{
			float32x16 src = 0.f;
			float rx[16], ry[16], rz[16], ru[16];
			int32x16 rawInd = ind * 4;
			uint32_t* rawIndUnsigned = (uint32_t*)&rawInd;
			const float* p = this->xyzp.data();
			for (int i = 0; i < 16; i += 4)
			{
				//xmmj = xyzp for vertex i+j
				Mask16 m = mask.mask & (1 << i) ? 15 : 0;
				__m128 v0 = _mm_maskz_loadu_epi32(m, p + rawIndUnsigned[i]);
				__m128 v1 = _mm_maskz_loadu_epi32(m, p + rawIndUnsigned[i + 1]);
				__m128 v2 = _mm_maskz_loadu_epi32(m, p + rawIndUnsigned[i + 2]);
				__m128 v3 = _mm_maskz_loadu_epi32(m, p + rawIndUnsigned[i + 3]);

				//_mm_shuffle_ps(xmm0, xmm1, _MM_SHUFFLE(0, 0, 0, 0));
				//now need to transpose the attributes. I.e. x = xmm0[0], xmm1[0], xmm2[0], xmm3[0], y = [1], z[2], uv=[3]. Relying heavily on compiler opitmizations here
				//operating on ints because _mm_extract_ps returns int lmao, this is just more explicit and less footgun-ey way to do it
				__m128 t0 = _mm_unpacklo_ps(v0, v1); // x0 x1 y0 y1
				__m128 t1 = _mm_unpackhi_ps(v0, v1); // z0 z1 uv0 uv1
				__m128 t2 = _mm_unpacklo_ps(v2, v3); // x2 x3 y2 y3
				__m128 t3 = _mm_unpackhi_ps(v2, v3); // z2 z3 uv2 uv3

				// Step 2: final assemble
				__m128 x = _mm_movelh_ps(t0, t2);   // x0 x1 x2 x3
				__m128 y = _mm_movehl_ps(t2, t0);   // y0 y1 y2 y3
				__m128 z = _mm_movelh_ps(t1, t3);   // z0 z1 z2 z3
				__m128 uv = _mm_movehl_ps(t3, t1);   // uv0 uv1 uv2 uv3

				_mm_storeu_ps(&rx[i], x);
				_mm_storeu_ps(&ry[i], y);
				_mm_storeu_ps(&rz[i], z);
				_mm_storeu_ps(&ru[i], uv);
			}
			
			/*
			retXYZ.x = _mm512_mask_i32gather_ps(src, mask, _mm512_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60), tmp.data(), 4);
			retXYZ.y = _mm512_mask_i32gather_ps(src, mask, _mm512_setr_epi32(1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61), tmp.data(), 4);
			retXYZ.z = _mm512_mask_i32gather_ps(src, mask, _mm512_setr_epi32(2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62), tmp.data(), 4);
			int32x16 packedUV = _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), mask, _mm512_setr_epi32(3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63), tmp.data(), 4);*/
			retXYZ.x = rx;
			retXYZ.y = ry;
			retXYZ.z = rz;
			interleaved_ph_to_ps(_mm512_load_ps(ru), retU, retV);
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