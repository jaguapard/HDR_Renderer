#pragma once
#include <string>
#include <d3d11.h>
#include <vector>
#include "../C_Input.h"
#include "../Vec.h"
#include "../helpers.h"
#include "TextureManager.h"
#include "../Statsman.h"
#undef min
#undef max
struct GameSettings;

struct RendererLoadSceneData
{
	std::vector<std::pair<std::string, std::string>> files;
};
class RendererBase
{
public:
	RendererBase() = default;
	//This method is called when renderer should perform it's setup routines. The constructor is the wrong place to do it since the rest of the pipeline may not yet be ready for it's setup
	virtual void setup() {};
	virtual void loadScene(RendererLoadSceneData scd) = 0;
	virtual void renderFrame(const GameSettings& settings) = 0;
	virtual ~RendererBase() {};

	template<size_t N>
	static void mask_store_vec4_f32x16_to_framebuffer(const SIMD_VectorPack<SIMD_Vector<float, N>, 4>& pack, void* frameBuffer, int x, int y, int w, const mask_t<float, N>& mask)
	{
		//we have px[0] == r0,r1,r2...,r15, px[1] == g0,..g15, ...
		//DX wants: r0,g0,b0,a0,r1,g1,b1,a1, etc
		//Meanings, that first 16-wide register to store should be r0,g0,b0,a0,...,r3,g3,b3,a3
		//Second - 4-7, third - 8-11, fourth - 12-15
		SIMD_Vector<fp16_t, N> ph_r = pack.x;
		SIMD_Vector<fp16_t, N> ph_g = pack.y;
		SIMD_Vector<fp16_t, N> ph_b = pack.z;
		SIMD_Vector<fp16_t, N> ph_a = pack.w;

		SIMD_Vector<uint16_t, N> rg_permind, ba_permind;
		mask_t<fp16_t, N> mov_mask = 0;
		for (size_t i = 0; i < N; i += 4)
		{
			rg_permind[i] = i / 4;
			rg_permind[i + 1] = 16 + i / 4;
			rg_permind[i + 2] = rg_permind[i + 3] = 0;

			ba_permind[i] = ba_permind[i + 1] = 0;
			ba_permind[i + 2] = i / 4;
			ba_permind[i + 3] = i / 4 + 16;
			mov_mask |= 0b1100ull << i;
		}
		for (int i = 0; i < 16; i += 4)
		{
			u16x16 rg_ind = rg_permind + i;
			u16x16 ba_ind = ba_permind + i;
			SIMD_Vector<fp16_t, N> rgxx = permx2(ph_r, ph_g, rg_ind);
			SIMD_Vector<fp16_t, N> xxba = permx2(ph_b, ph_a, ba_ind);
			SIMD_Vector<fp16_t, N> rgba = mask_mov(rgxx, mov_mask, xxba);
			store(vcast<u64x4>(rgba), (int64_t*)frameBuffer + y * w + x + i, mask >> i);
		}
	}
	static Vec4_f32x16 mask_load_vec4_f32x16_from_framebuffer(const void* frameBuffer, int x, int y, int w, mask16d mask);
	
	//Calculates barycentric coordinates for 2D vector P relative to vertices A, B, C and stores them to ret
	//Only x and y values from input vectors are used for the calculations, the rest are ignored.
	template<typename VectorType, typename ValueType>
	static __forceinline void calculateBarycentricCoordinates2D(const VectorType& P, const VectorType& A, const VectorType& B, const VectorType& C, const ValueType& rcpSignedArea, std::array<ValueType, 3>& ret)
	{
		ret[0] = (P - C).cross2d(B - C) * rcpSignedArea;
		ret[1] = (P - C).cross2d(C - A) * rcpSignedArea;
		ret[2] = (P - A).cross2d(A - B) * rcpSignedArea; //do NOT change this to 1-alpha-beta or 1-(alpha+beta). That causes wonkiness in textures on big triangles
	}

	//Calculates barycentric coordinates for 2D vector P relative to vertices A, B, C and stores them in retInitials
	//Calculates steps for 1 unit movements in X and Y, and stores them into retStepsX and retStepsY
	//Stepping can be done by calculating: barycentric[i] = retInitials[i] + (x-P.x)*retStepsX[i] + (y-P.y)*retStepsY[i]
	//Only x and y values from input vectors are used for the calculations, the rest are ignored.
	template<typename VectorType, typename ValueType>
	static __forceinline void calculateBarycentricCoordinatesAndSteps2D(const VectorType& P, const VectorType& A, const VectorType& B, const VectorType& C, const ValueType& rcpSignedArea, std::array<ValueType, 3>& retInitials, std::array<ValueType, 3>& retStepsX, std::array<ValueType, 3>& retStepsY)
	{
		calculateBarycentricCoordinates2D(P, A, B, C, rcpSignedArea, retInitials);
		ValueType group_dAlpha_dx = (B.y - C.y) * rcpSignedArea;
		ValueType group_dAlpha_dy = (C.x - B.x) * rcpSignedArea;
		ValueType group_dBeta_dx = (C.y - A.y) * rcpSignedArea;
		ValueType group_dBeta_dy = (A.x - C.x) * rcpSignedArea;
		ValueType group_dGamma_dx = (A.y - B.y) * rcpSignedArea; //this should have better precision than -group_dAlpha_dx - group_dBeta_dx since y2 should cancel out completely algebraically;
		ValueType group_dGamma_dy = (B.x - A.x) * rcpSignedArea; //same for -group_dAlpha_dy - group_dBeta_dy and x2
		retStepsX[0] = group_dAlpha_dx;
		retStepsX[1] = group_dBeta_dx;
		retStepsX[2] = group_dGamma_dx;
		retStepsY[0] = group_dAlpha_dy;
		retStepsY[1] = group_dBeta_dy;
		retStepsY[2] = group_dGamma_dy;
	}

	//Calculates 3D barycentric coordinates for vector P relative to vertices A, B, C and stores them to ret
	//Only x, y and z values from input vectors are used for the calculations, the rest are ignored.
	template<typename VectorType, typename ValueType>
	static __forceinline void calculateBarycentricCoordinates3D(const VectorType& P, const VectorType& A, const VectorType& B, const VectorType& C, std::array<ValueType, 3>& ret)
	{
		/* //this version is less precise, causes texture issues in some places
		Vec4_f32x16 v0 = B - A;
		Vec4_f32x16 v1 = C - A;
		Vec4_f32x16 v2 = P - A;

		float32x16 d00 = v0.dot3d(v0);
		float32x16 d01 = v0.dot3d(v1);
		float32x16 d11 = v1.dot3d(v1);
		float32x16 d20 = v2.dot3d(v0);
		float32x16 d21 = v2.dot3d(v1);
		float32x16 den = d00 * d11 - (d01 * d01);
		beta = (d11 * d20 - d01 * d21) / den;
		gamma = (d00 * d21 - d01 * d20) / den;
		alpha = float32x16(1) - beta - gamma; //doesn't seem to hurt calculating it like this*/

		VectorType n = (B - A).cross3d(C - A);
		ret[0] = ((B - P).cross3d(C - P)).dot<3>(n) / n.dot<3>(n);
		ret[1] = ((C - P).cross3d(A - P)).dot<3>(n) / n.dot<3>(n);
		ret[2] = ((A - P).cross3d(B - P)).dot<3>(n) / n.dot<3>(n);
	}

	/*
	//TODO: verify that it works
	//Stores each 128-bit quarter of 512-bit vector to consecutive rows (increasing X indices) of row-major-indexed buffer dst using mask, starting at (xStart,yStart)
	static __forceinline void mask_store_rows_512_to_4x128_ps(const f32x16& value, __mmask16 mask, void* dst, uint32_t xStart, uint32_t yStart, uint32_t w)
	{
		f32x4 v0 = extract<0, 4>(value);
		f32x4 v1 = extract<1, 4>(value);
		f32x4 v2 = extract<2, 4>(value);
		f32x4 v3 = extract<3, 4>(value);
		float* p = (float*)dst;
		store(v0, p + yStart * w + xStart, mask);
		store(v1, p + (1 + yStart) * w + xStart, mask >> 4);
		store(v2, p + (2 + yStart) * w + xStart, mask >> 8);
		store(v3, p + (3 + yStart) * w + xStart, mask >> 12);
	}
	*/
	//TODO: verify that it works
	static __forceinline __m512 mask_load_rows_4x128_to_512_ps(__mmask16 mask, const void* src, uint32_t xStart, uint32_t yStart, uint32_t w)
	{
		const float* p = (const float*)src;
		f32x4 v0 = load<f32x4>(p + yStart * w + xStart, mask);
		f32x4 v1 = load<f32x4>(p + (yStart + 1) * w + xStart, mask >> 4);
		f32x4 v2 = load<f32x4>(p + (yStart + 2) * w + xStart, mask >> 8);
		f32x4 v3 = load<f32x4>(p + (yStart + 3) * w + xStart, mask >> 12);
		return concat(concat(v0, v1), concat(v2, v3));
	}

	//Writes out colors to pixels (x,y) of the framebuffer using mask
	static void scatterToFrameBuffer(const Vec4_f32x16& colors, int32x16 x, int32x16 y, mask16d mask, void* frameBuf, int framebufW);
protected:
	TextureManager& textureManager = TextureManager::getInstance();
};