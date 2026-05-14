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
	//virtual void handleInputEvent(const SDL_Event& ev, C_Input& input) = 0;
	virtual void loadScene(RendererLoadSceneData scd) = 0;
	virtual void renderFrame(const GameSettings& settings) = 0;

	static void calculateBarycentricCoordinates3D(const Vec4_f32x16& P, const Vec4_f32x16& A, const Vec4_f32x16& B, const Vec4_f32x16& C, float32x16& alpha, float32x16& beta, float32x16& gamma);
	static void calculateBarycentricCoordinates3D(const Vec4_f32x8& P, const Vec4_f32x8& A, const Vec4_f32x8& B, const Vec4_f32x8& C, float32x8& alpha, float32x8& beta, float32x8& gamma);
	static void calculateBarycentricCoordinates3D(const Vec4_f32x16& P, const Vec4_f32x16& A, const Vec4_f32x16& B, const Vec4_f32x16& C, std::array<float32x16, 3>& outBarycentrics);
	static void calculateBarycentricCoordinates3D(const Vec4_f32x8& P, const Vec4_f32x8& A, const Vec4_f32x8& B, const Vec4_f32x8& C, std::array<float32x8, 3>& outBarycentrics);
	static void mask_store_vec4_f32x16_to_framebuffer(const Vec4_f32x16& pack, void* frameBuffer, int x, int y, int w, Mask16 mask);
	static Vec4_f32x16 mask_load_vec4_f32x16_from_framebuffer(const void* frameBuffer, int x, int y, int w, Mask16 mask);
	
	//Calculates barycentric coordinates for 2D vector P relative to vertices A, B, C and returns them in an array
	//Only x and y values from input vectors are used.
	template<typename VectorType, typename ValueType>
	static __forceinline std::array<ValueType, 3> calculateBarycentricCoordinates2D(const VectorType& P, const VectorType& A, const VectorType& B, const VectorType& C, const ValueType& rcpSignedArea)
	{
		std::array<ValueType, 3> bary;
		bary[0] = (P - C).cross2d(B - C) * rcpSignedArea;
		bary[1] = (P - C).cross2d(C - A) * rcpSignedArea;
		bary[2] = (P - A).cross2d(A - B) * rcpSignedArea; //do NOT change this to 1-alpha-beta or 1-(alpha+beta). That causes wonkiness in textures on big triangles
		return bary;
	}

	//Calculates barycentric coordinates for 2D vector P relative to vertices A, B, C and stores them in retInitials
	//Calculates steps for unit movements in X and Y, and stores them into retStepsX and retStepsY
	//Stepping can be done by calculating: barycentric[i] = retInitials[i] + (x-P.x)*retStepsX[i] + (y-P.y)*retStepsY[i]
	//Only x and y values from input vectors are used for the calculations, the rest are ignored.
	template<typename VectorType, typename ValueType>
	static __forceinline void calculateBarycentricCoordinatesAndSteps2D(const VectorType& P, const VectorType& A, const VectorType& B, const VectorType& C, const ValueType& rcpSignedArea, std::array<ValueType, 3>& retInitials, std::array<ValueType, 3>& retStepsX, std::array<ValueType, 3>& retStepsY)
	{
		retInitials = calculateBarycentricCoordinates2D(P, A, B, C, rcpSignedArea);
		float32x16 group_dAlpha_dx = (B.y - C.y) * rcpSignedArea;
		float32x16 group_dAlpha_dy = (C.x - B.x) * rcpSignedArea;
		float32x16 group_dBeta_dx = (C.y - A.y) * rcpSignedArea;
		float32x16 group_dBeta_dy = (A.x - C.x) * rcpSignedArea;
		float32x16 group_dGamma_dx = (A.y - B.y) * rcpSignedArea; //this should have better precision than -group_dAlpha_dx - group_dBeta_dx since y2 should cancel out completely algebraically;
		float32x16 group_dGamma_dy = (B.x - A.x) * rcpSignedArea; //same for -group_dAlpha_dy - group_dBeta_dy and x2
		retStepsX[0] = group_dAlpha_dx;
		retStepsX[1] = group_dBeta_dx;
		retStepsX[2] = group_dGamma_dx;
		retStepsY[0] = group_dAlpha_dy;
		retStepsY[1] = group_dBeta_dy;
		retStepsY[2] = group_dGamma_dy;
	}
	static __forceinline void mask_store_rows_512_to_4x128_ps(__m512 value, __mmask16 mask, void* dst, uint32_t xStart, uint32_t yStart, uint32_t w)
	{
		__m128 v0 = _mm512_extractf32x4_ps(value, 0);
		__m128 v1 = _mm512_extractf32x4_ps(value, 1);
		__m128 v2 = _mm512_extractf32x4_ps(value, 2);
		__m128 v3 = _mm512_extractf32x4_ps(value, 3);
		float* p = (float*)dst;
		_mm_mask_storeu_ps(p + yStart * w + xStart, mask, v0);
		_mm_mask_storeu_ps(p + (1+yStart) * w + xStart, mask >> 4, v1);
		_mm_mask_storeu_ps(p + (2+yStart) * w + xStart, mask >> 8, v2);
		_mm_mask_storeu_ps(p + (3+yStart) * w + xStart, mask >> 12, v3);
	}
	static __forceinline __m512 mask_load_rows_4x128_to_512_ps(__mmask16 mask, const void* src, uint32_t xStart, uint32_t yStart, uint32_t w)
	{
		const float* p = (const float*)src;
		__m128 v0 = _mm_maskz_loadu_ps(mask, p + yStart * w + xStart);
		__m128 v1 = _mm_maskz_loadu_ps(mask >> 4, p + (yStart + 1) * w + xStart);
		__m128 v2 = _mm_maskz_loadu_ps(mask >> 8, p + (yStart + 2) * w + xStart);
		__m128 v3 = _mm_maskz_loadu_ps(mask >> 12, p + (yStart + 3) * w + xStart);
		__m512 ret = _mm512_castps128_ps512(v0);
		ret = _mm512_insertf32x4(ret, v1, 1);
		ret = _mm512_insertf32x4(ret, v2, 2);
		ret = _mm512_insertf32x4(ret, v3, 3);
		return ret;
	}

	//Writes out colors to pixels (x,y) of the framebuffer using mask
	static void scatterToFrameBuffer(const Vec4_f32x16& colors, int32x16 x, int32x16 y, Mask16 mask, void* frameBuf, int framebufW);
protected:
	TextureManager textureManager;
};