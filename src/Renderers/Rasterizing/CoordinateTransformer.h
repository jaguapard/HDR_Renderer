#pragma once
#include "../../Vec.h"
#include "../lib/bob/Matrix4.h"
#include "../lib/bob/VectorPack.h"

class CoordinateTransformer
{
public:
	CoordinateTransformer() = default;
	CoordinateTransformer(int w, int h);
	void prepare(const Vec4f camPos, const Vec4f camAng);
	Vec4f screenSpaceToPixels(const Vec4f v) const;
	bob::Vec4_f32x16 screenSpaceToPixels(const bob::Vec4_f32x16& v) const;

	Vec4f rotateAndTranslate(Vec4f v) const;
	Vec4_f32x16 rotateAndTranslate(const Vec4_f32x16& v) const;
	Vec4f shift(const Vec4f v) const;

	Matrix4 getCurrentTransformationMatrix() const;
	Matrix4 getCurrentInverseTransformationMatrix() const;

	Vec4f inverseScreenPixelsToWorld(const Vec4f& v, float zInverse) const;

	template<size_t N>
	bob::VectorPack<SIMD_Vector<float,N>> inverseScreenPixelsToWorld(const bob::VectorPack<SIMD_Vector<float, N>>& v) const
	{
		bob::VectorPack<SIMD_Vector<float, N>> screenSpace = v * this->rcp_hVec;
		bob::VectorPack<SIMD_Vector<float, N>> post_zDivide = screenSpace - this->_shift;
		bob::VectorPack<SIMD_Vector<float, N>> pre_zDivide = post_zDivide / v.w;
		return inverseRotationTranslation * pre_zDivide;
	}
private:
	Matrix4 rotationTranslation;
	Matrix4 inverseRotationTranslation;
	Vec4f _shift, hVec, rcp_hVec, shift_hVec;
};