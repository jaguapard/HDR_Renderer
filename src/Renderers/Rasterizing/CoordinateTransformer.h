#pragma once
#include "../../Vec.h"
#include "../lib/bob/Matrix4.h"

class CoordinateTransformer
{
public:
	CoordinateTransformer() = default;
	CoordinateTransformer(int w, int h);
	void prepare(const Vec4f camPos, const Vec4f camAng);
	Vec4f screenSpaceToPixels(const Vec4f v) const;
	Vec4_f32x16 screenSpaceToPixels(const Vec4_f32x16& v) const;

	Vec4f rotateAndTranslate(Vec4f v) const;
	Vec4_f32x16 rotateAndTranslate(const Vec4_f32x16& v) const;
	Vec4f shift(const Vec4f v) const;

	Matrix4 getCurrentTransformationMatrix() const;
	Matrix4 getCurrentInverseTransformationMatrix() const;

	Vec4f inverseScreenPixelsToWorld(const Vec4f& v, float zInverse) const;

	template<size_t N>
	SIMD_VectorPack<SIMD_Vector<float,N>, 4> inverseScreenPixelsToWorld(const SIMD_VectorPack<SIMD_Vector<float, N>, 4>& v) const
	{
		using vp = SIMD_VectorPack<SIMD_Vector<float, N>, 4>;
		vp screenSpace = v * this->rcp_hVec;
		vp post_zDivide = screenSpace - vp(this->_shift.x, _shift.y, _shift.z, _shift.w);
		vp pre_zDivide = post_zDivide / v.w;
		return inverseRotationTranslation * pre_zDivide;
	}
private:
	Matrix4 rotationTranslation;
	Matrix4 inverseRotationTranslation;
	Vec4f _shift, hVec, rcp_hVec, shift_hVec;
};