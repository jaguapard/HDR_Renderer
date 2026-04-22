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
	bob::Vec4_f32x16 inverseScreenPixelsToWorld(const bob::Vec4_f32x16& v) const;
private:
	Matrix4 rotationTranslation;
	Matrix4 inverseRotationTranslation;
	Vec4f _shift;
	Vec4f hVec;
	Vec4f shift_hVec;
};