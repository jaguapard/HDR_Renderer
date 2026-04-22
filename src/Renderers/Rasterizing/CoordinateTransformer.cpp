#include "CoordinateTransformer.h"
using namespace bob;
CoordinateTransformer::CoordinateTransformer(int w, int h)
{
	float widthToHeightAspectRatio = float(w) / h;
	this->_shift = { widthToHeightAspectRatio / 2, 0.5, 0 };
	this->hVec = Vec4f(h, h, 1, 1);
	this->shift_hVec = Vec4f(_shift.x, _shift.y, h, h);
}

void CoordinateTransformer::prepare(const Vec4f camPos, const Vec4f camAng)
{
	Matrix4 rotation = Matrix4::rotationXYZ(camAng);
	Matrix4 translation = Matrix4::identity();
	translation[0][3] = -camPos.x;
	translation[1][3] = -camPos.y;
	translation[2][3] = -camPos.z;
	translation[3][3] = 1;

	this->rotationTranslation = (rotation * translation);//.transposed();
	this->inverseRotationTranslation = rotationTranslation.inverse();
	//this->translationRotation = translation * rotation;
}

/*Vec4f CoordinateTransformer::toScreenCoords(const Vec4f v) const
{
	assert(this->_shift.z == 0.0); //ensure to not touch z
	Vec4f camOffset = v - camPos;
	Vec4f rot = rotation.multiplyByTransposed(camOffset);
	Vec4f perspective = rot / rot.z; //screen space coords of vector

	Vec4f shifted = perspective + this->_shift; //convert so (0,0) in `perspective` corresponds to center of the screen
	Vec4f final = shifted * h;
	return final;
}*/

Vec4f CoordinateTransformer::screenSpaceToPixels(const Vec4f v) const
{
	assert(this->_shift.z == 0.0); //ensure to not touch z
	return (v + this->_shift) * hVec;
}

bob::Vec4_f32x16 CoordinateTransformer::screenSpaceToPixels(const bob::Vec4_f32x16& v) const
{
	Vec4_f32x16 ret;
	ret.x = (v.x + this->shift_hVec.x) * this->shift_hVec.z;
	ret.y = (v.y + this->shift_hVec.y) * this->shift_hVec.z;
	return ret;
}

Vec4f CoordinateTransformer::rotateAndTranslate(Vec4f v) const
{
	Vec4f interm = rotationTranslation * v;
	return interm;
}

Vec4_f32x16 CoordinateTransformer::rotateAndTranslate(const Vec4_f32x16& v) const
{
	return rotationTranslation * v;
}

Vec4f CoordinateTransformer::shift(const Vec4f v) const
{
	assert(this->_shift.z == 0.0); //ensure to not touch z
	return v + this->_shift;
}

Matrix4 CoordinateTransformer::getCurrentTransformationMatrix() const
{
	return rotationTranslation;
}

Matrix4 CoordinateTransformer::getCurrentInverseTransformationMatrix() const
{
	return inverseRotationTranslation;
}

Vec4f CoordinateTransformer::inverseScreenPixelsToWorld(const Vec4f& v, float zInverse) const
{
	//reverse transformations in backwards order
	Vec4f screenSpace = v / hVec;
	Vec4f post_zDivide = screenSpace - this->_shift;
	Vec4f pre_zDivide = post_zDivide / zInverse;
	//pre_zDivide.w = 1;
	return inverseRotationTranslation * pre_zDivide;
}

bob::Vec4_f32x16 CoordinateTransformer::inverseScreenPixelsToWorld(const bob::Vec4_f32x16& v) const
{
	Vec4_f32x16 screenSpace = v / hVec;
	Vec4_f32x16 post_zDivide = screenSpace - this->_shift;
	Vec4_f32x16 pre_zDivide = post_zDivide / v.w;
	return inverseRotationTranslation * pre_zDivide;
}
