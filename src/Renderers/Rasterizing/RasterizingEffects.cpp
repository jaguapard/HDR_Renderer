#include "RasterizingEffects.h"
#include <random>
using namespace Rasterizing;

Rasterizing::ExplodeAndRestoreSceneEffect::ExplodeAndRestoreSceneEffect(double startTime, double explosionDuration, double restoreDuration, size_t triangleCount)
{
	this->startTime = startTime;
	this->flipTime = startTime + explosionDuration;
	this->endTime = this->flipTime + restoreDuration;

	std::random_device dev;
	std::mt19937 gen(dev());

	std::normal_distribution<float> shiftDistrib(0, 300);
	std::uniform_real_distribution<float> rotDistrib(0, 4 * 3.14159 * 16);
	for (size_t i = 0; i < triangleCount; ++i)
	{
		for (int j = 0; j < 3; ++j) {
			this->shift[j].push_back(shiftDistrib(gen));
			this->rot[j].push_back(rotDistrib(gen));
		}
	}
}

void Rasterizing::ExplodeAndRestoreSceneEffect::onFrameStart(double gameTime)
{
	this->gameTime = gameTime;
	if (gameTime < this->flipTime)
	{
		this->effectProgress = (gameTime - this->startTime) / (this->flipTime - this->startTime);
	}
	else this->effectProgress = 1 - (gameTime - this->flipTime) / (this->endTime - this->flipTime);
}

std::array<VertexPack16, 3> Rasterizing::ExplodeAndRestoreSceneEffect::applyToTriangles(const std::array<VertexPack16, 3>& verts, const int32x16 triangleInd, Mask16 mask) const
{
	float32x16 z = 0.f;
	Vec4_f32x16 triShift, triRot;

	float lerpT = powf(this->effectProgress, 8);
	for (int i = 0; i < 3; ++i)
	{
		float32x16 shift = _mm512_mask_i32gather_ps(z, mask, triangleInd, this->shift[i].data(), 4);
		float32x16 rot = _mm512_mask_i32gather_ps(z, mask, triangleInd, this->rot[i].data(), 4);
		triShift[i] = lerp(0.f, shift, lerpT);
		triRot[i] = lerp(0.f, rot, lerpT);
	}
	
	std::array<VertexPack16, 3> ret = verts;
	float32x16 triangleArea = (verts[0].space - verts[1].space).cross3d(verts[0].space - verts[2].space).len3d() * 0.5f;
	Vec4_f32x16 triangleMiddle = (verts[0].space + verts[1].space + verts[2].space) / 3.f;
	mask &= triangleArea < 400.f;
	for (int i = 0; i < 16; ++i)
	{
		if (!(mask.mask & (1 << i))) continue;
		
		Matrix4 rotation = Matrix4::rotationXYZ(triRot.extractHorizontalVector(i));
		/*Matrix4 translation = Matrix4::identity();
		translation.elements[0][3] = -triShift.x[i];
		translation.elements[1][3] = -triShift.y[i];
		translation.elements[2][3] = -triShift.z[i];
		translation.elements[3][3] = 1.f;
		//Matrix4 total = rotation*translation;
		Matrix4 total = rotation;*/
		Vec4f tm = triangleMiddle.extractHorizontalVector(i);
		for (int j = 0; j < 3; ++j)
		{
			Vec4f v = verts[j].space.extractHorizontalVector(i);
			Vec4f relToMiddle = v - tm;
			Vec4f relToMiddleRotate = rotation * relToMiddle;
			Vec4f transformed = relToMiddleRotate + tm + triShift.extractHorizontalVector(i);
			//Vec4f transformed = total * verts[j].space.extractHorizontalVector(i);
			transformed.w = 1; //should you?
			for (int k = 0; k < 4; ++k) ret[j].space[k][i] = transformed[k];
		}
	}
	return ret;
}

bool Rasterizing::ExplodeAndRestoreSceneEffect::isFinished() const
{
	return this->gameTime > this->endTime;
}
