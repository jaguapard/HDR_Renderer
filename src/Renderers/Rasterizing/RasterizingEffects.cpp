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

	std::uniform_real_distribution<float> shiftDistrib(-10, 10);
	std::normal_distribution<float> rotDistrib(0, 4 * 3.14159);
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
		this->lerpT = (gameTime - this->startTime) / (this->flipTime - this->startTime);
	}
	else this->lerpT = 1 - (gameTime - this->flipTime) / (this->endTime - this->flipTime);
}

std::array<VertexPack16, 3> Rasterizing::ExplodeAndRestoreSceneEffect::applyToTriangles(const std::array<VertexPack16, 3>& verts, const int32x16 triangleInd, Mask16 mask) const
{
	float32x16 z = 0.f;
	Vec4_f32x16 triShift, triRot;

	for (int i = 0; i < 3; ++i)
	{
		float32x16 shift = _mm512_mask_i32gather_ps(z, mask, triangleInd, this->shift[i].data(), 4);
		float32x16 rot = _mm512_mask_i32gather_ps(z, mask, triangleInd, this->rot[i].data(), 4);
		triShift[i] = lerp(0.f, shift, this->lerpT);
		triRot[i] = lerp(0.f, shift, this->lerpT);
	}
	
	std::array<VertexPack16, 3> ret = verts;
	for (int i = 0; i < 16; ++i)
	{
		if (!(mask.mask & (1 << i))) continue;

		Matrix4 rotation = Matrix4::rotationXYZ(triRot.extractHorizontalVector(i));
		Matrix4 translation = Matrix4::identity();
		translation.elements[0][3] = -triShift.x[i];
		translation.elements[1][3] = -triShift.y[i];
		translation.elements[2][3] = -triShift.z[i];
		translation.elements[3][3] = 1.f;
		//Matrix4 total = translation*rotation;
		Matrix4 total = translation;//*rotation;

		for (int j = 0; j < 3; ++j)
		{
			Vec4f transformed = total * verts[j].space.extractHorizontalVector(i);
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
