#include "RasterizingExplodeAndRestoreSceneEffect.h"
#include <random>
using namespace Rasterizing;

Rasterizing::ExplodeAndRestoreSceneEffect::ExplodeAndRestoreSceneEffect(double startTime, double explosionDuration, double restoreDuration, size_t triangleCount)
{
	this->startTime = startTime;
	this->flipTime = startTime + explosionDuration;
	this->endTime = this->flipTime + restoreDuration;

	std::random_device dev;
	std::mt19937 gen(dev());

	//these values represent max shift and rotation during the effect
	std::normal_distribution<float> shiftDistrib(0, 300);
	std::uniform_real_distribution<float> rotDistrib(0, 4 * 3.14159 * 16);
	for (size_t i = 0; i < triangleCount; ++i)
	{
		for (int j = 0; j < 3; ++j) {
			this->packedShiftAndRotationPerTriangle.push_back(shiftDistrib(gen));
			this->packedShiftAndRotationPerTriangle.push_back(rotDistrib(gen));
		}
	}
}

void Rasterizing::ExplodeAndRestoreSceneEffect::onFrameStart(double gameTime)
{
	this->gameTime = gameTime;
	if (gameTime < this->flipTime)
	{
		this->lerpBase = (gameTime - this->startTime) / (this->flipTime - this->startTime);
	}
	else this->lerpBase = 1 - (gameTime - this->flipTime) / (this->endTime - this->flipTime);
}

void Rasterizing::ExplodeAndRestoreSceneEffect::applyToTrianglesInPlace(std::array<VertexPack16, 3>& verts, const int32x16 triangleInd, Mask16 mask) const
{
	float32x16 triangleArea = (verts[0].space - verts[1].space).cross3d(verts[0].space - verts[2].space).len3d() * 0.5f;
	Vec4_f32x16 triangleMiddle = (verts[0].space + verts[1].space + verts[2].space) / 3.f;
	//don't affect large triangles, since they pollute the screen with their movement and rotation. Also looks better, like only flimsy things exploding, while wall and floors stay firm
	//TODO: may tesselate them in the future?
	mask &= triangleArea < 400.f;
	if (!mask) return;

	float32x16 z = 0.f;
	Vec4_f32x16 triShift, triRot;
	float lerpT = powf(this->lerpBase, 8);
	std::array<float32x16, 6> unpacked = aos2soa_gather_and_transpose<float32x16, 6>(this->packedShiftAndRotationPerTriangle.data(), triangleInd, mask);
	for (int i = 0; i < 3; ++i)
	{
		triShift[i] = lerp<float32x16>(0.f, unpacked[i * 2], lerpT);
		triRot[i] = lerp<float32x16>(0.f, unpacked[i * 2 + 1], lerpT);
	}

	MatrixPack16_4x4 rotation = MatrixPack16_4x4::fast_rotationXYZ(triRot);
	for (int i = 0; i < 3; ++i)
	{
		Vec4_f32x16 transformed = rotation * (verts[i].space - triangleMiddle) + triangleMiddle + triShift;
		transformed.w = 1; //TODO: look at this when adding homogeneous coords
		for (int j = 0; j < 4; ++j) verts[i].space[j] = _mm512_mask_mov_ps(verts[i].space[j], mask, transformed[j]);
	}
}

bool Rasterizing::ExplodeAndRestoreSceneEffect::isFinished() const
{
	return this->gameTime > this->endTime;
}
