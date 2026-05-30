#pragma once
#include "Vec.h"
#include <d3d11.h>
#include <bob/Matrix4.h>
#undef min
#undef max

class Threadpool;
struct GameSettings
{
	Matrix4 viewMatrix;
	Vec4f camPos, camAng, forward, down, right;
	float verticalFovDegrees = 70, cameraPlane_zDist;
	void* graphicsOutputBuffer;
	uint32_t outputTextureW, outputTextureH, screenW, screenH;
	bool mouseCaptured = false;
	bool osdEnabled = false;
	bool texturingEnabled = true;
	bool vsyncEnabled = true;
	float flySpeed = 550;
	double gameTime = 0;
	double gameTimeLastDt = 0;

	Threadpool* threadpool = nullptr;
};