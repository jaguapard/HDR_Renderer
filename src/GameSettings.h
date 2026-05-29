#pragma once
#include "Vec.h"
#include <d3d11.h>
#include <bob/Matrix4.h>
#undef min
#undef max

class Threadpool;
struct GameSettings
{
	Vec4f camPos, camAng, forward, down, right;
	Matrix4 viewMatrix;
	float cameraPlane_zDist = 1;
	void* graphicsOutputBuffer;
	uint32_t outputTextureW, outputTextureH, screenW, screenH;
	bool mouseCaptured = false;
	bool osdEnabled = false;
	bool texturingEnabled = true;
	float flySpeed = 550;
	double gameTime = 0;
	double gameTimeLastDt = 0;

	Threadpool* threadpool = nullptr;
};