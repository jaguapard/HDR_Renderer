#pragma once
#include "Vec.h"
#include <d3d11.h>
#include <bob/Matrix4.h>
#undef min
#undef max

struct GameSettings
{
	Vec4f camPos, camAng, forward, down, right;
	Matrix4 viewMatrix;
	float cameraPlane_zDist = 1;
	void* graphicsOutputBuffer;
	DXGI_FORMAT graphicsOutputFormat;
	bool mouseCaptured = false;
	float flySpeed = 15;
};