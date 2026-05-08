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
	D3D11_TEXTURE2D_DESC outputTextureParams;
	bool mouseCaptured = false;
	bool osdEnabled = false;
	bool texturingEnabled = true;
	float flySpeed = 15;
	double gameTime = 0;

	Threadpool* threadpool = nullptr;
};