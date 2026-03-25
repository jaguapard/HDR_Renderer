#include "RayCastingRenderer.h"
#include <optional>
#include "../Vec.h"
#include "../GameSettings.h"

void RayCastingRenderer::loadScene(std::string path, std::string mode)
{
}

//https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
float rayTriangleIntersectionT(Vec4f rayOrigin, Vec4f rayDir, Vec4f triA, Vec4f triB, Vec4f triC)
{
	//TODO: change dot products to 3d!
	constexpr float epsilon = std::numeric_limits<float>::epsilon();
	constexpr float eps = std::numeric_limits<float>::epsilon();

	Vec4f edge1 = triB - triA;
	Vec4f edge2 = triC - triA;

	// Backface culling, assuming CCW-wound triangles.
	//const Vec4f normal = edge1.cross3d(edge2); // No need to normalize
	//if (normal.dot(rayDir) > 0) return INFINITY;

	Vec4f ray_cross_e2 = rayDir.cross3d(edge2);
	float det = edge1.dot(ray_cross_e2);

	if (abs(det) < epsilon) return INFINITY; // Ray is parallel to triangle

	float inv_det = 1.0 / det;
	Vec4f s = rayOrigin - triA;
	float u = inv_det * s.dot(ray_cross_e2);

	if (u < -eps || u - 1 > eps) return INFINITY; // Ray passes outside edge2's bounds

	Vec4f s_cross_e1 = s.cross3d(edge1);
	float v = inv_det * rayDir.dot(s_cross_e1);

	if (v < -eps || u + v - 1 > eps) return INFINITY; // Ray passes outside edge1's bounds

	// The ray line intersects with the triangle.
	// We compute t to find where on the ray the intersection is.
	float t = inv_det * edge2.dot(s_cross_e1);

	if (t > epsilon) // Ray intersection
	{
		return t;
	}
	else // This means that there is a line intersection but not a ray intersection.
		return INFINITY;
}
void RayCastingRenderer::renderFrame(const GameSettings& settings)
{
	int bufW = 2560, bufH = 1440;
	//coordinate check
	/*	first:	second:
		___		
		\  |	|\		
		 \ |	| \
		  \|	|__\
	*/
	Vec4f vertices[] = {
		{-10, 5, 10},
		{10, 5, 10},
		{10, -5, 10},

		{10, -5, 10},
		{-10, -5, 10},
		{-10, 5, 10},

		{-10, 5, 10},
		{10, 5, 10},
		{-10, 5, 30},
	};

	Vec4f colors[] = {
		{1,0,0,1},
		{0,1,0,1},
		{0,0,1,1},
		{1,1,0,1},
	};

	Vec4f forward = settings.forward;
	Vec4f right = settings.right;
	Vec4f down = settings.down;
	Vec4f camPos = settings.camPos;
	float widthToHeightRatio = double(bufW) / bufH;
	Vec4f* pixels = (Vec4f*)settings.graphicsOutputBuffer;

	for (int y = 0; y < bufH; ++y)
	{
		for (int x = 0; x < bufW; ++x)
		{
			float progressX = x / float(bufH);
			float progressY = 1 - y / float(bufH);
			Vec4f rayDir = forward * settings.cameraPlane_zDist + down * (progressY - 0.5) + right * (progressX - widthToHeightRatio * 0.5);
			bool hit = false;
			for (int i = 0; i < 3; ++i)
			{
				float t = rayTriangleIntersectionT(camPos, rayDir, vertices[i*3], vertices[i*3+1], vertices[i*3+2]);
				if (t != INFINITY)
				{
					float intensity = std::min(1.f / t, 1.f);
					pixels[y * bufW + x] = colors[i];
					hit = true;
				}
			}

			if (!hit)
			{
				pixels[y * bufW + x] = { 0,0,0,1 };
			}
		}
	}
}
