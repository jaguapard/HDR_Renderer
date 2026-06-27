#pragma once
#include "Vec.h"

struct BoundingBox
{
	float xmin, ymin, zmin, xmax, ymax, zmax;

	BoundingBox() = default;
	BoundingBox(Vec4f v1, Vec4f v2, Vec4f v3);

	BoundingBox unionWith(const BoundingBox& other) const;
	bool intersectsWith(const BoundingBox& other) const;
	bool containsFully(const BoundingBox& other) const;

	//Checks intersection of a single ray against this bounding box and returns {tmin, tmax} of the intersection
	//If the ray doesn't intersect the bounding box, {FLT_MAX, -FLT_MAX} is returned
	std::pair<float, float> getMinAndMaxIntestionsFor(Vec4f rayOrigin, Vec4f rcpRayDir) const;

	//Checks intersection of 16 rays against this bounding box, and writes out {minT, maxT} of the intersections.
	//Return value: mask with bits set for rays hitting this bounding box or cleared otherwise. ret_tMin and ret_tMax are undefined for rays not hitting this bounding box
	mask16d getMinAndMaxIntestionsFor(Vec4_f32x16 rayOrigin, Vec4_f32x16 rcpRayDir, float32x16& ret_tMin, float32x16& ret_tMax) const;
	mask8d getMinAndMaxIntestionsFor(Vec4_f32x8 rayOrigin, Vec4_f32x8 rcpRayDir, float32x8& ret_tMin, float32x8& ret_tMax) const;
	static BoundingBox infinite();
};
