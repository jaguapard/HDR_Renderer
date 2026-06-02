#include "BoundingBox.h"
BoundingBox::BoundingBox(Vec4f v1, Vec4f v2, Vec4f v3)
{
	xmin = ymin = zmin = FLT_MAX;
	xmax = ymax = zmax = -FLT_MAX;
	for (const auto& v : {v1,v2,v3})
	{
		xmin = std::min(xmin, v.x);
		ymin = std::min(ymin, v.y);
		zmin = std::min(zmin, v.z);
		xmax = std::max(xmax, v.x);
		ymax = std::max(ymax, v.y);
		zmax = std::max(zmax, v.z);
	}
}

BoundingBox BoundingBox::unionWith(const BoundingBox& other) const
{
	BoundingBox bb;
	bb.xmin = std::min(xmin, other.xmin);
	bb.ymin = std::min(ymin, other.ymin);
	bb.zmin = std::min(zmin, other.zmin);
	bb.xmax = std::max(xmax, other.xmax);
	bb.ymax = std::max(ymax, other.ymax);
	bb.zmax = std::max(zmax, other.zmax);
	return bb;
}

bool BoundingBox::intersectsWith(const BoundingBox& other) const
{
	return xmin <= other.xmax && xmax >= other.xmin
		&& ymin <= other.ymax && ymax >= other.ymin
		&& zmin <= other.zmax && zmax >= other.zmin;
}
bool BoundingBox::containsFully(const BoundingBox& other) const
{
	return xmin <= other.xmin && xmax >= other.xmax
		&& ymin <= other.ymin && ymax >= other.ymax
		&& zmin <= other.zmin && zmax >= other.zmax;
}

BoundingBox BoundingBox::infinite()
{
	BoundingBox bb;
	bb.xmin = bb.ymin = bb.zmin = -FLT_MAX;
	bb.xmax = bb.ymax = bb.zmax = FLT_MAX;
	return bb;
}

std::pair<float, float> BoundingBox::getMinAndMaxIntestionsFor(Vec4f rayOrigin, Vec4f rcpRayDir) const
{
	float tx1 = (xmin - rayOrigin.x) * rcpRayDir.x;
	float ty1 = (ymin - rayOrigin.y) * rcpRayDir.y;
	float tz1 = (zmin - rayOrigin.z) * rcpRayDir.z;

	float tx2 = (xmax - rayOrigin.x) * rcpRayDir.x;
	float ty2 = (ymax - rayOrigin.y) * rcpRayDir.y;
	float tz2 = (zmax - rayOrigin.z) * rcpRayDir.z;

	float tmin_x = std::min(tx1, tx2);
	float tmin_y = std::min(ty1, ty2);
	float tmin_z = std::min(tz1, tz2);

	float tmax_x = std::max(tx1, tx2);
	float tmax_y = std::max(ty1, ty2);
	float tmax_z = std::max(tz1, tz2);

	float tmin_total = std::max(std::max(0.f, tmin_z), std::max(tmin_x, tmin_y));
	float tmax_total = std::min(tmax_z, std::min(tmax_x, tmax_y));
	if (tmin_total > tmax_total) return { FLT_MAX, -FLT_MAX };
	return { tmin_total, tmax_total };
}

Mask16 BoundingBox::getMinAndMaxIntestionsFor(Vec4_f32x16 rayOrigins, Vec4_f32x16 rcpRayDirs, float32x16& ret_tMin, float32x16& ret_tMax) const
{
	float32x16 tx1 = (float32x16(this->xmin) - rayOrigins.x) * rcpRayDirs.x;
	float32x16 ty1 = (float32x16(this->ymin) - rayOrigins.y) * rcpRayDirs.y;
	float32x16 tz1 = (float32x16(this->zmin) - rayOrigins.z) * rcpRayDirs.z;

	float32x16 tx2 = (float32x16(this->xmax) - rayOrigins.x) * rcpRayDirs.x;
	float32x16 ty2 = (float32x16(this->ymax) - rayOrigins.y) * rcpRayDirs.y;
	float32x16 tz2 = (float32x16(this->zmax) - rayOrigins.z) * rcpRayDirs.z;

	float32x16 tmin_x = _mm512_min_ps(tx1, tx2);
	float32x16 tmin_y = _mm512_min_ps(ty1, ty2);
	float32x16 tmin_z = _mm512_min_ps(tz1, tz2);

	float32x16 tmax_x = _mm512_max_ps(tx1, tx2);
	float32x16 tmax_y = _mm512_max_ps(ty1, ty2);
	float32x16 tmax_z = _mm512_max_ps(tz1, tz2);

	float32x16 tmin_total = _mm512_max_ps(_mm512_max_ps(_mm512_setzero_ps(), tmin_z), _mm512_max_ps(tmin_x, tmin_y));
	float32x16 tmax_total = _mm512_min_ps(tmax_z, _mm512_min_ps(tmax_x, tmax_y));
	ret_tMin = tmin_total;
	ret_tMax = tmax_total;
	return tmin_total <= tmax_total; //TODO: should this have equality?
}
float32x8 BoundingBox::getMinAndMaxIntestionsFor(Vec4_f32x8 rayOrigins, Vec4_f32x8 rcpRayDirs, float32x8& ret_tMin, float32x8& ret_tMax) const
{
	float32x8 tx1 = (float32x8(this->xmin) - rayOrigins.x) * rcpRayDirs.x;
	float32x8 ty1 = (float32x8(this->ymin) - rayOrigins.y) * rcpRayDirs.y;
	float32x8 tz1 = (float32x8(this->zmin) - rayOrigins.z) * rcpRayDirs.z;

	float32x8 tx2 = (float32x8(this->xmax) - rayOrigins.x) * rcpRayDirs.x;
	float32x8 ty2 = (float32x8(this->ymax) - rayOrigins.y) * rcpRayDirs.y;
	float32x8 tz2 = (float32x8(this->zmax) - rayOrigins.z) * rcpRayDirs.z;

	float32x8 tmin_x = _mm256_min_ps(tx1, tx2);
	float32x8 tmin_y = _mm256_min_ps(ty1, ty2);
	float32x8 tmin_z = _mm256_min_ps(tz1, tz2);

	float32x8 tmax_x = _mm256_max_ps(tx1, tx2);
	float32x8 tmax_y = _mm256_max_ps(ty1, ty2);
	float32x8 tmax_z = _mm256_max_ps(tz1, tz2);

	float32x8 tmin_total = _mm256_max_ps(_mm256_max_ps(_mm256_setzero_ps(), tmin_z), _mm256_max_ps(tmin_x, tmin_y));
	float32x8 tmax_total = _mm256_min_ps(tmax_z, _mm256_min_ps(tmax_x, tmax_y));
	ret_tMin = tmin_total;
	ret_tMax = tmax_total;
	return mask2vec<float>(tmin_total <= tmax_total); //TODO: should this have equality?
}