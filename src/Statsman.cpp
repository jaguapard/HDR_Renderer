#include "Statsman.h"

void Statsman::reset()
{
	memset(this, 0, sizeof(*this));
}

Statsman Statsman::operator+(const Statsman& other) const
{
	Statsman ret = *this;
	ret.triangles.rendered += other.triangles.rendered;
	ret.triangles.total += other.triangles.total;
	for (int i = 0; i <= 3; ++i) ret.triangles.verticesBehindNearPlane[i] += other.triangles.verticesBehindNearPlane[i];
	
	ret.rendering.barycentricsCalculated += other.rendering.barycentricsCalculated;
	ret.rendering.pointsInsideTriangles += other.rendering.pointsInsideTriangles;
	ret.rendering.zBufferFetchLanes += other.rendering.zBufferFetchLanes;
	ret.rendering.zBufferFetchAliveLanes += other.rendering.zBufferFetchAliveLanes;
	ret.rendering.visiblePoints += other.rendering.visiblePoints;
	ret.rendering.opaquePixels += other.rendering.opaquePixels;
	ret.rendering.textureGatheredLanes += other.rendering.textureGatheredLanes;
	ret.rendering.textureGatherAliveLanes += other.rendering.textureGatherAliveLanes;
	ret.rendering.zBufferWriteLanes += other.rendering.zBufferWriteLanes;
	ret.rendering.zBufferWriteAliveLanes += other.rendering.zBufferWriteAliveLanes;
	ret.rendering.frameBufWriteLanes += other.rendering.frameBufWriteLanes;
	ret.rendering.frameBufWriteAliveLanes += other.rendering.frameBufWriteAliveLanes;

	ret.ag.transformMsTotal += other.time.transformMs;
	ret.ag.transformMsMin = std::min(this->time.transformMs, other.time.transformMs);
	ret.ag.transformMsMax = std::max(this->time.transformMs, other.time.transformMs);
	ret.ag.drawMsMin = std::min(time.drawMs, other.time.drawMs);
	ret.ag.drawMsMax = std::max(time.drawMs, other.time.drawMs);
	ret.ag.drawMsTotal += other.time.drawMs;
	return ret;
}

Statsman::Aggregated Statsman::getAggregatedInfo() const
{
	return ag;
}
