#include "Statsman.h"

void Statsman::reset()
{
	memset(this, 0, sizeof(*this));
}

std::pair<Statsman, Statsman::Aggregated> Statsman::aggregateAll()
{
	Statsman ret;
	Aggregated ag;
	for (auto& other : Statsman::statsmenForThreads)
	{
		ret.triangles.rendered += other.triangles.rendered;
		ret.triangles.total += other.triangles.total;
		for (int i = 0; i <= 3; ++i) ret.triangles.verticesBehindNearPlane[i] += other.triangles.verticesBehindNearPlane[i];
		ret.triangles.vertIndexDelta += other.triangles.vertIndexDelta;
		ret.triangles.vertIndexDeltaCount += other.triangles.vertIndexDeltaCount;
		if (other.triangles.vertIndexDeltaMin) ret.triangles.vertIndexDeltaMin = std::min(ret.triangles.vertIndexDeltaMin.value_or(UINT64_MAX), *other.triangles.vertIndexDeltaMin);
		if (other.triangles.vertIndexDeltaMax) ret.triangles.vertIndexDeltaMax = std::max(ret.triangles.vertIndexDeltaMax.value_or(0), *other.triangles.vertIndexDeltaMax);

		ret.rendering.barycentricsCalculated += other.rendering.barycentricsCalculated;
		ret.rendering.pointsInsideTriangles += other.rendering.pointsInsideTriangles;
		ret.rendering.zBufferFetchLanes += other.rendering.zBufferFetchLanes;
		ret.rendering.zBufferFetchAliveLanes += other.rendering.zBufferFetchAliveLanes;
		ret.rendering.notOccludedPoints += other.rendering.notOccludedPoints;
		ret.rendering.opaquePixels += other.rendering.opaquePixels;
		ret.rendering.textureGatheredLanes += other.rendering.textureGatheredLanes;
		ret.rendering.textureGatherAliveLanes += other.rendering.textureGatherAliveLanes;
		ret.rendering.zBufferWriteLanes += other.rendering.zBufferWriteLanes;
		ret.rendering.zBufferWriteAliveLanes += other.rendering.zBufferWriteAliveLanes;
		ret.rendering.frameBufWriteLanes += other.rendering.frameBufWriteLanes;
		ret.rendering.frameBufWriteAliveLanes += other.rendering.frameBufWriteAliveLanes;
		ret.rendering.renderJobCountProducer += other.rendering.renderJobCountProducer;
		ret.rendering.renderJobCountConsumer += other.rendering.renderJobCountConsumer;

		if (ag.framebufCleanMsMin || other.time.frameBufferCleanMs) ag.framebufCleanMsMin = std::min(ag.framebufCleanMsMin.value_or(INFINITY), other.time.frameBufferCleanMs.value_or(INFINITY));
		if (ag.framebufCleanMsMax || other.time.frameBufferCleanMs) ag.framebufCleanMsMax = std::max(ag.framebufCleanMsMax.value_or(-INFINITY), other.time.frameBufferCleanMs.value_or(-INFINITY));
		if (ag.framebufCleanMsTotal || other.time.frameBufferCleanMs) ag.framebufCleanMsTotal = ag.framebufCleanMsTotal.value_or(0) + other.time.frameBufferCleanMs.value_or(0);

		if (ag.zBufferCleanMsMin || other.time.zBufferCleanMs) ag.zBufferCleanMsMin = std::min(ag.zBufferCleanMsMin.value_or(INFINITY), other.time.zBufferCleanMs.value_or(INFINITY));
		if (ag.zBufferCleanMsMax || other.time.zBufferCleanMs) ag.zBufferCleanMsMax = std::max(ag.zBufferCleanMsMin.value_or(-INFINITY), other.time.zBufferCleanMs.value_or(-INFINITY));
		if (ag.zBufferCleanMsTotal || other.time.zBufferCleanMs) ag.zBufferCleanMsTotal = ag.zBufferCleanMsTotal.value_or(0) + other.time.zBufferCleanMs.value_or(0);

		if (ag.drawMsMin || other.time.drawMs) ag.drawMsMin = std::min(ag.drawMsMin.value_or(INFINITY), other.time.drawMs.value_or(INFINITY));
		if (ag.drawMsMax || other.time.drawMs) ag.drawMsMax = std::max(ag.drawMsMax.value_or(-INFINITY), other.time.drawMs.value_or(-INFINITY));
		if (ag.drawMsTotal || other.time.drawMs) ag.drawMsTotal = ag.drawMsTotal.value_or(0) + other.time.drawMs.value_or(0);

		if (ag.transformMsMin || other.time.transformMs) ag.transformMsMin = std::min(ag.transformMsMin.value_or(INFINITY), other.time.transformMs.value_or(INFINITY));
		if (ag.transformMsMax || other.time.transformMs) ag.transformMsMax = std::max(ag.transformMsMax.value_or(-INFINITY), other.time.transformMs.value_or(-INFINITY));
		if (ag.transformMsTotal || other.time.transformMs) ag.transformMsTotal = ag.transformMsTotal.value_or(0) + other.time.transformMs.value_or(0);
	}
	return { ret,ag };
}