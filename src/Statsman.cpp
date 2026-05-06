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
		ret.allocsByNew += other.allocsByNew;
		ret.freesByDelete += other.freesByDelete;

		ret.rasterizing.trianglesRendered += other.rasterizing.trianglesRendered;
		ret.rasterizing.trianglesTotal += other.rasterizing.trianglesTotal;

		for (int i = 0; i <= 3; ++i)
			ret.rasterizing.verticesBehindNearPlane[i] += other.rasterizing.verticesBehindNearPlane[i];

		ret.rasterizing.vertIndexDelta += other.rasterizing.vertIndexDelta;
		ret.rasterizing.vertIndexDeltaCount += other.rasterizing.vertIndexDeltaCount;

		if (other.rasterizing.vertIndexDeltaMin)
			ret.rasterizing.vertIndexDeltaMin =
			std::min(ret.rasterizing.vertIndexDeltaMin.value_or(UINT64_MAX),
				*other.rasterizing.vertIndexDeltaMin);

		if (other.rasterizing.vertIndexDeltaMax)
			ret.rasterizing.vertIndexDeltaMax =
			std::max(ret.rasterizing.vertIndexDeltaMax.value_or(0),
				*other.rasterizing.vertIndexDeltaMax);

		ret.rasterizing.barycentricsCalculated += other.rasterizing.barycentricsCalculated;
		ret.rasterizing.pointsInsideTriangles += other.rasterizing.pointsInsideTriangles;
		ret.rasterizing.notOccludedPoints += other.rasterizing.notOccludedPoints;
		ret.rasterizing.opaquePixels += other.rasterizing.opaquePixels;
		ret.rasterizing.textureGatheredLanes += other.rasterizing.textureGatheredLanes;
		ret.rasterizing.textureGatherAliveLanes += other.rasterizing.textureGatherAliveLanes;
		ret.rasterizing.zBufferFetchLanes += other.rasterizing.zBufferFetchLanes;
		ret.rasterizing.zBufferFetchAliveLanes += other.rasterizing.zBufferFetchAliveLanes;
		ret.rasterizing.zBufferWriteLanes += other.rasterizing.zBufferWriteLanes;
		ret.rasterizing.zBufferWriteAliveLanes += other.rasterizing.zBufferWriteAliveLanes;
		ret.rasterizing.frameBufWriteLanes += other.rasterizing.frameBufWriteLanes;
		ret.rasterizing.frameBufWriteAliveLanes += other.rasterizing.frameBufWriteAliveLanes;
		ret.rasterizing.renderJobCountProducer += other.rasterizing.renderJobCountProducer;
		ret.rasterizing.renderJobCountConsumer += other.rasterizing.renderJobCountConsumer;

		if (ag.framebufCleanMsMin || other.rasterizing.frameBufferCleanMs)
			ag.framebufCleanMsMin =
			std::min(ag.framebufCleanMsMin.value_or(INFINITY),
				other.rasterizing.frameBufferCleanMs.value_or(INFINITY));

		if (ag.framebufCleanMsMax || other.rasterizing.frameBufferCleanMs)
			ag.framebufCleanMsMax =
			std::max(ag.framebufCleanMsMax.value_or(-INFINITY),
				other.rasterizing.frameBufferCleanMs.value_or(-INFINITY));

		if (ag.framebufCleanMsTotal || other.rasterizing.frameBufferCleanMs)
			ag.framebufCleanMsTotal =
			ag.framebufCleanMsTotal.value_or(0) +
			other.rasterizing.frameBufferCleanMs.value_or(0);

		if (ag.zBufferCleanMsMin || other.rasterizing.zBufferCleanMs)
			ag.zBufferCleanMsMin =
			std::min(ag.zBufferCleanMsMin.value_or(INFINITY),
				other.rasterizing.zBufferCleanMs.value_or(INFINITY));

		if (ag.zBufferCleanMsMax || other.rasterizing.zBufferCleanMs)
			ag.zBufferCleanMsMax =
			std::max(ag.zBufferCleanMsMax.value_or(-INFINITY),
				other.rasterizing.zBufferCleanMs.value_or(-INFINITY));

		if (ag.zBufferCleanMsTotal || other.rasterizing.zBufferCleanMs)
			ag.zBufferCleanMsTotal =
			ag.zBufferCleanMsTotal.value_or(0) +
			other.rasterizing.zBufferCleanMs.value_or(0);

		if (ag.drawMsMin || other.rasterizing.drawMs)
			ag.drawMsMin =
			std::min(ag.drawMsMin.value_or(INFINITY),
				other.rasterizing.drawMs.value_or(INFINITY));

		if (ag.drawMsMax || other.rasterizing.drawMs)
			ag.drawMsMax =
			std::max(ag.drawMsMax.value_or(-INFINITY),
				other.rasterizing.drawMs.value_or(-INFINITY));

		if (ag.drawMsTotal || other.rasterizing.drawMs)
			ag.drawMsTotal =
			ag.drawMsTotal.value_or(0) +
			other.rasterizing.drawMs.value_or(0);

		if (ag.transformMsMin || other.rasterizing.transformMs)
			ag.transformMsMin =
			std::min(ag.transformMsMin.value_or(INFINITY),
				other.rasterizing.transformMs.value_or(INFINITY));

		if (ag.transformMsMax || other.rasterizing.transformMs)
			ag.transformMsMax =
			std::max(ag.transformMsMax.value_or(-INFINITY),
				other.rasterizing.transformMs.value_or(-INFINITY));

		if (ag.transformMsTotal || other.rasterizing.transformMs)
			ag.transformMsTotal =
			ag.transformMsTotal.value_or(0) +
			other.rasterizing.transformMs.value_or(0);
	}

	return { ret, ag };
}