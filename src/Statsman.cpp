#include "Statsman.h"
#include <assert.h>

void Statsman::reset()
{
	memset(this, 0, sizeof(*this));
	allocsByNew = freesByDelete = 0;
}

Statsman Statsman::aggregateAll()
{
	Statsman ret;

	for (auto& other : Statsman::statsmenForThreads)
	{
		ret.allocsByNew += other.allocsByNew;
		ret.freesByDelete += other.freesByDelete;

		ret.rasterizing.trianglesRendered += other.rasterizing.trianglesRendered;
		ret.rasterizing.trianglesTotal += other.rasterizing.trianglesTotal;
		for (int i = 0; i <= 3; ++i) ret.rasterizing.verticesBehindNearPlane[i] += other.rasterizing.verticesBehindNearPlane[i];

		ret.rasterizing.vertIndexDelta += other.rasterizing.vertIndexDelta;
		ret.rasterizing.vertIndexDeltaCount += other.rasterizing.vertIndexDeltaCount;
		ret.rasterizing.vertIndexDelta.aggregateWith(other.rasterizing.vertIndexDelta);

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

		ret.rasterizing.frameBufferCleanMs.aggregateWith(other.rasterizing.frameBufferCleanMs);
		ret.rasterizing.zBufferCleanMs.aggregateWith(other.rasterizing.zBufferCleanMs);
		ret.rasterizing.triangleIndexBufferCleanMs.aggregateWith(other.rasterizing.triangleIndexBufferCleanMs);
		ret.rasterizing.shadowMapDepthBufferCleanMs.aggregateWith(other.rasterizing.shadowMapDepthBufferCleanMs);

		ret.rasterizing.drawMs.aggregateWith(other.rasterizing.drawMs);
		ret.rasterizing.transformMs.aggregateWith(other.rasterizing.transformMs);

		ret.rayCasting.triangleIntersectionTests += other.rayCasting.triangleIntersectionTests;
		ret.rayCasting.triangleIntersectionTestsLive += other.rayCasting.triangleIntersectionTestsLive;
		ret.rayCasting.nodesInspected += other.rayCasting.nodesInspected;
		ret.rayCasting.trianglesInspected += other.rayCasting.trianglesInspected;
		ret.rayCasting.rayNodeIntersections += other.rayCasting.rayNodeIntersections;
		ret.rayCasting.rayNodeIntersectionTests += other.rayCasting.rayNodeIntersectionTests;
	}
	return ret;
}

StatsmanLine::StatsmanLine(double d)
{
	value = d;
}

StatsmanLine::operator double() const
{
	return value.value_or(0);
}

StatsmanLine& StatsmanLine::operator=(double d)
{
	value = d;
	return *this;
}

StatsmanLine StatsmanLine::operator+(double d) const
{
	auto ret = *this;
	ret.value = ret.value.value_or(0) + d;
	return ret;
}

StatsmanLine StatsmanLine::operator+(const StatsmanLine& line) const
{
	StatsmanLine ret;
	if (value || line.value) ret.value = value.value_or(0) + line.value.value_or(0);
	return ret;
}

StatsmanLine& StatsmanLine::operator+=(const StatsmanLine& line)
{
	*this = *this + line;
	return *this;
}

StatsmanLine StatsmanLine::operator-(double d) const
{
	auto ret = *this;
	ret.value = ret.value.value_or(0) - d;
	return ret;
}

StatsmanLine StatsmanLine::operator*(double d) const
{
	auto ret = *this;
	ret.value = ret.value.value_or(0) * d;
	return ret;
}

StatsmanLine StatsmanLine::operator/(double d) const
{
	auto ret = *this;
	ret.value = ret.value.value_or(0) / d;
	return ret;
}

StatsmanLine& StatsmanLine::operator+=(double d)
{
	*this = *this + d;
	return *this;
}

StatsmanLine& StatsmanLine::operator-=(double d)
{
	*this = *this - d;
	return *this;
}

StatsmanLine& StatsmanLine::operator*=(double d)
{
	*this = *this * d;
	return *this;
}

StatsmanLine& StatsmanLine::operator/=(double d)
{
	*this = *this / d;
	return *this;
}

StatsmanLine& StatsmanLine::aggregateWith(const StatsmanLine& other)
{
	assert(!other.min && !other.max && !other.sum);
	assert(!this->value);
	constexpr double inf = std::numeric_limits<double>::max();
	
	if (other.value)
	{
		double oval = *other.value;
		this->min = std::min(this->min.value_or(inf), oval);
		this->max = std::max(this->max.value_or(-inf), oval);
		this->sum = this->sum.value_or(0) + oval;
	}
	return *this;
}

StatsmanLine& StatsmanLine::operator++()
{
	return (*this += 1);
}

StatsmanLine StatsmanLine::operator++(int)
{
	StatsmanLine tmp = *this;
	++(*this);
	return tmp;
}

bool StatsmanLine::isAggregate() const
{
	if (min || max || sum) return true;
	return false;
}
