#pragma once
#include <vector>
#include <optional>
#include <atomic>

class OSD;
struct Statsman;
struct StatsmanLine
{
	friend struct Statsman;
	friend class OSD;

	StatsmanLine() = default;
	StatsmanLine(double d);
	operator double() const;
	StatsmanLine& operator=(double d);
	StatsmanLine operator+(double d) const;
	StatsmanLine operator+(const StatsmanLine& line) const;
	StatsmanLine& operator+=(const StatsmanLine& line);
	StatsmanLine operator-(double d) const;
	StatsmanLine operator*(double d) const;
	StatsmanLine operator/(double d) const;
	StatsmanLine& operator+=(double d);
	StatsmanLine& operator-=(double d);
	StatsmanLine& operator*=(double d);
	StatsmanLine& operator/=(double d);
	StatsmanLine& operator++();
	StatsmanLine operator++(int);
	//StatsmanLine& operator=
	
	bool isAggregate() const; //unknowable if all opts are null
private:
	std::optional<double> value, min, max, sum;
	StatsmanLine& aggregateWith(const StatsmanLine& other);
};
struct alignas(64) Statsman
{

	/*
#if defined(REL_DBG) || !defined(NDEBUG)
	static constexpr bool ENABLED = true;
#else
	static constexpr bool ENABLED = false;
#endif
*/

	static constexpr bool ENABLED = true;
	void reset();

	struct Rasterizing
	{
		StatsmanLine trianglesTotal, trianglesRendered, verticesBehindNearPlane[4], vertIndexDelta, vertIndexDeltaCount, vertIndexDeltaMin, vertIndexDeltaMax;

		StatsmanLine barycentricsCalculated, pointsInsideTriangles, notOccludedPoints, opaquePixels, textureGatheredLanes, textureGatherAliveLanes, zBufferFetchLanes, zBufferFetchAliveLanes, zBufferWriteLanes, zBufferWriteAliveLanes, frameBufWriteLanes, frameBufWriteAliveLanes, renderJobCountProducer, renderJobCountConsumer;

		StatsmanLine transformMs, drawMs, zBufferCleanMs, frameBufferCleanMs, triangleIndexBufferCleanMs, shadowMapDepthBufferCleanMs;
	};
	

	/*
	//these parametes are auto-calculated in aggregateAll() from multiple Statsman insances
	struct Aggregated
	{
		std::optional<double> transformMsMin, transformMsMax, transformMsTotal, drawMsMin, drawMsMax, drawMsTotal, zBufferCleanMsMin, zBufferCleanMsMax, zBufferCleanMsTotal, framebufCleanMsMin, framebufCleanMsMax, framebufCleanMsTotal, triangleIndexBufferCleanMsMin, triangleIndexBufferCleanMsMax, triangleIndexBufferCleanMsTotal, shadowMapDepthBufferCleanMsMin, shadowMapDepthBufferCleanMsMax, shadowMapDepthBufferCleanMsTotal;
	};
	
	*/
	Statsman() { reset(); }
	Statsman(const Statsman& s) { memcpy(this, &s, sizeof(*this)); }
	Rasterizing rasterizing;

	std::atomic<uint64_t> allocsByNew, freesByDelete;

	static Statsman aggregateAll();
	//static std::pair<Statsman,Aggregated> aggregateAll();

	static std::vector<Statsman> statsmenForThreads;
};

//#define StatCount(__threadIndex, action) if (Statsman::ENABLED) {Statsman::statsmenForThreads[__threadIndex].action;};
#define StatCount(action) if (Statsman::ENABLED) Statsman::statsmenForThreads[threadIndex].action;};
#define MyStatsman Statsman::statsmenForThreads[threadIndex]
//#define MyStatsman(__variableName) Statsman::statsmenForThreads[__variableName]

inline std::vector<Statsman> Statsman::statsmenForThreads;