#pragma once
#include <vector>

struct alignas(64) Statsman
{
	static constexpr bool ENABLED = true;
	static inline std::vector<Statsman> statsmenForThreads;

	void reset();

	struct Triangles
	{
		uint64_t total, rendered, verticesBehindNearPlane[4];
	};
	
	struct Rendering
	{
		uint64_t barycentricsCalculated, pointsInsideTriangles, visiblePoints, opaquePixels, textureGatheredLanes, textureGatherAliveLanes, zBufferFetchLanes, zBufferFetchAliveLanes, zBufferWriteLanes, zBufferWriteAliveLanes, frameBufWriteLanes, frameBufWriteAliveLanes;
		
	};

	struct Time
	{
		double transformMs, drawMs;
	};
	//these parametes are auto-calculated in operator+ from multiple Statsman insances
	struct Aggregated
	{
		double transformMsMin, transformMsMax, transformMsTotal, drawMsMin, drawMsMax, drawMsTotal;
	};
	Statsman() { reset(); }
	Triangles triangles;
	Rendering rendering;
	Time time;
	Statsman operator+(const Statsman& other) const;
	Aggregated getAggregatedInfo() const;

private:
	Aggregated ag;
};

//#define StatCount(__threadIndex, action) if (Statsman::ENABLED) {Statsman::statsmenForThreads[__threadIndex].action;};
#define StatCount(action) if (Statsman::ENABLED) Statsman::statsmenForThreads[threadIndex].action;};
#define MyStatsman Statsman::statsmenForThreads[threadIndex]
//#define MyStatsman(__variableName) Statsman::statsmenForThreads[__variableName]