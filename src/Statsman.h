#pragma once
#include <vector>
#include <optional>

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
		std::optional<double> transformMs, drawMs, zBufferCleanMs, frameBufferCleanMs;
	};
	//these parametes are auto-calculated in operator+ from multiple Statsman insances
	struct Aggregated
	{
		std::optional<double> transformMsMin, transformMsMax, transformMsTotal, drawMsMin, drawMsMax, drawMsTotal, zBufferCleanMsMin, zBufferCleanMsMax, zBufferCleanMsTotal, framebufCleanMsMin, framebufCleanMsMax, framebufCleanMsTotal;
	};
	Statsman() { reset(); }
	Triangles triangles;
	Rendering rendering;
	Time time;

	static std::pair<Statsman,Aggregated> aggregateAll();

private:
};

//#define StatCount(__threadIndex, action) if (Statsman::ENABLED) {Statsman::statsmenForThreads[__threadIndex].action;};
#define StatCount(action) if (Statsman::ENABLED) Statsman::statsmenForThreads[threadIndex].action;};
#define MyStatsman Statsman::statsmenForThreads[threadIndex]
//#define MyStatsman(__variableName) Statsman::statsmenForThreads[__variableName]