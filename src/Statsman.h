#pragma once
#include <vector>
#include <optional>
#include <atomic>

struct alignas(64) Statsman
{
#if 1 || defined(REL_DBG) || !defined(NDEBUG)
	static constexpr bool ENABLED = true;
#else
	static constexpr bool ENABLED = false;
#endif
	void reset();

	struct Triangles
	{
		uint64_t total, rendered, verticesBehindNearPlane[4], vertIndexDelta, vertIndexDeltaCount;
		std::optional<uint64_t> vertIndexDeltaMin, vertIndexDeltaMax;
	};
	
	struct Rendering
	{
		uint64_t barycentricsCalculated, pointsInsideTriangles, notOccludedPoints, opaquePixels, textureGatheredLanes, textureGatherAliveLanes, zBufferFetchLanes, zBufferFetchAliveLanes, zBufferWriteLanes, zBufferWriteAliveLanes, frameBufWriteLanes, frameBufWriteAliveLanes, renderJobCountProducer, renderJobCountConsumer;
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
	Statsman(const Statsman& s) {memcpy(this, &s, sizeof(*this));}
	Triangles triangles;
	Rendering rendering;
	Time time;
	std::atomic<uint64_t> allocsByNew, freesByDelete;

	static std::pair<Statsman,Aggregated> aggregateAll();

	static std::vector<Statsman> statsmenForThreads;
};

//#define StatCount(__threadIndex, action) if (Statsman::ENABLED) {Statsman::statsmenForThreads[__threadIndex].action;};
#define StatCount(action) if (Statsman::ENABLED) Statsman::statsmenForThreads[threadIndex].action;};
#define MyStatsman Statsman::statsmenForThreads[threadIndex]
//#define MyStatsman(__variableName) Statsman::statsmenForThreads[__variableName]

inline std::vector<Statsman> Statsman::statsmenForThreads;