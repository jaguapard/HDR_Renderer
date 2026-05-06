#pragma once
#include <vector>
#include <optional>
#include <atomic>

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
		uint64_t trianglesTotal, trianglesRendered, verticesBehindNearPlane[4], vertIndexDelta, vertIndexDeltaCount;
		std::optional<uint64_t> vertIndexDeltaMin, vertIndexDeltaMax;

		uint64_t barycentricsCalculated, pointsInsideTriangles, notOccludedPoints, opaquePixels, textureGatheredLanes, textureGatherAliveLanes, zBufferFetchLanes, zBufferFetchAliveLanes, zBufferWriteLanes, zBufferWriteAliveLanes, frameBufWriteLanes, frameBufWriteAliveLanes, renderJobCountProducer, renderJobCountConsumer;

		std::optional<double> transformMs, drawMs, zBufferCleanMs, frameBufferCleanMs;
	};
	

	//these parametes are auto-calculated in aggregateAll() from multiple Statsman insances
	struct Aggregated
	{
		std::optional<double> transformMsMin, transformMsMax, transformMsTotal, drawMsMin, drawMsMax, drawMsTotal, zBufferCleanMsMin, zBufferCleanMsMax, zBufferCleanMsTotal, framebufCleanMsMin, framebufCleanMsMax, framebufCleanMsTotal;
	};
	Statsman() { reset(); }
	Statsman(const Statsman& s) {memcpy(this, &s, sizeof(*this));}

	Rasterizing rasterizing;

	std::atomic<uint64_t> allocsByNew, freesByDelete;

	static std::pair<Statsman,Aggregated> aggregateAll();

	static std::vector<Statsman> statsmenForThreads;
};

//#define StatCount(__threadIndex, action) if (Statsman::ENABLED) {Statsman::statsmenForThreads[__threadIndex].action;};
#define StatCount(action) if (Statsman::ENABLED) Statsman::statsmenForThreads[threadIndex].action;};
#define MyStatsman Statsman::statsmenForThreads[threadIndex]
//#define MyStatsman(__variableName) Statsman::statsmenForThreads[__variableName]

inline std::vector<Statsman> Statsman::statsmenForThreads;