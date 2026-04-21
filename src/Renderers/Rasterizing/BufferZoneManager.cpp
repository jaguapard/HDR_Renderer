#include "BufferZoneManager.h"
#include "../../Threadpool.h"
#include <cassert>
#include <algorithm>
Rasterizing::BufferZoneManager::BufferZoneManager(int threadCount, int w, int h)
{
	this->threadCount = threadCount;
	this->w = w;
	this->h = h;
}

int Rasterizing::BufferZoneManager::getThreadsResponsible(int* out, float minX, float minY, float maxX, float maxY) const
{
	assert(minY < maxY);
	if (minY >= h || maxY < 0) return 0;
	double perThread = h / double(this->threadCount);
	int beg = minY / perThread;
	int end = ceil(maxY / perThread);
	beg = std::clamp(beg, 0, threadCount - 1);
	end = std::clamp(end, 0, threadCount - 1);
	for (int i = beg; i <= end; ++i) //TODO: should it be <=?
	{
		*out++ = i;
	}
	return end - beg + 1;
}

void Rasterizing::BufferZoneManager::getLimitsForThread(int threadIndex, float& minX, float& minY, float& maxX, float& maxY) const
{
	minX = 0;
	maxX = w - 1;
	auto [d_low, d_high] = Threadpool::instance->getLimitsForThread(threadIndex, 0, this->h, this->threadCount);
	minY = floor(d_low);
	maxY = std::min<float>(d_high, h - 1);
}
