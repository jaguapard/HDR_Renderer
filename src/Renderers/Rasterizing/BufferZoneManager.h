#pragma once
namespace Rasterizing
{
	class BufferZoneManager
	{
	public:
		BufferZoneManager() {};
		BufferZoneManager(int threadCount, int w, int h);
		//Writes out threads responsible for this bounding box into out. Returns recorded element count. out must be large enough to store at least threadCount elements 
		int getThreadsResponsible(int* out, float minX, float minY, float maxX, float maxY) const;
		void getLimitsForThread(int threadIndex, float& minX, float& minY, float& maxX, float& maxY) const;
	private:
		int threadCount, w, h;
	};
}