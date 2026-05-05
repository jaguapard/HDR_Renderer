#pragma once
#include "bob/Timer.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_ttf.h>
#include <deque>
#include <map>

#include "smart.h"
//#include "Statsman.h"
#include "Vec.h"

class OSD
{
public:
	OSD();
	void reset();
	void registerFrameBegin(); //must be called before any drawing and processing is attempted
	void registerFrameDone(bool remember = true);

	std::string composeString(const std::vector<std::pair<std::string, std::string>>& additionalInfo = {});
	Smart_Surface draw(float scalingFactor, const std::vector<std::pair<std::string, std::string>>& additionalInfo = {});

	struct PercentileInfo
	{
		PercentileInfo(uint64_t frameNumber, std::deque<double> frameTimesMs); //copy is intentional, since constructor will garble the data
		std::string toString();

		std::optional<double> fps_inst, fps_avg, fps_1pct_low_by_count, fps_1pct_low_by_time, fps_point1pct_low_by_count, fps_point1pct_low_by_time;
		uint64_t frameNumber;
	};

	PercentileInfo getPercentileInfo() const;
	uint64_t getFrameNumber() const;
private:
	Smart_Font font;
	Smart_Font fontOutline;
	bob::Timer timer;

	//Statsman oldStats;
	uint64_t frameNumber = 0;
	std::deque<double> frameTimesMs;
};