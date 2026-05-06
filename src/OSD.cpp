#include "OSD.h"
#include <numeric>
#include "Statsman.h"
#include <iostream>
#include "helpers.h"
//#include "Statsman.h"
#include <sstream>

OSD::OSD(uint32_t fontSize)
{
	if (!fontSize) fontSize = 22;
	std::string path = "C:/Windows/Fonts/lucon.ttf";
	font = Smart_Font(TTF_OpenFont(path.c_str(), fontSize));
	if (!font) throw std::runtime_error("Failed to open main font: " + path);
	fontOutline = Smart_Font(TTF_OpenFont(path.c_str(), fontSize));
	TTF_SetFontOutline(fontOutline.get(), 1);
	if (!fontOutline) throw std::runtime_error("Failed to open font outline: " + path);
}

void OSD::reset()
{
	frameTimesMs.clear();
	timer = bob::Timer();
}

void OSD::registerFrameBegin()
{
	//oldStats = statsman;
}

void OSD::registerFrameDone(bool remember)
{
	double time = timer.getTimeSinceLastPoll() * 1000;
	if (remember)
	{
		++frameNumber;
		frameTimesMs.push_back(time);
		if (frameTimesMs.size() > 1000) frameTimesMs.pop_front();
	}
}

Smart_Surface OSD::draw(float scalingFactor, const std::vector<std::pair<std::string, std::string>>& additionalInfo)
{
	std::string str = composeString(additionalInfo);

	Smart_Surface s = Smart_Surface(TTF_RenderText_LCD_Wrapped(this->font.get(), str.c_str(), str.length(), { 255,255,255,255 }, { 0,0,0,255 }, 1500)); //alpha channel is always 1!
	//Small fonts can be bitmap fonts, that will return null on RenderText_LCD. It's a workaround, since it doesn't look like there's a way to check before rendering. TTF_IsFontScalable returns true even for small font (Lucida Console 6pt for instance)
	if (!s) s = Smart_Surface(TTF_RenderText_Blended_Wrapped(this->font.get(), str.c_str(), str.length(), { 255,255,255,255 }, 1500));
	//auto s = Smart_Surface(TTF_RenderText(font.get(), str.c_str(), str.length(), { 255,255,255,255 }, { 0,0,0,255 }, 1500)); //alpha channel is always 1!
	//auto s = Smart_Surface(TTF_RenderText_Blended_Wrapped(font.get(), str.c_str(), 0, { 255,255,255,255 }, 1000));
	if (!s) throw std::runtime_error("Failed to draw OSD text");

	SDL_PixelFormat outputFormat = SDL_PIXELFORMAT_RGBA128_FLOAT;
	int scaledW = scalingFactor * s->w;
	int scaledH = scalingFactor * s->h;
	auto s_scaled = Smart_Surface(SDL_CreateSurface(scaledW, scaledH, outputFormat));
	if (!s_scaled) throw std::runtime_error("Failed to create scaled surface in OSD::draw.");

	if (!SDL_BlitSurfaceScaled(s.get(), nullptr, s_scaled.get(), nullptr, SDL_SCALEMODE_LINEAR)) throw std::runtime_error("Failed to blit scaled OSD surface.");
	return s_scaled;
	/*
	auto conv = Smart_Surface(SDL_ConvertSurface(s.get(), outputFormat));
	if (!conv) throw std::runtime_error("Failed to convert OSD surface");
	return conv;*/

	//auto bg = Smart_Surface(TTF_RenderText_LCD_Wrapped(fontOutline.get(), str.c_str(), 0, { 0,0,0,SDL_ALPHA_OPAQUE }, 1500));
	//auto fg = Smart_Surface(TTF_RenderText_Blended_Wrapped(font.get(), str.c_str(), 0, { 255,255,255,SDL_ALPHA_OPAQUE }, 1500));
	//TTF_RenderText_LCD_Wrapped
	/*
	

	if (!bg || !fg) throw std::runtime_error(std::string("Failed to draw FPS info: ") + SDL_GetError());

	SDL_SetSurfaceBlendMode(bg.get(), SDL_BLENDMODE_BLEND);
	SDL_BlitSurface(fg.get(), nullptr, bg.get(), nullptr);

	auto nfg = Smart_Surface(SDL_ConvertSurface(bg.get(), SDL_PIXELFORMAT_ABGR8888));
	//return { std::move(fg),std::move(bg) };
	return { std::move(nfg),std::move(bg) };*/
	//SDL_Rect rect1 = { pixelsFromUpperLeftCorner.x, pixelsFromUpperLeftCorner.y, fg->w, fg->h };
	//SDL_Rect rect2 = { pixelsFromUpperLeftCorner.x, pixelsFromUpperLeftCorner.y, bg->w, bg->h };

	//SDL_SetSurfaceBlendMode(bg.get(), SDL_BLENDMODE_BLEND);
	//SDL_BlitSurface(bg.get(), nullptr, dst, &rect1);
	//SDL_BlitSurface(fg.get(), nullptr, dst, &rect2);
}

OSD::PercentileInfo OSD::getPercentileInfo() const
{
	return PercentileInfo(this->frameNumber, frameTimesMs);
}

uint64_t OSD::getFrameNumber() const
{
	return frameNumber;
}

std::string laneSurvivalRateString(uint64_t laneCount, uint64_t aliveCount)
{
	return toThousandsSeparatedString(laneCount) + " (" + toThousandsSeparatedString(aliveCount) + " alive, " + std::to_string(aliveCount * 100.0 / laneCount) + "%)";
}
std::string OSD::composeString(const std::vector<std::pair<std::string, std::string>>& additionalInfo)
{
	std::string text = PercentileInfo(frameNumber, frameTimesMs).toString();
	std::stringstream ss;

	ss << timer.getTime() << " sec\n";
	ss << text;

	/*
	if (true && Statsman::display_enabled) //statsman stuff
	{
		ss << (statsman - oldStats).toString() << "\n";
	}*/

	for (const auto& kv : additionalInfo)
	{
		ss << kv.first << ": " << kv.second << "\n";
	}

	if (Statsman::ENABLED)
	{
		ss << "\n";
		auto [s, ag] = Statsman::aggregateAll();

		ss << toThousandsSeparatedString(s.rasterizing.trianglesRendered) << " triangles rendered\n";
		ss << toThousandsSeparatedString(s.rasterizing.trianglesTotal) << " total triangles\n";

		ss << "Render jobs: "
			<< toThousandsSeparatedString(s.rasterizing.renderJobCountProducer)
			<< " (producer-side), "
			<< toThousandsSeparatedString(s.rasterizing.renderJobCountConsumer)
			<< " (consumer-side, "
			<< s.rasterizing.renderJobCountConsumer * 100.0 / s.rasterizing.renderJobCountProducer
			<< "%)\n";

		ss << "Vertices behind near plane: ";
		for (int i = 0; i < 4; ++i)
			ss << i << ": "
			<< toThousandsSeparatedString(s.rasterizing.verticesBehindNearPlane[i])
			<< (i != 3 ? ", " : "");

		ss << "\nVertice index deltas: "
			<< double(s.rasterizing.vertIndexDelta) / s.rasterizing.vertIndexDeltaCount
			<< " avg, "
			<< s.rasterizing.vertIndexDeltaMax.value_or(NAN)
			<< " max, "
			<< s.rasterizing.vertIndexDeltaMin.value_or(NAN)
			<< " min\n";

		ss << "Memory allocations by new: "
			<< toThousandsSeparatedString(s.allocsByNew)
			<< ", frees by delete: "
			<< toThousandsSeparatedString(s.freesByDelete)
			<< ", new - delete: "
			<< toThousandsSeparatedString(s.allocsByNew - s.freesByDelete)
			<< "\n";

		double count = Statsman::statsmenForThreads.size();

		ss << "\n";
		ss << "Barycentircs calculated: "
			<< toThousandsSeparatedString(s.rasterizing.barycentricsCalculated) << "\n"
			<< "Points inside triangles: "
			<< toThousandsSeparatedString(s.rasterizing.pointsInsideTriangles) << "\n"
			<< "Depth buffer fetch lanes: "
			<< laneSurvivalRateString(s.rasterizing.zBufferFetchLanes, s.rasterizing.zBufferFetchAliveLanes) << "\n"
			<< "Not occluded points: "
			<< toThousandsSeparatedString(s.rasterizing.notOccludedPoints) << "\n"
			<< "Opaque pixels: "
			<< toThousandsSeparatedString(s.rasterizing.opaquePixels) << "\n"
			<< "Texture gather lanes: "
			<< laneSurvivalRateString(s.rasterizing.textureGatheredLanes, s.rasterizing.textureGatherAliveLanes) << "\n"
			<< "Depth buffer write lanes: "
			<< laneSurvivalRateString(s.rasterizing.zBufferWriteLanes, s.rasterizing.zBufferWriteAliveLanes) << "\n"
			<< "Frame buffer write lanes: "
			<< laneSurvivalRateString(s.rasterizing.frameBufWriteLanes, s.rasterizing.frameBufWriteAliveLanes) << "\n\n"
			<< "Transformation times: "
			<< ag.transformMsMax.value_or(NAN) << " ms max, "
			<< ag.transformMsTotal.value_or(NAN) / count << " ms avg, "
			<< ag.transformMsMin.value_or(NAN) << " ms min\n"
			<< "Draw times: "
			<< ag.drawMsMax.value_or(NAN) << " ms max, "
			<< ag.drawMsTotal.value_or(NAN) / count << " ms avg, "
			<< ag.drawMsMin.value_or(NAN) << " ms min\n"
			<< "Depth buffer clean times: "
			<< ag.zBufferCleanMsMax.value_or(NAN) << " ms max\n"
			<< "Frame buffer clean times: "
			<< ag.framebufCleanMsMax.value_or(NAN) << " ms max\n";
	}
	return ss.str();
}


OSD::PercentileInfo::PercentileInfo(uint64_t frameNumber, std::deque<double> frameTimesMs)
{
	this->frameNumber = frameNumber;
	if (frameTimesMs.empty()) return;

	fps_inst = 1000 / frameTimesMs.back();
	double totalTime = std::accumulate(frameTimesMs.begin(), frameTimesMs.end(), 0);
	fps_avg = 1000 / (totalTime / frameTimesMs.size());

	std::sort(frameTimesMs.rbegin(), frameTimesMs.rend());
	int n = frameTimesMs.size();
	if (frameTimesMs.size() >= 100) fps_1pct_low_by_count = 1000.0 / frameTimesMs[n / 100 - 1];
	if (frameTimesMs.size() >= 1000) fps_point1pct_low_by_count = 1000.0 / frameTimesMs[n / 1000 - 1];

	double acc = 0;
	for (auto& ms : frameTimesMs)
	{
		acc += ms;
		if (!fps_1pct_low_by_time && acc >= totalTime / 100) fps_1pct_low_by_time = 1000/ms;
		if (!fps_point1pct_low_by_time && acc >= totalTime / 1000) fps_point1pct_low_by_time = 1000/ms;
	}
}

std::string OSD::PercentileInfo::toString()
{
	std::stringstream text;
	text << "Frame " << frameNumber << "\n";

	if (fps_inst) text << *fps_inst << " FPS inst\n" << 1000/fps_inst.value() << " ms";
	else text << "n/a FPS inst";
	text << "\n\n";
	
	if (fps_avg) text << *fps_avg;
	else text << "n/a";
	text << " FPS avg\n";

	text << "1% low: ";
	if (fps_1pct_low_by_count) text << *fps_1pct_low_by_count << " (by count), ";
	else text << "n/a (by count), ";
	if (fps_1pct_low_by_time) text << *fps_1pct_low_by_time << " (by time)";
	else text << "n/a (by time)";

	text << "\n0.1% low: ";
	if (fps_point1pct_low_by_count) text << *fps_point1pct_low_by_count << " (by count), ";
	else text << "n/a (by count), ";
	if (fps_point1pct_low_by_time) text << *fps_point1pct_low_by_time << " (by time)";
	else text << "n/a (by time)";
	text << "\n\n";
	return text.str();
}
