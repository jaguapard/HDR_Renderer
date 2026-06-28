#include "Mappers.h"

WrappingMapper::WrappingMapper(uint32_t w, uint32_t h)
{
	this->params.w = w;
	this->params.h = h;
	this->params.fw = w;
	this->params.fh = h;
	this->params.u64_rcpW = uint64_t(1ull << 32) / w;
	this->params.u64_rcpH = uint64_t(1ull << 32) / h;
	this->params.f64_rcpW = 1.0 / w;
	this->params.f64_rcpH = 1.0 / h;
	this->params.f64_w = w;
	this->params.f64_h = h;
}