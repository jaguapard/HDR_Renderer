#include "Swizzler.h"
#include <cmath>
#include <stdexcept>

Swizzler::Swizzler(uint32_t w, uint32_t h, uint32_t tileBitShift)
{
	if (w == 0 || h == 0 || tileBitShift == 0 || tileBitShift >= 32) throw std::runtime_error("Incorrect parameters given to swizzler");
	this->params.w = w;
	this->params.h = h;
	this->params.tileSizePerAxis = 1 << tileBitShift;
	this->params.tileArea = params.tileSizePerAxis * params.tileSizePerAxis;
	this->params.tileCountX = std::ceil(double(w) / this->params.tileSizePerAxis);
	this->params.tileCountY = std::ceil(double(h) / this->params.tileSizePerAxis);
	this->params.paddedW = this->params.tileCountX * this->params.tileSizePerAxis;
	this->params.paddedH = this->params.tileCountY * this->params.tileSizePerAxis;
	this->params.tileBitShift = tileBitShift;
	this->params.tileBitwiseAndMask = this->params.tileSizePerAxis - 1;
}

const SwizzlerParams& Swizzler::getParams() const
{
	return this->params;
}
