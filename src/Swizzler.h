#pragma once
#include <stdint.h>
#include "Vec.h"

struct SwizzlerParams
{
	uint32_t w, h, paddedW, paddedH, tileSizePerAxis, tileArea, tileBitShift, tileBitwiseAndMask, tileCountX, tileCountY;
};

//Class that performs remapping of integer coordinates (X,Y) to memory indices that the elements reside in based on parameters supplied at class instance's creation
//Supports only fixed-size row-major tiling
class Swizzler
{
public:
	Swizzler() = default;
	Swizzler(uint32_t w, uint32_t h, uint32_t tileBitShift);
	const SwizzlerParams& getParams() const;

	template<typename T>
	__forceinline T getIndicesForXY(T x, T y) const
		requires (std::is_integral_v<T> || std::is_same_v<T, int32x16>)
	{
		T tileIndexX = x >> this->params.tileBitShift;
		T tileIndexY = y >> this->params.tileBitShift;
		T insideTileX = x & this->params.tileBitwiseAndMask;
		T insideTileY = y & this->params.tileBitwiseAndMask;
		T ret = (tileIndexY * params.tileCountX + tileIndexX) * params.tileArea + ((insideTileY << params.tileBitShift) + insideTileX);
		return ret;
	}
private:
	SwizzlerParams params;
};