#pragma once
#include <stdint.h>
#include "Vec.h"
#include <stdexcept>

struct SwizzlerParams
{
	uint32_t w, h, paddedW, paddedH, tileCountX, tileCountY;
};

template<typename T>
struct SwizzlerLocations
{
	T tileIndex, tileIndexX, tileIndexY, insideTileIndex, insideTileX, insideTileY;
};

//Class that performs remapping of integer coordinates (X,Y) to memory indices that the elements reside in based on parameters supplied at class instance's creation
//Supports only fixed-size row-major tiling
template<uint32_t _Log2_PerAxisSize>
requires (_Log2_PerAxisSize >= 1 && _Log2_PerAxisSize < 32)
class Swizzler
{
public:
	static constexpr uint32_t Log2_PerAxisSize = _Log2_PerAxisSize;
	static constexpr uint32_t TileSizePerAxis = 1 << Log2_PerAxisSize;
	static constexpr uint32_t TileArea = TileSizePerAxis * TileSizePerAxis;
	static constexpr uint32_t BitwiseAndMask = TileSizePerAxis - 1;

	Swizzler() = default;
	Swizzler(uint32_t w, uint32_t h)
	{
		if (w == 0 || h == 0) throw std::runtime_error("Incorrect parameters given to swizzler");
		this->params.w = w;
		this->params.h = h;
		this->params.tileCountX = std::ceil(double(w) / TileSizePerAxis);
		this->params.tileCountY = std::ceil(double(h) / TileSizePerAxis);
		this->params.paddedW = this->params.tileCountX * TileSizePerAxis;
		this->params.paddedH = this->params.tileCountY * TileSizePerAxis;
	}
	const SwizzlerParams& getParams() const
	{
		return params;
	}

	//Gets indices in memory for integer positions X and Y. If locationsOutput is not nullptr, tile index and inside tile index data will be stored there
	template<typename T>
	__forceinline T getIndicesForXY(T x, T y, SwizzlerLocations<T>* locationsOutput = nullptr) const
		requires (std::is_integral_v<T> || std::is_same_v<T, int32x16>)
	{
		T tileIndexX = x >> Log2_PerAxisSize;
		T tileIndexY = y >> Log2_PerAxisSize;
		T insideTileX = x & BitwiseAndMask;
		T insideTileY = y & BitwiseAndMask;
		T ret = (tileIndexY * params.tileCountX + tileIndexX) * TileArea + ((insideTileY << Log2_PerAxisSize) + insideTileX);
		if (locationsOutput)
		{
			locationsOutput->insideTileX = insideTileX;
			locationsOutput->insideTileY = insideTileY;
			locationsOutput->tileIndexX = tileIndexX;
			locationsOutput->tileIndexY = tileIndexY;
			locationsOutput->tileIndex = tileIndexY * params.tileCountX + tileIndexX;
			locationsOutput->insideTileIndex = insideTileY * TileSizePerAxis + insideTileX;
		}
		return ret;
	}
private:
	SwizzlerParams params;
};