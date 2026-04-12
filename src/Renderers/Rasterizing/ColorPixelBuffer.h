#pragma once

#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <cassert>
#include "../../Vec.h"
#include <SDL3/SDL.h>

namespace Rasterizing
{
	enum class MappingType
	{
		NONE,
		CLAMP,
		WRAP,
	};

	enum class InterpolationType
	{
		NONE,
		BILINEAR,
	};

	enum class SwizzlingType
	{
		NONE,
	};

	enum class FormatType
	{
		PLAIN,
		DIFFUSE_MAP_WITH_OPACITY_MAP,
	};

	enum class ElementType
	{
		FLOAT32,
		R10G11B10A1_GAMMA2,
	};

	template<ElementType E>
	struct PixelElementStorage;

	template<>
	struct PixelElementStorage<ElementType::FLOAT32>
	{
		using type = float;
	};

	template<>
	struct PixelElementStorage<ElementType::R10G11B10A1_GAMMA2>
	{
		using type = uint32_t;
	};

	template<MappingType MAPPING, SwizzlingType SWIZZLING, FormatType FORMAT, ElementType ELEMENT>
	class PixelBuffer
	{
	public:
		using StorageType = typename PixelElementStorage<ELEMENT>::type;
		struct Sizes
		{
			int w = 0, h = 0;
			float fw = 0, fh = 0, rcpW = 0, rcpH = 0;
			float float_maxSafeX = 0, float_maxSafeY = 0, rcp_maxSafeX = 0, rcp_maxSafeY = 0;
		};

		PixelBuffer() = default;
		PixelBuffer(int w, int h);
		PixelBuffer(PixelBuffer&& dying) = default;
		PixelBuffer& operator=(PixelBuffer&& dying) = default;

		int width() const { return this->w; }
		int height() const { return this->h; }
		StorageType* data() { return this->elements.get(); }
		const StorageType* data() const { return this->elements.get(); }

	protected:
		void init(int newW, int newH);
		static float mapCoordinate(float uv, float axisSize)
		{
			if constexpr (MAPPING == MappingType::NONE)
			{
				return uv * axisSize;
			}
			else if constexpr (MAPPING == MappingType::CLAMP)
			{
				return std::clamp(uv, 0.0f, 1.0f) * axisSize;
			}
			else
			{
				float wrapped = uv - std::floor(uv);
				return wrapped * axisSize;
			}
		}
		static float32x16 mapCoordinate(float32x16 uv, float axisSize)
		{
			if constexpr (MAPPING == MappingType::NONE)
			{
				return uv * axisSize;
			}
			else if constexpr (MAPPING == MappingType::CLAMP)
			{
				return uv.clamp(0.f, 1.f) * axisSize;
			}
			else
			{
				float32x16 wrapped = uv - _mm512_floor_ps(uv);
				return wrapped * axisSize;
			}
		}
		static void mapUVToPixelCoordinates(float u, float v, const Sizes& sizes, float& outX, float& outY)
		{
			outX = mapCoordinate(u, sizes.float_maxSafeX);
			outY = mapCoordinate(v, sizes.float_maxSafeY);
		}
		static void mapUVToPixelCoordinates(float32x16 u, float32x16 v, const Sizes& sizes, float32x16& outX, float32x16& outY)
		{
			outX = mapCoordinate(u, sizes.float_maxSafeX);
			outY = mapCoordinate(v, sizes.float_maxSafeY);
		}
		static int swizzleIndex(int x, int y, int width, int height)
		{
			if constexpr (SWIZZLING == SwizzlingType::NONE)
			{
				assert(x >= 0 && x < width);
				assert(y >= 0 && y < height);
				return y * width + x;
			}
			return y * width + x;
		}
		static int32x16 swizzleIndex(int32x16 x, int32x16 y, const Sizes& sizes)
		{
			if constexpr (SWIZZLING == SwizzlingType::NONE)
			{
				return y * sizes.w + x;
			}
			return y * sizes.w + x;
		}
		int w = 0, h = 0;
		Sizes sizes;
		std::unique_ptr<StorageType[]> elements;
	};

	template<MappingType MAPPING, SwizzlingType SWIZZLING, FormatType FORMAT, ElementType ELEMENT>
	inline PixelBuffer<MAPPING, SWIZZLING, FORMAT, ELEMENT>::PixelBuffer(int w, int h)
	{
		this->init(w, h);
	}

	template<MappingType MAPPING, SwizzlingType SWIZZLING, FormatType FORMAT, ElementType ELEMENT>
	inline void PixelBuffer<MAPPING, SWIZZLING, FORMAT, ELEMENT>::init(int newW, int newH)
	{
		this->w = newW;
		this->h = newH;
		this->sizes.w = newW;
		this->sizes.h = newH;
		this->sizes.fw = float(newW);
		this->sizes.fh = float(newH);
		this->sizes.rcpW = (newW != 0) ? (float(1) / newW) : 0.f;
		this->sizes.rcpH = (newH != 0) ? (float(1) / newH) : 0.f;
		this->sizes.float_maxSafeX = float(std::max(0, newW - 1));
		this->sizes.float_maxSafeY = float(std::max(0, newH - 1));
		this->sizes.rcp_maxSafeX = (this->sizes.float_maxSafeX != 0.f) ? (1.f / this->sizes.float_maxSafeX) : 0.f;
		this->sizes.rcp_maxSafeY = (this->sizes.float_maxSafeY != 0.f) ? (1.f / this->sizes.float_maxSafeY) : 0.f;
		this->elements = std::make_unique<StorageType[]>(newW * newH);
	}

	template<MappingType MAPPING, SwizzlingType SWIZZLING, FormatType FORMAT, ElementType ELEMENT>
	struct PixelBufferGatherAccessor;

	using ColorPixelBuffer = PixelBuffer<MappingType::WRAP, SwizzlingType::NONE, FormatType::DIFFUSE_MAP_WITH_OPACITY_MAP, ElementType::R10G11B10A1_GAMMA2>;
	using ColorPixelBufferGatherAccessor = PixelBufferGatherAccessor<MappingType::WRAP, SwizzlingType::NONE, FormatType::DIFFUSE_MAP_WITH_OPACITY_MAP, ElementType::R10G11B10A1_GAMMA2>;
	class TextureManager;

	template<>
	struct PixelBufferGatherAccessor<MappingType::WRAP, SwizzlingType::NONE, FormatType::DIFFUSE_MAP_WITH_OPACITY_MAP, ElementType::R10G11B10A1_GAMMA2>
	{
	public:
		void gatherLinearRGB(Vec4_f32x16& output) const;
		float32x16 gatherA() const;

		int32x16 gatherInd;
		Mask16 gatherMask;
		const ColorPixelBuffer* buf;
	};

	template<>
	class PixelBuffer<MappingType::WRAP, SwizzlingType::NONE, FormatType::DIFFUSE_MAP_WITH_OPACITY_MAP, ElementType::R10G11B10A1_GAMMA2>
	{
	public:
		struct Sizes
		{
			int w, h;
			float fw, fh, rcpW, rcpH;
			float float_maxSafeX, float_maxSafeY, rcp_maxSafeX, rcp_maxSafeY;
		};

		PixelBuffer(PixelBuffer&& dying);
		PixelBuffer(int w, int h);
		PixelBuffer(const SDL_Surface* s);
		Vec4_f32x16 gatherLinearIntensities(float32x16 x, float32x16 y, Mask16 mask = 0xFFFF) const;
		ColorPixelBufferGatherAccessor getGatherAccessor(float32x16 u, float32x16 v, Mask16 mask = 0xFFFF) const;
		Vec4f getLinearIntensity(float u, float v) const;
		bool areAllPixelsOpaque() const;

	private:
		void init(int w, int h);
		static int32x16 getLinearIndices(float32x16 u, float32x16 v, const Sizes& sizes, Mask16 mask);

		std::unique_ptr<uint32_t[]> packedColors, opacityMap;
		bool isFullyOpaque = true;
		Sizes sizes;

		friend class Rasterizing::TextureManager;
		friend struct Rasterizing::PixelBufferGatherAccessor<MappingType::WRAP, SwizzlingType::NONE, FormatType::DIFFUSE_MAP_WITH_OPACITY_MAP, ElementType::R10G11B10A1_GAMMA2>;
	};
}
