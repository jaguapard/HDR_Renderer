#pragma once
#include <stdint.h>
#include <memory>
#include <stdexcept>
#include "../../Threadpool.h"

namespace Rasterizing
{
	template<typename T>
	struct Buffer
	{
	private:
		std::unique_ptr<T[]> ownedData;
		T* unownedData = nullptr;
	public:
		uint32_t w, h;
		T clearValue = T();
		Buffer() = default;
		Buffer(T* p, uint32_t w, uint32_t h, T clearValue)
		{
			//this->ownedData.reset();
			this->unownedData = p;
			this->w = w;
			this->h = h;
			this->clearValue = clearValue;
		}
		Buffer(uint32_t w, uint32_t h, T clearValue)
		{
			this->resize(w, h);
			this->clearValue = clearValue;
		}

		void resize(uint32_t w, uint32_t h)
		{
			if (this->unownedData) throw std::runtime_error("Can't resize an unowned buffer!");
			if (!this->ownedData || this->w != w || this->h != h)
			{
				this->ownedData = std::make_unique<T[]>(w * h);
				this->w = w;
				this->h = h;
			}
		}
		//Clears whole buffer with it's clear value
		void clear()
		{
			this->clear(0, w, 0, h);
		}
		//Clears a rectangular section of the buffer with it's clear value
		void clear(size_t xStart, size_t xEnd, size_t yStart, size_t yEnd)
		{
			T* p = this->get();
			assert(p);
			assert(!this->ownedData || !this->unownedData); //exactly one of this ptrs should be null at all times
			size_t wc = w, hc = h;
			for (size_t y = yStart; y < yEnd; ++y)
			{
				for (size_t x = xStart; x < xEnd; ++x)
				{
					p[y * wc + x] = this->clearValue;
				}
			}
		}

		void clearThreadZone(int threadIndex)
		{
			auto [lo, hi] = Threadpool::instance->getLimitsForThread(threadIndex, 0, this->h);
			this->clear(0, w, lo, hi);
		}

		//Returns pointer to buffer data
		T* get() const
		{
			return this->unownedData ? this->unownedData : this->ownedData.get();
		}
	};
}