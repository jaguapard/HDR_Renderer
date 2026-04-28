#pragma once
#include <array>
#include <optional>
#include "Vec.h"
#include <memory>

template<typename KeyType, typename ValueType, size_t CAPACITY>
class CacheController
{
public:
	CacheController()
	{
		this->buckets = std::make_unique<std::pair<KeyType, ValueType>[]>(CACHE_CAPACITY);
	}
	static inline constexpr size_t CACHE_CAPACITY = CAPACITY;
	void insert(const KeyType& k, const ValueType& v)
	{
		uint64_t keyHash = this->hash(k);
		this->buckets[keyHash % CAPACITY] = std::make_pair(k, v);
	}
	std::optional<ValueType> tryFetch(const KeyType& k) const
	{
		uint64_t keyHash = this->hash(k);
		const auto& bucket = this->buckets[keyHash % CAPACITY];
		if (bucket.first == k) return bucket.second;
		return std::nullopt;
	}

	uint64_t hash(const KeyType& k) const 
	{
		const uint8_t* data = reinterpret_cast<const uint8_t*>(&k);
		uint64_t x = 0xcbf29ce484222325ULL;

		for (size_t i = 0; i < sizeof(KeyType); ++i) 
		{
			x ^= data[i];
			x *= 0x100000001b3ULL;
		}
		return x;
	}
	/*
	uint64_t hash(const KeyType& k) const
	{
		constexpr size_t sz = sizeof(k);
		size_t p = size_t(&k);
		uint64_t xorshift_state = 20260428155641;
		for (size_t i = 0; i < sz; i += 8)
		{
			size_t rem = sz - i;
			uint64_t zxV;
			if (rem >= 8)
			{
				zxV = *reinterpret_cast<const uint64_t*>(p);
				p += 8;
			}
			else if (rem >= 4)
			{
				zxV = *reinterpret_cast<const uint32_t*>(p);
				p += 4;
			}
			else if (rem >= 2)
			{
				zxV = *reinterpret_cast<const uint16_t*>(p);
				p += 2;
			}
			else
			{
				zxV = *reinterpret_cast<const uint8_t*>(p);
				++p;
			}
			
			xorshift_state ^= zxV << 13;
			xorshift_state ^= xorshift_state >> 7;
			xorshift_state ^= xorshift_state << 17;
		}

		return xorshift_state;
	}*/
private:
	std::unique_ptr<std::pair<KeyType, ValueType>[]> buckets;
};