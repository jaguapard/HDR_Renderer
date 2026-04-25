#pragma once
#include <immintrin.h>
#include <array>

#include "SSE_Vec4.h"
#include "float32x8.h"
#include "float32x16.h"

//class representing 8 packed 4-dimensional vectors.
//Only AVX2 is supported with this one!
//The pack fits into 4 ymm registers.
//Each ymm register contains values from 8 vectors,
//i.e. ymm0 may have x0,x1,...,x7, ymm1 - y0,y1,...,y7, etc
namespace bob
{
	template <typename PackType>
	struct
#ifdef __AVX512F__
		alignas(64)
#else
		alignas(32)
#endif
		VectorPack
	{
		union {
			struct { PackType x, y, z, w; };
			struct { PackType r, g, b, a; };
			PackType packs[4];
		};

		VectorPack() {};
		VectorPack(const float x); //broadcast x to all elements of all vectors
		VectorPack(const bob::_SSE_Vec4_float& v); //broadcast a single vector to all vectors in the pack
		VectorPack(const PackType& pack); //broadcast a single pack to all values
		VectorPack(const PackType& x, const PackType& y, const PackType& z, const PackType& w);
		//VectorPack<PackType>(const __m256& pack);
		//VectorPack(const std::initializer_list<bob::_SSE_Vec4_float>& list);

		VectorPack(const VectorPack<PackType>& other);
		//VectorPack<PackType>& operator=(const VectorPack<PackType>& other);

		template <typename Container>
		static VectorPack<PackType> fromHorizontalVectors(const Container& cont);

		__forceinline VectorPack<PackType> operator+(const float other) const; //Add a single value to all elements of the vector pack
		__forceinline VectorPack<PackType> operator-(const float other) const; //Subtract a single value from all elements of the vector pack
		__forceinline VectorPack<PackType> operator*(const float other) const; //Multiply all elements of the vector pack by a single value 
		__forceinline VectorPack<PackType> operator/(const float other) const; //Divide all elements of the vector pack by a single value

		__forceinline VectorPack<PackType> operator+=(const float other); //Add a single value to all elements of the vector in-place
		__forceinline VectorPack<PackType> operator-=(const float other); //Subtract a single value from all elements of the vector in-place
		__forceinline VectorPack<PackType> operator*=(const float other); //Multiply all elements of the vector by a single value in-place
		__forceinline VectorPack<PackType> operator/=(const float other); //Divide all elements of the vector by a single value in-place

		__forceinline VectorPack<PackType> operator+(const VectorPack<PackType>& other) const;
		__forceinline VectorPack<PackType> operator-(const VectorPack<PackType>& other) const;
		__forceinline VectorPack<PackType> operator*(const VectorPack<PackType>& other) const;
		__forceinline VectorPack<PackType> operator/(const VectorPack<PackType>& other) const;
		__forceinline VectorPack<PackType>& operator+=(const VectorPack<PackType>& other);
		__forceinline VectorPack<PackType>& operator-=(const VectorPack<PackType>& other);
		__forceinline VectorPack<PackType>& operator*=(const VectorPack<PackType>& other);
		__forceinline VectorPack<PackType>& operator/=(const VectorPack<PackType>& other);

		__forceinline std::array<Mask16, 4> operator>(const VectorPack<PackType>& other) const;
		__forceinline std::array<Mask16, 4> operator>=(const VectorPack<PackType>& other) const;
		__forceinline std::array<Mask16, 4> operator<(const VectorPack<PackType>& other) const;
		__forceinline std::array<Mask16, 4> operator<=(const VectorPack<PackType>& other) const;
		__forceinline std::array<Mask16, 4> operator==(const VectorPack<PackType>& other) const;
		__forceinline std::array<Mask16, 4> operator!=(const VectorPack<PackType>& other) const;


		__forceinline VectorPack<PackType> operator+(const PackType& other) const;
		__forceinline VectorPack<PackType> operator-(const PackType& other) const;
		__forceinline VectorPack<PackType> operator*(const PackType& other) const;
		__forceinline VectorPack<PackType> operator/(const PackType& other) const;
		__forceinline VectorPack<PackType>& operator+=(const PackType& other);
		__forceinline VectorPack<PackType>& operator-=(const PackType& other);
		__forceinline VectorPack<PackType>& operator*=(const PackType& other);
		__forceinline VectorPack<PackType>& operator/=(const PackType& other);

		__forceinline std::array<__mmask16, 4> operator>(const PackType& other) const;
		__forceinline std::array<__mmask16, 4> operator>=(const PackType& other) const;
		__forceinline std::array<__mmask16, 4> operator<(const PackType& other) const;
		__forceinline std::array<__mmask16, 4> operator<=(const PackType& other) const;
		__forceinline std::array<__mmask16, 4> operator==(const PackType& other) const;
		__forceinline std::array<__mmask16, 4> operator!=(const PackType& other) const;


		__forceinline VectorPack<PackType> operator&(const VectorPack<PackType>& other) const;
		__forceinline VectorPack<PackType> operator|(const VectorPack<PackType>& other) const;
		__forceinline VectorPack<PackType> operator^(const VectorPack<PackType>& other) const;
		__forceinline VectorPack<PackType>& operator&=(const VectorPack<PackType>& other);
		__forceinline VectorPack<PackType>& operator|=(const VectorPack<PackType>& other);
		__forceinline VectorPack<PackType>& operator^=(const VectorPack<PackType>& other);

		__forceinline PackType* begin();
		__forceinline PackType* end();
		__forceinline const PackType* begin() const;
		__forceinline const PackType* end() const;

		PackType& operator[](size_t i);
		__forceinline const PackType& operator[](size_t i) const;

		__forceinline float product() const;
		__forceinline PackType len3d() const;
		__forceinline PackType len2d() const;
		__forceinline PackType lenSq() const;
		__forceinline PackType lenSq3d() const;
		__forceinline PackType lenSq2d() const;

		__forceinline VectorPack<PackType> operator-() const;
		__forceinline VectorPack<PackType> operator~() const;
		__forceinline VectorPack<PackType> unit() const;

		__forceinline PackType dot(const VectorPack<PackType>& other) const;
		__forceinline PackType dot3d(const VectorPack<PackType>& other) const; //ignore w coordinate
		__forceinline PackType dot2d(const VectorPack<PackType>& other) const; //ignore w coordinate
		__forceinline PackType cross2d(const VectorPack<PackType>& other) const;
		__forceinline VectorPack<PackType> cross3d(const VectorPack<PackType>& other) const;
		__forceinline VectorPack<PackType> lerp(const VectorPack<PackType>& dst, const PackType& amount) const;

		__forceinline bob::_SSE_Vec4_float extractHorizontalVector(size_t index) const;
	};

	template <typename PackType>
	__forceinline VectorPack<PackType>::VectorPack(const float f) : x(f), y(f), z(f), w(f) {} //broadcast a single float to all elements

	template <typename PackType>
	__forceinline VectorPack<PackType>::VectorPack(const bob::_SSE_Vec4_float& v) : x(v.x), y(v.y), z(v.z), w(v.w) {};

	template <typename PackType>
	__forceinline VectorPack<PackType>::VectorPack(const PackType& pack) : x(pack), y(pack), z(pack), w(pack) {};

	template<typename PackType>
	__forceinline VectorPack<PackType>::VectorPack(const PackType& x, const PackType& y, const PackType& z, const PackType& w) :x(x), y(y), z(z), w(w) {}

	/*
	template <typename PackType>
 __forceinline VectorPack<PackType>::VectorPack(const std::initializer_list<bob::_SSE_Vec4_float>& list)
	{
		size_t sz = std::end(list) - std::begin(list);
		for (size_t i = 0; i < sz; ++i)
		{
			x.f[i] = (std::begin(list) + i)->x;
			y.f[i] = (std::begin(list) + i)->y;
			z.f[i] = (std::begin(list) + i)->z;
			w.f[i] = (std::begin(list) + i)->w;
		}
	}*/

	template<typename PackType>
	__forceinline VectorPack<PackType>::VectorPack(const VectorPack<PackType>& other) : x(other.x), y(other.y), z(other.z), w(other.w) {}

	template <typename PackType>
	__forceinline VectorPack<PackType> VectorPack<PackType>::operator+(const float other) const
	{
		return { x + other,y + other,z + other,w + other };
	}

	template <typename PackType>
	__forceinline VectorPack<PackType> VectorPack<PackType>::operator-(const float other) const
	{
		return { x - other,y - other,z - other,w - other };
	}

	template <typename PackType>
	__forceinline VectorPack<PackType> VectorPack<PackType>::operator*(const float other) const
	{
		return { x * other,y * other,z * other,w * other };
	}

	template <typename PackType>
	__forceinline VectorPack<PackType> VectorPack<PackType>::operator/(const float other) const
	{
		return { x / other,y / other,z / other,w / other };
	}

	template <typename PackType>
	__forceinline VectorPack<PackType> VectorPack<PackType>::operator+=(const float other)
	{
		return *this = *this + other;
	}

	template <typename PackType>
	__forceinline VectorPack<PackType> VectorPack<PackType>::operator-=(const float other)
	{
		return *this = *this - other;
	}

	template <typename PackType>
	__forceinline VectorPack<PackType> VectorPack<PackType>::operator*=(const float other)
	{
		return *this = *this * other;
	}

	template <typename PackType>
	__forceinline VectorPack<PackType> VectorPack<PackType>::operator/=(const float other)
	{
		return *this = *this / other;
	}

	template <typename PackType>
	__forceinline VectorPack<PackType> VectorPack<PackType>::operator+(const VectorPack<PackType>& other) const
	{
		return { x + other.x, y + other.y, z + other.z, w + other.w };
	}

	template <typename PackType>
	__forceinline VectorPack<PackType> VectorPack<PackType>::operator-(const VectorPack<PackType>& other) const
	{
		return { x - other.x, y - other.y, z - other.z, w - other.w };
	}

	template <typename PackType>
	__forceinline VectorPack<PackType> VectorPack<PackType>::operator*(const VectorPack<PackType>& other) const
	{
		return { x * other.x, y * other.y, z * other.z, w * other.w };
	}

	template <typename PackType>
	__forceinline VectorPack<PackType> VectorPack<PackType>::operator/(const VectorPack<PackType>& other) const
	{
		return { x / other.x, y / other.y, z / other.z, w / other.w };
	}

	template <typename PackType>
	__forceinline std::array<Mask16, 4> VectorPack<PackType>::operator>(const VectorPack<PackType>& other) const
	{
		std::array<Mask16, 4> ret;
		for (size_t i = 0; i < std::size(packs); ++i) ret[i] = (*this)[i] > other[i];
		return ret;
	}

	template <typename PackType>
	__forceinline std::array<Mask16, 4> VectorPack<PackType>::operator>=(const VectorPack<PackType>& other) const
	{
		std::array<Mask16, 4> ret;
		for (size_t i = 0; i < std::size(packs); ++i) ret[i] = (*this)[i] >= other[i];
		return ret;
	}

	template <typename PackType>
	__forceinline std::array<Mask16, 4> VectorPack<PackType>::operator<(const VectorPack<PackType>& other) const
	{
		std::array<Mask16, 4> ret;
		for (size_t i = 0; i < std::size(packs); ++i) ret[i] = (*this)[i] < other[i];
		return ret;
	}

	template <typename PackType>
	__forceinline std::array<Mask16, 4> VectorPack<PackType>::operator<=(const VectorPack<PackType>& other) const
	{
		std::array<Mask16, 4> ret;
		for (size_t i = 0; i < std::size(packs); ++i) ret[i] = (*this)[i] <= other[i];
		return ret;
	}

	template <typename PackType>
	__forceinline std::array<Mask16, 4> VectorPack<PackType>::operator==(const VectorPack<PackType>& other) const
	{
		std::array<Mask16, 4> ret;
		for (size_t i = 0; i < std::size(packs); ++i) ret[i] = (*this)[i] == other[i];
		return ret;
	}

	template <typename PackType>
	__forceinline std::array<Mask16, 4> VectorPack<PackType>::operator!=(const VectorPack<PackType>& other) const
	{
		std::array<Mask16, 4> ret;
		for (size_t i = 0; i < std::size(packs); ++i) ret[i] = (*this)[i] != other[i];
		return ret;
	}

	template<typename PackType>
	__forceinline VectorPack<PackType> VectorPack<PackType>::operator+(const PackType& other) const
	{
		return { x + other, y + other, z + other, w + other };
	}

	template<typename PackType>
	__forceinline VectorPack<PackType> VectorPack<PackType>::operator-(const PackType& other) const
	{
		return { x - other, y - other, z - other, w - other };
	}

	template<typename PackType>
	__forceinline VectorPack<PackType> VectorPack<PackType>::operator*(const PackType& other) const
	{
		return { x * other, y * other, z * other, w * other };
	}


	template<typename PackType>
	__forceinline VectorPack<PackType> VectorPack<PackType>::operator/(const PackType& other) const
	{
		return { x / other, y / other, z / other, w / other };
	}

	template <typename PackType>
	__forceinline VectorPack<PackType> VectorPack<PackType>::operator&(const VectorPack<PackType>& other) const
	{
		return { x & other.x, y & other.y, z & other.z, w & other.w };
	}

	template <typename PackType>
	__forceinline VectorPack<PackType> VectorPack<PackType>::operator|(const VectorPack<PackType>& other) const
	{
		return { x | other.x, y | other.y, z | other.z, w | other.w };
	}

	template <typename PackType>
	__forceinline VectorPack<PackType> VectorPack<PackType>::operator^(const VectorPack<PackType>& other) const
	{
		return { x ^ other.x, y ^ other.y, z ^ other.z, w ^ other.w };
	}

	template <typename PackType>
	__forceinline VectorPack<PackType>& VectorPack<PackType>::operator&=(const VectorPack<PackType>& other)
	{
		return *this = *this & other;
	}

	template <typename PackType>
	__forceinline VectorPack<PackType>& VectorPack<PackType>::operator|=(const VectorPack<PackType>& other)
	{
		return *this = *this | other;
	}

	template <typename PackType>
	__forceinline VectorPack<PackType>& VectorPack<PackType>::operator^=(const VectorPack<PackType>& other)
	{
		return *this = *this ^ other;
	}

	template <typename PackType>
	__forceinline PackType* VectorPack<PackType>::begin()
	{
		return const_cast<PackType*>(static_cast<const VectorPack*>(this)->begin());
	}

	template <typename PackType>
	__forceinline PackType* VectorPack<PackType>::end()
	{
		return const_cast<PackType*>(static_cast<const VectorPack*>(this)->end());
	}

	template <typename PackType>
	__forceinline const PackType* VectorPack<PackType>::begin() const
	{
		return &this->packs[0];
	}

	template <typename PackType>
	__forceinline const PackType* VectorPack<PackType>::end() const
	{
		return &this->packs[0] + std::size(packs);
	}

	template <typename PackType>
	__forceinline PackType& VectorPack<PackType>::operator[](size_t i)
	{
		const VectorPack<PackType>& p = *this;
		return const_cast<PackType&>(p[i]);
	}

	template <typename PackType>
	__forceinline const PackType& VectorPack<PackType>::operator[](size_t i) const
	{
		return packs[i];
	}

	template <typename PackType>
	__forceinline VectorPack<PackType>& VectorPack<PackType>::operator+=(const VectorPack<PackType>& other)
	{
		return *this = *this + other;
	}

	template <typename PackType>
	__forceinline VectorPack<PackType>& VectorPack<PackType>::operator-=(const VectorPack<PackType>& other)
	{
		return *this = *this - other;
	}

	template <typename PackType>
	__forceinline VectorPack<PackType>& VectorPack<PackType>::operator*=(const VectorPack<PackType>& other)
	{
		return *this = *this * other;
	}

	template <typename PackType>
	__forceinline VectorPack<PackType>& VectorPack<PackType>::operator/=(const VectorPack<PackType>& other)
	{
		return *this = *this / other;
	}

	template <typename PackType>
	__forceinline VectorPack<PackType>& VectorPack<PackType>::operator+=(const PackType& other)
	{
		return *this = *this + other;
	}

	template <typename PackType>
	__forceinline VectorPack<PackType>& VectorPack<PackType>::operator-=(const PackType& other)
	{
		return *this = *this - other;
	}

	template <typename PackType>
	__forceinline VectorPack<PackType>& VectorPack<PackType>::operator*=(const PackType& other)
	{
		return *this = *this * other;
	}

	template <typename PackType>
	__forceinline VectorPack<PackType>& VectorPack<PackType>::operator/=(const PackType& other)
	{
		return *this = *this / other;
	}

	template <typename PackType>
	__forceinline VectorPack<PackType> VectorPack<PackType>::operator-() const
	{
		return { -x,-y,-z,-w };
	}

	template <typename PackType>
	__forceinline VectorPack<PackType> VectorPack<PackType>::operator~() const
	{
		return { ~x, ~y, ~z, ~w };
	}

	template <typename PackType>
	__forceinline PackType VectorPack<PackType>::dot(const VectorPack<PackType>& other) const
	{
		return x * other.x + y * other.y + z * other.z + w * other.w;
	}

	template <typename PackType>
	__forceinline PackType VectorPack<PackType>::dot3d(const VectorPack<PackType>& other) const
	{
		return x * other.x + y * other.y + z * other.z;
	}

	template <typename PackType>
	__forceinline PackType VectorPack<PackType>::dot2d(const VectorPack<PackType>& other) const
	{
		return x * other.x + y * other.y;
	}

	template <typename PackType>
	__forceinline PackType VectorPack<PackType>::cross2d(const VectorPack<PackType>& other) const
	{
		return x * other.y - y * other.x;
	}

	template <typename PackType>
	__forceinline VectorPack<PackType> VectorPack<PackType>::cross3d(const VectorPack<PackType>& other) const
	{
		return { y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x, 0.f };
	}

	template <typename PackType>
	__forceinline VectorPack<PackType> VectorPack<PackType>::lerp(const VectorPack<PackType>& dst, const PackType& amount) const
	{
		return *this + (dst - *this) * amount;
	}

	template <typename PackType>
	__forceinline bob::_SSE_Vec4_float VectorPack<PackType>::extractHorizontalVector(size_t index) const
	{
		return bob::_SSE_Vec4_float(x.f[index], y.f[index], z.f[index], w.f[index]);
	}

	template <typename PackType>
	template<typename Container>
	__forceinline VectorPack<PackType> VectorPack<PackType>::fromHorizontalVectors(const Container& cont)
	{
		//assert(std::size(cont) <= std::size(packs));
		VectorPack<PackType> ret;
		int i = 0;
		for (auto it = std::begin(cont); it != std::end(cont); ++it, ++i)
		{
			ret.x[i] = (*it)[i][0];
			ret.y[i] = (*it)[i][1];
			ret.z[i] = (*it)[i][2];
			ret.w[i] = (*it)[i][4];
		}
		return ret;
	}

	template<typename PackType>
	__forceinline PackType VectorPack<PackType>::len3d() const
	{
		return this->lenSq3d().sqrt();
	}

	template<typename PackType>
	__forceinline PackType VectorPack<PackType>::len2d() const
	{
		return this->lenSq2d().sqrt();
	}

	template <typename PackType>
	__forceinline PackType VectorPack<PackType>::lenSq() const
	{
		return this->dot(*this);
	}

	template <typename PackType>
	__forceinline PackType VectorPack<PackType>::lenSq3d() const
	{
		return this->dot3d(*this);
	}

	template <typename PackType>
	__forceinline PackType VectorPack<PackType>::lenSq2d() const
	{
		return this->dot2d(*this);
	}

	//typedef VectorPack<float32x8> Vec4_f32x8;
	typedef VectorPack<float32x16> Vec4_f32x16;
	typedef VectorPack<float32x8> Vec4_f32x8;
}