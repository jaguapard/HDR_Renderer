#pragma once
#include "SIMD_Vector.h"
#include "internals/SIMD_Mask.h"
#include "funcs.h"

namespace AVXXY_NAMESPACE
{
	namespace internals
	{

		template<typename V, size_t Dim>
			requires (Dim >= 1)
		struct SIMD_VectorPackStorage
		{
			SIMD_VectorPackStorage() {};
			union {
#ifdef AVXXY_VECTOR_PACK_XYZW_FIELDS
				struct { V x, y, z, w; };
#endif
#ifdef AVXXY_VECTOR_PACK_RGBA_FIELDS
				struct { V r, g, b, a; };
#endif
				V packs[Dim];
			};
		};

		template<typename V>
		struct SIMD_VectorPackStorage<V, 1>
		{
			SIMD_VectorPackStorage() {};
			union {
#ifdef AVXXY_VECTOR_PACK_XYZW_FIELDS
				V x;
#endif
#ifdef AVXXY_VECTOR_PACK_RGBA_FIELDS
				V r;
#endif
				V packs[1];
			};
		};
		template<typename V>
		struct SIMD_VectorPackStorage<V, 2>
		{
			SIMD_VectorPackStorage() {};
			union {
#ifdef AVXXY_VECTOR_PACK_XYZW_FIELDS
				struct { V x, y; };
#endif
#ifdef AVXXY_VECTOR_PACK_RGBA_FIELDS
				struct { V r, g; };
#endif
				V packs[2];
			};
		};
		template<typename V>
		struct SIMD_VectorPackStorage<V, 3>
		{
			SIMD_VectorPackStorage() {};
			union {
#ifdef AVXXY_VECTOR_PACK_XYZW_FIELDS
				struct { V x, y, z; };
#endif
#ifdef AVXXY_VECTOR_PACK_RGBA_FIELDS
				struct { V r, g, b; };
#endif
				V packs[3];
			};
		};
	}
	//Represents N independent Dim-dimensional vectors, where N is the lane count of V.
	//i.e. pack[0] can be X coordinate, pack[1] - Y, etc,
	//while pack[2][6] is Z coordinate of mathematical vector at index 6
	//Distinction should be made between SIMD_Vector (a packed type of N scalar values), and mathematical vector (Dim-dimensional collection of scalars)
	template<meta::IsSimdVector V, size_t Dim>
		requires (Dim >= 1)
	class SIMD_VectorPack : public internals::SIMD_VectorPackStorage<V, Dim>
	{
	public:
		static constexpr size_t LaneCount = V::LaneCount;
		using ScalarT = typename V::ScalarT;
		using mask_array_t = std::array<mask_t<ScalarT, LaneCount>, Dim>;
		using IntermediateFloatT = std::conditional_t<meta::is_f64<ScalarT> || meta::any_i32<ScalarT> || meta::any_i64<ScalarT>, double, float>;

		SIMD_VectorPack() {};

		//Sets all values of all vectors to a single scalar value
		template<typename T>
		SIMD_VectorPack(const T& s)
		{
			for (size_t i = 0; i < Dim; ++i) (*this)[i] = s;
		}
		//Generic constructor. Assigns elements from left to right to vectors [0..Dim-1] respectively. Input count must equal Dim.
		//Assignees may perform conversions of inputs, i.e. this function will also accept scalars for instance
		template<typename... Ts> requires (sizeof...(Ts) == Dim && Dim != 1)
			SIMD_VectorPack(const Ts&... s)
		{
			size_t i = 0;
			auto append = [&](auto x) {
				(*this)[i++] = x;
				};
			(append(s), ...);
		}

#if 0
		//Type used for intermediate calculations if a floating point type is required for them.
		//For floating point scalar types: same as scalar type
		//Otherwise: float for integers smaller than 4 bytes, double for integers >= 4 bytes.
		using ComputeT = std::conditional_t<meta::any_float<ScalarT>, ScalarT,
			std::conditional_t<(sizeof(ScalarT) < 4), float, double>>;
#endif
		//Returns a const reference to i'th SIMD_Vector. Does not perform range checks.
		const V& operator[](size_t i) const { return this->packs[i]; }
		//Returns a non-const reference to i'th SIMD_Vector. Does not perform range checks. Can be used to modify packs.
		V& operator[](size_t i) { return this->packs[i]; }

		template<typename V2> SIMD_VectorPack<V, Dim> operator+(const SIMD_VectorPack<V2, Dim>& other) const { SIMD_VectorPack<V, Dim> ret; for (size_t i = 0; i < Dim; ++i) ret[i] = (*this)[i] + other[i]; return ret; }
		template<typename V2> SIMD_VectorPack<V, Dim> operator-(const SIMD_VectorPack<V2, Dim>& other) const { SIMD_VectorPack<V, Dim> ret; for (size_t i = 0; i < Dim; ++i) ret[i] = (*this)[i] - other[i]; return ret; }
		template<typename V2> SIMD_VectorPack<V, Dim> operator*(const SIMD_VectorPack<V2, Dim>& other) const { SIMD_VectorPack<V, Dim> ret; for (size_t i = 0; i < Dim; ++i) ret[i] = (*this)[i] * other[i]; return ret; }
		template<typename V2> SIMD_VectorPack<V, Dim> operator/(const SIMD_VectorPack<V2, Dim>& other) const { SIMD_VectorPack<V, Dim> ret; for (size_t i = 0; i < Dim; ++i) ret[i] = (*this)[i] / other[i]; return ret; }
		template<typename V2> SIMD_VectorPack<V, Dim> operator&(const SIMD_VectorPack<V2, Dim>& other) const { SIMD_VectorPack<V, Dim> ret; for (size_t i = 0; i < Dim; ++i) ret[i] = (*this)[i] & other[i]; return ret; }
		template<typename V2> SIMD_VectorPack<V, Dim> operator|(const SIMD_VectorPack<V2, Dim>& other) const { SIMD_VectorPack<V, Dim> ret; for (size_t i = 0; i < Dim; ++i) ret[i] = (*this)[i] | other[i]; return ret; }
		template<typename V2> SIMD_VectorPack<V, Dim> operator^(const SIMD_VectorPack<V2, Dim>& other) const { SIMD_VectorPack<V, Dim> ret; for (size_t i = 0; i < Dim; ++i) ret[i] = (*this)[i] ^ other[i]; return ret; }
		template<typename V2> mask_array_t operator==(const SIMD_VectorPack<V2, Dim>& other) const { mask_array_t ret; for (size_t i = 0; i < Dim; ++i) ret[i] = (*this)[i] == other[i]; return ret; }
		template<typename V2> mask_array_t operator!=(const SIMD_VectorPack<V2, Dim>& other) const { mask_array_t ret; for (size_t i = 0; i < Dim; ++i) ret[i] = (*this)[i] != other[i]; return ret; }
		template<typename V2> mask_array_t operator< (const SIMD_VectorPack<V2, Dim>& other) const { mask_array_t ret; for (size_t i = 0; i < Dim; ++i) ret[i] = (*this)[i] <  other[i]; return ret; }
		template<typename V2> mask_array_t operator<=(const SIMD_VectorPack<V2, Dim>& other) const { mask_array_t ret; for (size_t i = 0; i < Dim; ++i) ret[i] = (*this)[i] <= other[i]; return ret; }
		template<typename V2> mask_array_t operator> (const SIMD_VectorPack<V2, Dim>& other) const { mask_array_t ret; for (size_t i = 0; i < Dim; ++i) ret[i] = (*this)[i] >  other[i]; return ret; }
		template<typename V2> mask_array_t operator>=(const SIMD_VectorPack<V2, Dim>& other) const { mask_array_t ret; for (size_t i = 0; i < Dim; ++i) ret[i] = (*this)[i] >= other[i]; return ret; }

		template<typename T> SIMD_VectorPack<V, Dim> operator+(const T& other) const { SIMD_VectorPack<V, Dim> ret; for (size_t i = 0; i < Dim; ++i) ret[i] = (*this)[i] + other; return ret; }
		template<typename T> SIMD_VectorPack<V, Dim> operator-(const T& other) const { SIMD_VectorPack<V, Dim> ret; for (size_t i = 0; i < Dim; ++i) ret[i] = (*this)[i] - other; return ret; }
		template<typename T> SIMD_VectorPack<V, Dim> operator*(const T& other) const { SIMD_VectorPack<V, Dim> ret; for (size_t i = 0; i < Dim; ++i) ret[i] = (*this)[i] * other; return ret; }
		template<typename T> SIMD_VectorPack<V, Dim> operator/(const T& other) const { SIMD_VectorPack<V, Dim> ret; for (size_t i = 0; i < Dim; ++i) ret[i] = (*this)[i] / other; return ret; }
		template<typename T> SIMD_VectorPack<V, Dim> operator&(const T& other) const { SIMD_VectorPack<V, Dim> ret; for (size_t i = 0; i < Dim; ++i) ret[i] = (*this)[i] & other; return ret; }
		template<typename T> SIMD_VectorPack<V, Dim> operator|(const T& other) const { SIMD_VectorPack<V, Dim> ret; for (size_t i = 0; i < Dim; ++i) ret[i] = (*this)[i] | other; return ret; }
		template<typename T> SIMD_VectorPack<V, Dim> operator^(const T& other) const { SIMD_VectorPack<V, Dim> ret; for (size_t i = 0; i < Dim; ++i) ret[i] = (*this)[i] ^ other; return ret; }
		template<typename T> mask_array_t operator==(const T& other) const { mask_array_t ret; for (size_t i = 0; i < Dim; ++i) ret[i] = (*this)[i] == other; return ret; }
		template<typename T> mask_array_t operator!=(const T& other) const { mask_array_t ret; for (size_t i = 0; i < Dim; ++i) ret[i] = (*this)[i] != other; return ret; }
		template<typename T> mask_array_t operator< (const T& other) const { mask_array_t ret; for (size_t i = 0; i < Dim; ++i) ret[i] = (*this)[i] <  other; return ret; }
		template<typename T> mask_array_t operator<=(const T& other) const { mask_array_t ret; for (size_t i = 0; i < Dim; ++i) ret[i] = (*this)[i] <= other; return ret; }
		template<typename T> mask_array_t operator>=(const T& other) const { mask_array_t ret; for (size_t i = 0; i < Dim; ++i) ret[i] = (*this)[i] >= other; return ret; }
		template<typename T> mask_array_t operator> (const T& other) const { mask_array_t ret; for (size_t i = 0; i < Dim; ++i) ret[i] = (*this)[i] >  other; return ret; }

		template<typename T> SIMD_VectorPack<V, Dim>& operator&=(const T& other) { *this = *this & other; return *this; }
		template<typename T> SIMD_VectorPack<V, Dim>& operator|=(const T& other) { *this = *this | other; return *this; }
		template<typename T> SIMD_VectorPack<V, Dim>& operator^=(const T& other) { *this = *this ^ other; return *this; }
		template<typename T> SIMD_VectorPack<V, Dim>& operator+=(const T& other) { *this = *this + other; return *this; }
		template<typename T> SIMD_VectorPack<V, Dim>& operator-=(const T& other) { *this = *this - other; return *this; }
		template<typename T> SIMD_VectorPack<V, Dim>& operator*=(const T& other) { *this = *this * other; return *this; }
		template<typename T> SIMD_VectorPack<V, Dim>& operator/=(const T& other) { *this = *this / other; return *this; }

		SIMD_VectorPack<V, Dim> operator~() const { SIMD_VectorPack<V, Dim> ret; for (size_t i = 0; i < Dim; ++i) ret[i] = ~((*this)[i]); return ret; }
		SIMD_VectorPack<V, Dim> operator-() const { SIMD_VectorPack<V, Dim> ret; for (size_t i = 0; i < Dim; ++i) ret[i] = -((*this)[i]); return ret; }


		//Computes dot product for each mathematical vector in 2 vector packs. SIMD_Vector at index D and above are ignored and do not affect the output
		template<size_t D = Dim>
			requires (D >= 1 && D <= Dim)
		SIMD_Vector<ScalarT, V::LaneCount> dot(const SIMD_VectorPack<V, Dim>& other) const
		{
			SIMD_Vector<ScalarT, V::LaneCount> ret = (*this)[0] * other[0];
			for (size_t i = 1; i < D; ++i) ret += (*this)[i] * other[i];
			return ret;
		}

		//Computes squared length of each mathematical vector in the pack. SIMD_Vector at index D and above are ignored and do not affect the output
		template<size_t D = Dim>
			requires (D >= 1 && D <= Dim)
		SIMD_Vector<V::ScalarT, V::LaneCount> lenSq() const
		{
			return this->dot<D>(*this);
		}

		//Computes length of each mathematical vector in the pack. SIMD_Vector at index D and above are ignored and do not affect the output
		template<size_t D = Dim, meta::any_float RetScalarT = IntermediateFloatT>
			requires (D >= 1 && D <= Dim)
		SIMD_Vector<RetScalarT, V::LaneCount> len() const
		{
			return vsqrt<RetScalarT>(this->lenSq<D>());
		}

		//Returns a 2D cross product of two vector packs. SIMD_Vector at index 2 and above are ignored and do not affect the output
		SIMD_Vector<ScalarT, LaneCount> cross2d(const SIMD_VectorPack<V, Dim>& other) const
			requires (Dim >= 2)
		{
			return (*this)[0] * other[1] - (*this)[1] * other[0];
		}

		//Returns a 3D cross product of two vector packs. SIMD_Vector at index 3 and above are ignored and do not affect the output
		SIMD_VectorPack<V, 3> cross3d(const SIMD_VectorPack<V, Dim>& other)
			requires (Dim >= 3)
		{
			return {
				(*this)[1] * other[2] - (*this)[2] * other[1],
				(*this)[2] * other[0] - (*this)[0] * other[2],
				(*this)[0] * other[1] - (*this)[1] * other[0]
			};
			//return { y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x, 0.f };
		}
	private:

	};
}