#pragma once
#include "SSE_Vec4.h"
#include <string>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <cstring>
#include "VectorPack.h"

class alignas(64) Matrix4
{
public:
	typedef bob::_SSE_Vec4_float Vec4;
	union {
		bob::_SSE_Vec4_float val[4];
		float elements[4][4];
		AVXXY_NAMESPACE::f32x16 zmm;
	};

	Matrix4() {};
	Matrix4(const std::initializer_list<bob::_SSE_Vec4_float> lst);
	Matrix4(const AVXXY_NAMESPACE::f32x16& m);

	__forceinline Matrix4 operator*(const float other) const;
	__forceinline Matrix4 operator-(const Matrix4& other) const;
	__forceinline Matrix4 operator+(const Matrix4& other) const;
	Matrix4 operator*(const Matrix4& other) const; //result = this * other

	Vec4 operator*(const Vec4 v) const;
	//VectorPack16 operator*(const VectorPack16& v) const;
	__forceinline bob::Vec4_f32x16 operator*(const bob::Vec4_f32x16& v) const
	{
		bob::Vec4_f32x16 ret = 0;
		AVXXY_NAMESPACE::f32x16 mat = AVXXY_NAMESPACE::load<AVXXY_NAMESPACE::f32x16>(this);
		for (int i = 0; i < 4; ++i)
		{
			for (int j = 0; j < 4; ++j)
			{
				ret[i] += v[j] * AVXXY_NAMESPACE::f32x16(mat[i * 4 + j]);
			}
		}
		return ret;
	}

	Matrix4 transposed() const;
	Vec4 multiplyByTransposed(const Vec4 v) const; //result = A^T * x (multiply transposed matrix by column vector v)

	const bob::_SSE_Vec4_float& operator[](int i) const;
	bob::_SSE_Vec4_float& operator[](int i);

	std::string toString(int precision = 5) const;

	float det() const;
	Matrix4 inverse() const;

	static Matrix4 rotationX(float theta);
	static Matrix4 rotationY(float theta);
	static Matrix4 rotationZ(float theta);
	static Matrix4 rotationXYZ(const Vec4& angle);
	static Matrix4 identity(float value = 1.0, int dim = 4);
	static Matrix4 zeros();
private:
	float det3(int excludeRow, int excludeCol) const;
};

//class representing pack of 16 4x4 matrices.
//Value at elements[i][j][k] is i'th row and j'th column of k'th matrix in the pack
class alignas(64) MatrixPack16_4x4
{
public:
	AVXXY_NAMESPACE::f32x16 elements[4][4];
	static MatrixPack16_4x4 rotationX(AVXXY_NAMESPACE::f32x16 theta);
	static MatrixPack16_4x4 rotationY(AVXXY_NAMESPACE::f32x16 theta);
	static MatrixPack16_4x4 rotationZ(AVXXY_NAMESPACE::f32x16 theta);
	static MatrixPack16_4x4 rotationXYZ(const bob::Vec4_f32x16& angle);

	//These function return very fast low quality approximations for the rotation matrices. Use only if inaccuracies don't matter
	static MatrixPack16_4x4 fast_rotationX(AVXXY_NAMESPACE::f32x16 theta);
	static MatrixPack16_4x4 fast_rotationY(AVXXY_NAMESPACE::f32x16 theta);
	static MatrixPack16_4x4 fast_rotationZ(AVXXY_NAMESPACE::f32x16 theta);
	static MatrixPack16_4x4 fast_rotationXYZ(const bob::Vec4_f32x16& angle);

	static MatrixPack16_4x4 identity();

	__forceinline bob::Vec4_f32x16 operator*(const bob::Vec4_f32x16& v) const
	{
		bob::Vec4_f32x16 ret = 0;
		for (int i = 0; i < 4; ++i)
		{
			for (int j = 0; j < 4; ++j)
			{
				ret[i] += v[j] * elements[i][j];
			}
		}
		return ret;
	}

	__forceinline MatrixPack16_4x4 operator*(const MatrixPack16_4x4& other) const
	{
		MatrixPack16_4x4 ret;
		memset(&ret, 0, sizeof(ret));
		for (int i = 0; i < 4; ++i)
		{
			for (int j = 0; j < 4; ++j)
			{
				for (int k = 0; k < 4; ++k)
				{
					ret.elements[i][j] += this->elements[i][k] * other.elements[k][j];
				}
			}
		}
		return ret;
	}
};