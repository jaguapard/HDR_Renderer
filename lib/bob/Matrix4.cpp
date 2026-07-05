#include "Matrix4.h"
#include "../../src/LUTMan.h"
using namespace bob;
using namespace AVXXY_NAMESPACE;
Matrix4::Matrix4(const std::initializer_list<bob::_SSE_Vec4_float> lst)
{
	assert(lst.size() == 4);
	for (int i = 0; i < 4; ++i) this->val[i] = *(lst.begin() + i);
}

Matrix4::Matrix4(const f32x16& m)
{
	zmm = m;
}

Matrix4 Matrix4::operator*(const float other) const
{
	return zmm * other;
}

Matrix4 Matrix4::operator-(const Matrix4& other) const
{
	return zmm - other.zmm;
}

Matrix4 Matrix4::operator+(const Matrix4& other) const
{
	return zmm + other.zmm;
}

Matrix4 Matrix4::operator*(const Matrix4& other) const
{
	Matrix4 ret = Matrix4::zeros();
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

bob::_SSE_Vec4_float Matrix4::operator*(const bob::_SSE_Vec4_float v) const
{
	/*
#if __AVX512F__
	__m512 cast = _mm512_castps128_ps512(v);
	__m512 broadcasted_v = _mm512_shuffle_f32x4(cast, cast, 0);
	__m512 preSum = _mm512_mul_ps(broadcasted_v, zmm); //sum elements 0-3 to get result x, 4-7 for y, 8-11 for z, 12-15 for w
	__m512 parts = _mm512_permutexvar_ps(_mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15), preSum); //add up each __m128 part to get final answer

	return Vec4(_mm512_extractf32x4_ps(parts, 0)) + Vec4(_mm512_extractf32x4_ps(parts, 1)) + Vec4(_mm512_extractf32x4_ps(parts, 2)) + Vec4(_mm512_extractf32x4_ps(parts, 3));
#elif __AVX2__
	__m256 vv = _mm256_setr_ps(v.x, v.y, v.z, v.w, v.x, v.y, v.z, v.w);
	__m256 preSum_xy = _mm256_mul_ps(vv, ymm0); //sum elements 0-3 to get result x, 4-7 for y
	__m256 preSum_zw = _mm256_mul_ps(vv, ymm1); //sum elements 0-3 to get result z, 4-7 for w

	__m256 preSum_xyzw = _mm256_hadd_ps(preSum_xy, preSum_zw); //to get final: x = 0+1, z = 2+3, y = 4+5, w = 6+7
	//permute values so x=0+4; y=1+5; z=2+6; w=3+7. This allows us to extract the upper half and just add it to the lower to get the result
	__m256 perm = _mm256_permutevar8x32_ps(preSum_xyzw, _mm256_setr_epi32(0, 4, 2, 6, 1, 5, 3, 7));
	return Vec4(_mm256_extractf128_ps(perm, 0)) + Vec4(_mm256_extractf128_ps(perm, 1));
#elif defined(__SSE2__)
	Vec4 r1 = val[0] * v;
	Vec4 r2 = val[1] * v;
	Vec4 r3 = val[2] * v;
	Vec4 r4 = val[3] * v;

	__m128 res1 = _mm_hadd_ps(r1, r2); //x = 0+1, y = 2+3
	__m128 res2 = _mm_hadd_ps(r3, r4); //z = 0+1, w = 2+3

	return _mm_hadd_ps(res1, res2);
#else*/
	Vec4 ret(0, 0, 0, 0);
	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			ret[i] += (*this)[i][j] * v[j];
		}
	}
	return ret;
	//#endif
}

/*
VectorPack16 Matrix4::operator*(const VectorPack16& v) const
{
	VectorPack16 ret = 0.0f;
	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			ret[i] += v[j] * (*this)[i][j];
		}
	}
	return ret;
}
*/

Matrix4 Matrix4::transposed() const
{
	/*
#if __AVX512F__
	return _mm512_permutexvar_ps(_mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15), zmm);
#else*/
	Matrix4 ret;
	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			ret.elements[i][j] = this->elements[j][i];
		}
	}
	return ret;
	//#endif
}

bob::_SSE_Vec4_float Matrix4::multiplyByTransposed(const bob::_SSE_Vec4_float v) const
{
#if __AVX512F__
	__m512 bcst = _mm512_permutexvar_ps(_mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3), _mm512_castps128_ps512(v));
	__m512 mul = _mm512_mul_ps(zmm, bcst);
	__m256 red1 = _mm256_add_ps(_mm512_extractf32x8_ps(mul, 0), _mm512_extractf32x8_ps(mul, 1));
	return _mm_add_ps(_mm256_extractf32x4_ps(red1, 0), _mm256_extractf32x4_ps(red1, 1));
#elif __AVX2__ && 0
	__m256 bcst1 = _mm256_permutevar8x32_ps(_mm256_castps128_ps256(v), _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1));
	__m256 bcst2 = _mm256_permutevar8x32_ps(_mm256_castps128_ps256(v), _mm256_setr_epi32(2, 2, 2, 2, 3, 3, 3, 3));
	__m256 mul1 = _mm256_mul_ps(ymm0, bcst1);
	__m256 mul2 = _mm256_mul_ps(ymm1, bcst2);
	__m256 res1 = _mm256_add_ps(mul1, mul2);
	return _mm_add_ps(_mm256_extractf32x4_ps(res1, 0), _mm256_extractf32x4_ps(res1, 1));
#else
	const float* p = (const float*)this;
	f32x4 p1 = load<f32x4>(p) * v.x;
	f32x4 p2 = load<f32x4>(p+4) * v.y;
	f32x4 p3 = load<f32x4>(p+8) * v.z;
	f32x4 p4 = load<f32x4>(p+12) * v.w;
	return vcast<__m128>((p1 + p2) + (p3 + p4));
#endif
}

const bob::_SSE_Vec4_float& Matrix4::operator[](int i) const
{
	return val[i];
}

bob::_SSE_Vec4_float& Matrix4::operator[](int i)
{
	return const_cast<bob::_SSE_Vec4_float&>(const_cast<const Matrix4*>(this)->operator[](i));
}

std::string Matrix4::toString(int precision) const
{
	std::stringstream ss;
	ss.precision(precision);

	std::string s[16];
	int sInd = 0;
	int maxLen = -1;

	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			ss << elements[i][j];
			s[sInd++] = ss.str();
			ss.str("");
			maxLen = std::max<int>(s[sInd - 1].length(), maxLen);
		}
	}

	std::string ret;
	for (int i = 0; i < 16; ++i)
	{
		ret += std::string(7 + precision - s[i].size(), ' ') + s[i];
		if (i % 4 == 3) ret += '\n';
	}
	return ret;
}

float Matrix4::det() const
{
	float ret = 0.0;
	float sign = 1;
	for (int i = 0; i < 4; ++i)
	{
		ret += sign * (*this)[0][i] * det3(0, i);
		sign = -sign;
	}
	return ret;
}

Matrix4 Matrix4::inverse() const
{
	float rcpDet = 1.0 / this->det();

	Matrix4 ret;
	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			float sign = (i + j) % 2 ? -1 : 1;
			ret[i][j] = det3(i, j) * sign;
		}
	}

	return (ret * rcpDet).transposed(); //TODO: why the hell is this transposed? The result is correct though. It is not correct without transposition. Det3 exclusion working bad?
}

Matrix4 Matrix4::rotationX(float theta)
{
	const float sinTheta = sin(theta);
	const float cosTheta = cos(theta);
	return {
		bob::_SSE_Vec4_float(cosTheta,	sinTheta, 0.0, 0.0),
		bob::_SSE_Vec4_float(-sinTheta, cosTheta, 0.0, 0.0),
		bob::_SSE_Vec4_float(0.0,		0.0,	  1.0, 0.0),
		bob::_SSE_Vec4_float(0.0,		0.0,	  0.0, 1.0),
	};
}

Matrix4 Matrix4::rotationY(float theta)
{
	const float sinTheta = sin(theta);
	const float cosTheta = cos(theta);
	return {
		bob::_SSE_Vec4_float(cosTheta,  0.0,	-sinTheta,	0.0),
		bob::_SSE_Vec4_float(0.0,		1.0,	0.0,		0.0),
		bob::_SSE_Vec4_float(sinTheta,  0.0,	cosTheta,	0.0),
		bob::_SSE_Vec4_float(0.0,		0.0,	0.0,		1.0),
	};
}

Matrix4 Matrix4::rotationZ(float theta)
{
	const float sinTheta = sin(theta);
	const float cosTheta = cos(theta);
	return {
		bob::_SSE_Vec4_float(1.0,	0.0,		0.0,		0.0),
		bob::_SSE_Vec4_float(0.0,	cosTheta,	sinTheta,	0.0),
		bob::_SSE_Vec4_float(0.0,	-sinTheta,	cosTheta,	0.0),
		bob::_SSE_Vec4_float(0.0,	0.0,		0.0,		1.0),
	};
}

Matrix4 Matrix4::rotationXYZ(const Vec4& angle)
{
	return rotationZ(angle.z) * rotationY(angle.y) * rotationX(angle.x);
	//return rotationX(angle.x) * rotationY(angle.y) * rotationZ(angle.z);
	//return (rotationZ(angle.z) * rotationY(angle.y) * rotationX(angle.x)).transposed();
}

Matrix4 Matrix4::identity(float value, int dim)
{
	/*
#if __AVX512F__
	__m512 bcst = _mm512_set1_ps(value);
	switch (dim)
	{
	case 1:
		return _mm512_setr_ps(value, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	case 2:
		return _mm512_setr_ps(value, 0, 0, 0, 0, value, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	case 3:
		return _mm512_setr_ps(value, 0, 0, 0, 0, value, 0, 0, 0, 0, value, 0, 0, 0, 0, 0);
	case 4:
		return _mm512_setr_ps(value, 0, 0, 0, 0, value, 0, 0, 0, 0, value, 0, 0, 0, 0, value);
	default:
		break;
	}
#else*/
	assert(dim > 0 && dim <= 4);
	Matrix4 ret = Matrix4::zeros();
	for (int i = 0; i < dim; ++i) ret.elements[i][i] = value;
	return ret;
	//#endif
}

Matrix4 Matrix4::zeros()
{
	return f32x16(0);
}

float Matrix4::det3(int excludeRow, int excludeCol) const
{
	float matr3x3[3][3];

	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			if (i != excludeRow && j != excludeCol)
			{
				int ni = i - (i > excludeRow);
				int nj = j - (j > excludeCol);
				matr3x3[ni][nj] = elements[i][j];
			}
		}
	}

	float a = matr3x3[0][0], b = matr3x3[0][1], c = matr3x3[0][2],
		d = matr3x3[1][0], e = matr3x3[1][1], f = matr3x3[1][2],
		g = matr3x3[2][0], h = matr3x3[2][1], i = matr3x3[2][2]; //TODO: maybe union?

	return a * e * i + b * f * g + c * d * h - c * e * g - b * d * i - a * f * h;
}


//todo: these will need to be changed to faster versions
AVXXY_NAMESPACE::f32x16 sin16(AVXXY_NAMESPACE::f32x16 x)
{
	AVXXY_NAMESPACE::f32x16 ret;
	for (int i = 0; i < 16; ++i) ret[i] = sinf(x[i]);
	return ret;
}
AVXXY_NAMESPACE::f32x16 cos16(AVXXY_NAMESPACE::f32x16 x)
{
	AVXXY_NAMESPACE::f32x16 ret;
	for (int i = 0; i < 16; ++i) ret[i] = cosf(x[i]);
	return ret;
}
MatrixPack16_4x4 MatrixPack16_4x4::rotationX(AVXXY_NAMESPACE::f32x16 theta)
{
	AVXXY_NAMESPACE::f32x16 sinTheta = sin16(theta), cosTheta = cos16(theta);
	MatrixPack16_4x4 ret;
	ret.elements[0][0] = cosTheta;
	ret.elements[0][1] = sinTheta;
	ret.elements[0][2] = 0.f;
	ret.elements[0][3] = 0.f;

	ret.elements[1][0] = -sinTheta;
	ret.elements[1][1] = cosTheta;
	ret.elements[1][2] = 0.f;
	ret.elements[1][3] = 0.f;

	ret.elements[2][0] = 0.f;
	ret.elements[2][1] = 0.f;
	ret.elements[2][2] = 1.f;
	ret.elements[2][3] = 0.f;

	ret.elements[3][0] = 0.f;
	ret.elements[3][1] = 0.f;
	ret.elements[3][2] = 0.f;
	ret.elements[3][3] = 1.f;
	return ret;
}

MatrixPack16_4x4 MatrixPack16_4x4::rotationY(AVXXY_NAMESPACE::f32x16 theta)
{
	AVXXY_NAMESPACE::f32x16 sinTheta = sin16(theta), cosTheta = cos16(theta);
	MatrixPack16_4x4 ret;
	ret.elements[0][0] = cosTheta;
	ret.elements[0][1] = 0.f;
	ret.elements[0][2] = -sinTheta;
	ret.elements[0][3] = 0.f;

	ret.elements[1][0] = 0.f;
	ret.elements[1][1] = 1.f;
	ret.elements[1][2] = 0.f;
	ret.elements[1][3] = 0.f;

	ret.elements[2][0] = sinTheta;
	ret.elements[2][1] = 0.f;
	ret.elements[2][2] = cosTheta;
	ret.elements[2][3] = 0.f;

	ret.elements[3][0] = 0.f;
	ret.elements[3][1] = 0.f;
	ret.elements[3][2] = 0.f;
	ret.elements[3][3] = 1.f;
	return ret;
}

MatrixPack16_4x4 MatrixPack16_4x4::rotationZ(AVXXY_NAMESPACE::f32x16 theta)
{
	AVXXY_NAMESPACE::f32x16 sinTheta = sin16(theta), cosTheta = cos16(theta);
	MatrixPack16_4x4 ret;
	ret.elements[0][0] = 1.f;
	ret.elements[0][1] = 0.f;
	ret.elements[0][2] = 0.f;
	ret.elements[0][3] = 0.f;

	ret.elements[1][0] = 0.f;
	ret.elements[1][1] = cosTheta;
	ret.elements[1][2] = sinTheta;
	ret.elements[1][3] = 0.f;

	ret.elements[2][0] = 0.f;
	ret.elements[2][1] = -sinTheta;
	ret.elements[2][2] = cosTheta;
	ret.elements[2][3] = 0.f;

	ret.elements[3][0] = 0.f;
	ret.elements[3][1] = 0.f;
	ret.elements[3][2] = 0.f;
	ret.elements[3][3] = 1.f;
	return ret;
}

MatrixPack16_4x4 MatrixPack16_4x4::rotationXYZ(const bob::Vec4_f32x16& angle)
{
	return rotationZ(angle.z) * rotationY(angle.y) * rotationX(angle.x);
}

MatrixPack16_4x4 MatrixPack16_4x4::fast_rotationX(AVXXY_NAMESPACE::f32x16 theta)
{
	AVXXY_NAMESPACE::f32x16 sinTheta = LUTMan::sin(theta), cosTheta = LUTMan::cos(theta);
	MatrixPack16_4x4 ret;
	ret.elements[0][0] = cosTheta;
	ret.elements[0][1] = sinTheta;
	ret.elements[0][2] = 0.f;
	ret.elements[0][3] = 0.f;

	ret.elements[1][0] = -sinTheta;
	ret.elements[1][1] = cosTheta;
	ret.elements[1][2] = 0.f;
	ret.elements[1][3] = 0.f;

	ret.elements[2][0] = 0.f;
	ret.elements[2][1] = 0.f;
	ret.elements[2][2] = 1.f;
	ret.elements[2][3] = 0.f;

	ret.elements[3][0] = 0.f;
	ret.elements[3][1] = 0.f;
	ret.elements[3][2] = 0.f;
	ret.elements[3][3] = 1.f;
	return ret;
}

MatrixPack16_4x4 MatrixPack16_4x4::fast_rotationY(AVXXY_NAMESPACE::f32x16 theta)
{
	AVXXY_NAMESPACE::f32x16 sinTheta = LUTMan::sin(theta), cosTheta = LUTMan::cos(theta);
	MatrixPack16_4x4 ret;
	ret.elements[0][0] = cosTheta;
	ret.elements[0][1] = 0.f;
	ret.elements[0][2] = -sinTheta;
	ret.elements[0][3] = 0.f;

	ret.elements[1][0] = 0.f;
	ret.elements[1][1] = 1.f;
	ret.elements[1][2] = 0.f;
	ret.elements[1][3] = 0.f;

	ret.elements[2][0] = sinTheta;
	ret.elements[2][1] = 0.f;
	ret.elements[2][2] = cosTheta;
	ret.elements[2][3] = 0.f;

	ret.elements[3][0] = 0.f;
	ret.elements[3][1] = 0.f;
	ret.elements[3][2] = 0.f;
	ret.elements[3][3] = 1.f;
	return ret;
}

MatrixPack16_4x4 MatrixPack16_4x4::fast_rotationZ(AVXXY_NAMESPACE::f32x16 theta)
{
	AVXXY_NAMESPACE::f32x16 sinTheta = LUTMan::sin(theta), cosTheta = LUTMan::cos(theta);
	MatrixPack16_4x4 ret;
	ret.elements[0][0] = 1.f;
	ret.elements[0][1] = 0.f;
	ret.elements[0][2] = 0.f;
	ret.elements[0][3] = 0.f;

	ret.elements[1][0] = 0.f;
	ret.elements[1][1] = cosTheta;
	ret.elements[1][2] = sinTheta;
	ret.elements[1][3] = 0.f;

	ret.elements[2][0] = 0.f;
	ret.elements[2][1] = -sinTheta;
	ret.elements[2][2] = cosTheta;
	ret.elements[2][3] = 0.f;

	ret.elements[3][0] = 0.f;
	ret.elements[3][1] = 0.f;
	ret.elements[3][2] = 0.f;
	ret.elements[3][3] = 1.f;
	return ret;
}

MatrixPack16_4x4 MatrixPack16_4x4::fast_rotationXYZ(const bob::Vec4_f32x16& angle)
{
	return fast_rotationZ(angle.z) * fast_rotationY(angle.y) * fast_rotationX(angle.x);
}


MatrixPack16_4x4 MatrixPack16_4x4::identity()
{
	MatrixPack16_4x4 ret;
	memset(&ret, 0, sizeof(ret));
	for (int i = 0; i < 4; ++i) ret.elements[i][i] = 1.f;
	return ret;
}
