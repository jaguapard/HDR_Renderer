#pragma once
#include "Vec.h"

//Provides very fast low-accuracy approximations of various functions
class LUTMan
{
public:
	static void init();
	static float32x16 sin(float32x16 x);
	static float32x16 cos(float32x16 x);
	static float32x16 log2(float32x16 x);

	struct __LutMan_tables_t
	{
		alignas(64) std::array<int16_t, 256> rgbToLinear_fp16; //unlike other LUTs here, this is exact aside rounding differences
		alignas(64) std::array<float, 256> rgbToLinear_fp32;  //unlike other LUTs here, this is exact aside rounding differences
		alignas(64) std::array<float, 32> sin_fp32, cos_fp32;
	};

	static const __LutMan_tables_t tables;
	//Provides a polynomial approximation of degree PolynomialDegree of function y = x^2.2. Only valid for 0 <= x <= 1
	template<typename T, int PolynomialDegree>
	static __forceinline T gamma2p2_toLinear(T x)
	{
		if constexpr (PolynomialDegree == 1)
		{
			return x * 9.8235255420328238e-001 + -1.7794104671501504e-001;
		}
		else if constexpr (PolynomialDegree == 2)
		{
			return x * x * 1.1330807809920531e+000 + x * -1.5072822678876630e-001 + 1.0165174443139940e-002;
		}
		else if constexpr (PolynomialDegree == 3)
		{
			return x * x * x * 1.7081311000596841e-001 + x * x * 8.7686111598309835e-001 + x * -4.8440792008458383e-002 + 1.7247345544788347e-003;
		}
		else if constexpr (PolynomialDegree == 4)
		{
			return x * x * x * x * -8.5677436345126590e-002 + x * x * x * 3.4216798269626630e-001 + x * x * 7.6684746665375036e-001 + x * -2.4104579024277189e-002 + 5.2936370664979937e-004;
		}
		else static_assert(false, "Unsupported polynomial degree request for LUTMan::rgbToLinear");
	}
private:
};