#pragma once
#include "../namespace.h"
#include "../tags.h"
#include "../SIMD_BitMask.h"
#include "../SIMD_Vector.h"
#include "../FeatureSet.h"
#include <iostream>
#include <source_location>

namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		namespace ISA
		{
			template<internals::FeatureSet FS>
			struct Scalar
			{
				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_add, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					scream();
					SIMD_Vector<S, N> ret;
					for (size_t i = 0; i < N; ++i) ret[i] = a[i] + b[i];
					return ret;
				}

				template<typename To, size_t N, typename From>
				static SIMD_Vector<To, N> eval(op_cvt<To>, const SIMD_Vector<From, N>& a)
				{
					scream();
					SIMD_Vector<To, N> ret;
					for (size_t i = 0; i < N; ++i) ret[i] = a[i];
					return ret;
				}

				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_sub, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					scream();
					SIMD_Vector<S, N> ret;
					for (size_t i = 0; i < N; ++i) ret[i] = a[i] - b[i];
					return ret;
				}
				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_mul, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					scream();
					SIMD_Vector<S, N> ret;
					for (size_t i = 0; i < N; ++i) ret[i] = a[i] * b[i];
					return ret;
				}
				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_div, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					scream();
					SIMD_Vector<S, N> ret;
					for (size_t i = 0; i < N; ++i) ret[i] = a[i] / b[i];
					return ret;
				}
				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_mod, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					scream();
					SIMD_Vector<S, N> ret;
					for (size_t i = 0; i < N; ++i)
						if constexpr (concepts::any_int<S>) ret[i] = a[i] % b[i];
						else ret[i] = std::fmod(a[i], b[i]); //TODO: should this even exist?
					return ret;
				}


				//TODO: limit bitwise operations to int types?
				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_or, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					scream();
					SIMD_Vector<S, N> ret;
					using T = typename concepts::same_size_uint_t<S>::type;
					for (size_t i = 0; i < N; ++i) ret[i] = std::bit_cast<S>(std::bit_cast<T>(a[i]) | std::bit_cast<T>(b[i]));
					return ret;
				}
				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_and, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					scream();
					SIMD_Vector<S, N> ret;
					using T = typename concepts::same_size_uint_t<S>::type;
					for (size_t i = 0; i < N; ++i) ret[i] = std::bit_cast<S>(std::bit_cast<T>(a[i]) & std::bit_cast<T>(b[i]));
					return ret;
				}
				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_xor, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					scream();
					SIMD_Vector<S, N> ret;
					using T = typename concepts::same_size_uint_t<S>::type;
					for (size_t i = 0; i < N; ++i) ret[i] = std::bit_cast<S>(std::bit_cast<T>(a[i]) ^ std::bit_cast<T>(b[i]));
					return ret;
				}
				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_not, const SIMD_Vector<S, N>& a)
				{
					scream();
					SIMD_Vector<S, N> ret;
					using T = typename concepts::same_size_uint_t<S>::type;
					for (size_t i = 0; i < N; ++i) ret[i] = std::bit_cast<S>(~std::bit_cast<T>(a[i]));
					return ret;
				}


				template<typename S, size_t N, typename I>
					requires (concepts::any_int<S>&& concepts::any_int<I>)
				static SIMD_Vector<S, N> eval(op_shl, const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& b)
				{
					scream();
					SIMD_Vector<S, N> ret;
					using T = typename concepts::same_size_uint_t<S>::type;
					for (size_t i = 0; i < N; ++i) ret[i] = a[i] << b[i];
					return ret;
				}
				template<typename S, size_t N, typename I>
					requires (concepts::any_int<S>&& concepts::any_int<I>)
				static SIMD_Vector<S, N> eval(op_shr, const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& b)
				{
					scream();
					SIMD_Vector<S, N> ret;
					using T = typename concepts::same_size_uint_t<S>::type;
					for (size_t i = 0; i < N; ++i) ret[i] = a[i] >> b[i];
					return ret;
				}

				template<typename S, size_t N>
				static SIMD_Vector<float, N> eval(op_sqrtf, const SIMD_Vector<S, N>& a)
				{
					scream();
					SIMD_Vector<float, N> ret;
					for (size_t i = 0; i < N; ++i) ret[i] = std::sqrt(float(a[i]));
					return ret;
				}
				template<typename S, size_t N>
				static SIMD_Vector<double, N> eval(op_sqrtd, const SIMD_Vector<S, N>& a)
				{
					scream();
					SIMD_Vector<double, N> ret;
					for (size_t i = 0; i < N; ++i) ret[i] = std::sqrt(double(a[i]));
					return ret;
				}

				template<typename S, size_t N, typename I>
					requires (concepts::any_int<I>)
				static SIMD_Vector<S, N> eval(op_permx, const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& ind)
				{
					scream();
					SIMD_Vector<S, N> ret;
					for (size_t i = 0; i < N; ++i) ret[i] = a[ind[i] & (N - 1)];
					return ret;
				}
				template<typename S, size_t N, typename I>
					requires (concepts::any_int<I>)
				static SIMD_Vector<S, N> eval(op_permx2, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b, const SIMD_Vector<I, N>& ind)
				{
					scream();
					SIMD_Vector<S, N> ret;
					for (size_t i = 0; i < N; ++i)
					{
						auto j = ind[i] & (2 * N - 1);
						ret[i] = j < N ? a[j] : b[j - N];
					}
					return ret;
				}

				template<typename S, size_t N>
					requires (std::is_floating_point_v<S>)
				static SIMD_Vector<S, N> eval(op_floor, const SIMD_Vector<S, N>& a)
				{
					scream();
					SIMD_Vector<S, N> ret;
					for (size_t i = 0; i < N; ++i) ret[i] = std::floor(a[i]);
					return ret;
				}
				template<typename S, size_t N>
					requires (std::is_floating_point_v<S>)
				static SIMD_Vector<S, N> eval(op_ceil, const SIMD_Vector<S, N>& a)
				{
					scream();
					SIMD_Vector<S, N> ret;
					for (size_t i = 0; i < N; ++i) ret[i] = std::ceil(a[i]);
					return ret;
				}


				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_load<S, N>, const void* p, const SIMD_BitMask<N>& mask = SIMD_BitMask<N>::AllOnes, const SIMD_Vector<S, N>& src = 0)
				{
					scream();
					SIMD_Vector<S, N> ret;
					const S* sp = static_cast<const S*>(p);
					for (size_t i = 0; i < N; ++i) ret[i] = mask[i] ? sp[i] : src[i];
					return ret;
				}
				template<typename S, size_t N>
				static void eval(op_store, SIMD_Vector<S, N> vec, void* p, const SIMD_BitMask<N>& mask = SIMD_BitMask<N>::AllOnes)
				{
					scream();
					S* sp = static_cast<S*>(p);
					for (size_t i = 0; i < N; ++i) if (mask[i]) sp[i] = vec[i];
				}
				template<typename S, size_t N, size_t Scale, typename I>
					requires (concepts::any_int<I>)
				static SIMD_Vector<S, N> eval(op_gather<S, N, Scale>, const void* base, const SIMD_Vector<I, N>& ind, const SIMD_BitMask<N>& mask = SIMD_BitMask<N>::AllOnes, const SIMD_Vector<S, N>& src = 0)
				{
					scream();
					SIMD_Vector<S, N> ret;
					size_t addr = size_t(base);
					for (size_t i = 0; i < N; ++i) ret[i] = mask[i] ? *(const S*)(addr + Scale * ind[i]) : src[i];
					return ret;
				}
				template<typename S, size_t N, size_t Scale, typename I>
					requires (concepts::any_int<I>)
				static void eval(op_scatter<Scale>, const SIMD_Vector<S, N>& v, void* base, const SIMD_Vector<I, N>& ind, const SIMD_BitMask<N>& mask = SIMD_BitMask<N>::AllOnes)
				{
					scream();
					size_t addr = size_t(base);
					for (size_t i = 0; i < N; ++i) if (mask[i]) *(S*)(addr + Scale * ind[i]) = v[i];
				}


				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_compress, const SIMD_BitMask<N>& mask, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& src = 0)
				{
					scream();
					SIMD_Vector<S, N> ret;
					size_t j = 0;
					for (size_t i = 0; i < N; ++i) if (mask[i]) ret[j++] = a[i];
					for (; j < N; ++j) ret[j] = src[j];
					return ret;
				}

				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_mask_mov, const SIMD_Vector<S, N>& ifBitClear, const SIMD_BitMask<N>& mask, const SIMD_Vector<S, N>& ifBitSet)
				{
					scream();
					SIMD_Vector<S, N> ret;
					for (size_t i = 0; i < N; ++i) ret[i] = mask[i] ? ifBitSet[i] : ifBitClear[i];
					return ret;
				}

				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_unpacklo, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					scream();
					return unpack_base<S, N, true>(a, b);
				}
				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_unpackhi, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					scream();
					return unpack_base<S, N, false>(a, b);
				}

				template<typename S, size_t N>
				static SIMD_BitMask<N> eval(op_cmpeq, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					scream();
					SIMD_BitMask<N> ret = 0;
					for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] == b[i]);
					return ret;
				}
				template<typename S, size_t N>
				static SIMD_BitMask<N> eval(op_cmpneq, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					scream();
					SIMD_BitMask<N> ret = 0;
					for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] != b[i]);
					return ret;
				}
				template<typename S, size_t N>
				static SIMD_BitMask<N> eval(op_cmplt, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					scream();
					SIMD_BitMask<N> ret = 0;
					for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] < b[i]);
					return ret;
				}
				template<typename S, size_t N>
				static SIMD_BitMask<N> eval(op_cmple, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					scream();
					SIMD_BitMask<N> ret = 0;
					for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] <= b[i]);
					return ret;
				}
				template<typename S, size_t N>
				static SIMD_BitMask<N> eval(op_cmpgt, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					scream();
					SIMD_BitMask<N> ret = 0;
					for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] > b[i]);
					return ret;
				}
				template<typename S, size_t N>
				static SIMD_BitMask<N> eval(op_cmpge, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					scream();
					SIMD_BitMask<N> ret = 0;
					for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] >= b[i]);
					return ret;
				}


				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_abs, const SIMD_Vector<S, N>& a)
				{
					scream();
					if constexpr (std::is_unsigned_v<S>) return a;
					SIMD_Vector<S, N> ret;
					for (size_t i = 0; i < N; ++i) ret[i] = std::abs(a[i]);
					return ret;
				}
				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_min, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					scream();
					SIMD_Vector<S, N> ret;
					for (size_t i = 0; i < N; ++i) ret[i] = std::min(a[i], b[i]);
					return ret;
				}
				template<typename S, size_t N>
				static SIMD_Vector<S, N> eval(op_max, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					scream();
					SIMD_Vector<S, N> ret;
					for (size_t i = 0; i < N; ++i) ret[i] = std::max(a[i], b[i]);
					return ret;
				}

				//TODO: verify that it works. Maybe a 2^16 entry LUT is better?
				static inline float fp16_to_fp32(uint16_t h)
				{
					uint32_t sign = (h & 0x8000) << 16;
					uint32_t exp = (h >> 10) & 0x1F;
					uint32_t frac = h & 0x03FF;

					uint32_t f;

					if (exp == 0)
					{
						if (frac == 0)
						{
							// Zero
							f = sign;
						}
						else
						{
							// Subnormal half -> normalized float

							// Normalize mantissa
							exp = 1;

							while ((frac & 0x0400) == 0)
							{
								frac <<= 1;
								exp--;
							}

							frac &= 0x03FF;

							uint32_t fp32_exp = exp + (127 - 15);
							uint32_t fp32_frac = frac << 13;

							f = sign | (fp32_exp << 23) | fp32_frac;
						}
					}
					else if (exp == 31)
					{
						// Inf or NaN
						uint32_t fp32_exp = 0xFF;
						uint32_t fp32_frac = frac << 13;

						f = sign | (fp32_exp << 23) | fp32_frac;
					}
					else
					{
						// Normalized number
						uint32_t fp32_exp = exp + (127 - 15);
						uint32_t fp32_frac = frac << 13;

						f = sign | (fp32_exp << 23) | fp32_frac;
					}

					float result;
					memcpy(&result, &f, sizeof(result));
					return result;
				}

				//TODO: verify that it works
				static inline uint16_t fp32_to_fp16(float x)
				{
					uint32_t f;
					memcpy(&f, &x, sizeof(f));

					uint32_t sign = (f >> 16) & 0x8000;
					uint32_t exp = (f >> 23) & 0xFF;
					uint32_t frac = f & 0x7FFFFF;

					// NaN or Inf
					if (exp == 0xFF)
					{
						if (frac == 0)
						{
							// Infinity
							return sign | 0x7C00;
						}
						else
						{
							// NaN
							uint16_t nan = frac >> 13;

							// Ensure mantissa nonzero
							if (nan == 0)
								nan = 1;

							return sign | 0x7C00 | nan;
						}
					}

					// Rebias exponent
					int32_t new_exp = (int32_t)exp - 127 + 15;

					// Overflow -> Inf
					if (new_exp >= 31)
					{
						return sign | 0x7C00;
					}

					// Underflow / subnormal
					if (new_exp <= 0)
					{
						// Too small -> zero
						if (new_exp < -10)
						{
							return sign;
						}

						// Produce subnormal FP16
						frac |= 0x800000;

						int shift = 14 - new_exp;

						uint32_t mant = frac >> shift;

						// Round to nearest even
						uint32_t round_bit = 1u << (shift - 1);

						if ((frac & round_bit) &&
							((frac & (round_bit - 1)) || (mant & 1)))
						{
							mant++;
						}

						return sign | (uint16_t)mant;
					}

					// Normalized FP16
					uint16_t half_exp = (uint16_t)(new_exp << 10);
					uint16_t half_frac = (uint16_t)(frac >> 13);

					// Round to nearest even
					uint32_t round_bits = frac & 0x1FFF;

					if (round_bits > 0x1000 || (round_bits == 0x1000 && (half_frac & 1)))
					{
						half_frac++;

						// Mantissa overflow
						if (half_frac == 0x400)
						{
							half_frac = 0;
							half_exp += 0x0400;

							// Exponent overflow
							if (half_exp >= 0x7C00)
							{
								half_exp = 0x7C00;
							}
						}
					}

					return sign | half_exp | half_frac;
				}

				template <size_t N>
				static SIMD_Vector<uint16_t, N> eval(op_fp32_to_fp16, const SIMD_Vector<float, N>& a)
				{
					scream();
					SIMD_Vector<uint16_t, N> ret;
					for (size_t i = 0; i < N; ++i) ret[i] = fp32_to_fp16(a[i]);
					return ret;
				}
				template <size_t N>
				static SIMD_Vector<float, N> eval(op_fp16_to_fp32, const SIMD_Vector<uint16_t, N>& a)
				{
					scream();
					SIMD_Vector<float, N> ret;
					for (size_t i = 0; i < N; ++i) ret[i] = fp16_to_fp32(a[i]);
					return ret;
				}
			private:
				template<typename S, size_t N, bool Lo>
				static SIMD_Vector<S, N> unpack_base(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
				{
					SIMD_Vector<S, N> ret;
					constexpr size_t pairs_per_xmm = 8 / sizeof(S); //8, since unpack only processes lower/upper half of each input
					constexpr size_t elements_per_xmm = 16 / sizeof(S); //how much elements of type S fit into one 128 bit lane
					constexpr size_t xmm_count = sizeof(ret) / 16;
					for (size_t xmm_i = 0; xmm_i < xmm_count; ++xmm_i) //for each 128-bit lane
					{
						for (size_t i = 0; i < elements_per_xmm; i += 2)
						{
							size_t srcI = xmm_i * elements_per_xmm + i / 2 + (Lo ? 0 : elements_per_xmm / 2);
							ret[xmm_i * elements_per_xmm + i] = a[srcI];
							ret[xmm_i * elements_per_xmm + i + 1] = b[srcI];
						}
					}
					return ret;
				}

				//scream your lungs out if scalar fallback is reached and this function is enabled via AVXXY_NOISY_SCALAR define
				static void scream(std::source_location loc = std::source_location::current())
				{
#ifdef AVXXY_NOISY_SCALAR
					std::cout << "\nScalar fallback reached:" << loc.function_name() << "\n";
#endif
				}
			};
		}
	}
}