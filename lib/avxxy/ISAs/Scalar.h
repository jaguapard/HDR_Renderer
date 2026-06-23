#pragma once
#include "shared.h"
#include <iostream>

namespace AVXXY_NAMESPACE
{
	namespace internals
	{
#if 1
		struct ISA_Scalar {};
#else
		struct ISA_Scalar
		{
			//static inline constexpr FeatureSet FS = internals::FS_current;
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_add>)
			static SIMD_Vector<S, N> eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = a[i] + b[i];
				return ret;
			}

			template<typename Op, size_t N, typename From>
				requires (meta::IsCvtOp<Op>)
			static SIMD_Vector<typename Op::cvt_to_t, N> eval(const SIMD_Vector<From, N>& a)
			{
				scream();
				using To = typename Op::cvt_to_t;
				SIMD_Vector<To, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = a[i];
				return ret;
			}

			//TODO: can make return type bigger for larger N!
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_conflict> && sizeof(S) * 8 >= N)
			static SIMD_Vector<typename meta::ScalarTraits<S>::UintT, N> eval(const SIMD_Vector<S, N>& a)
			{
				using U = meta::ScalarTraits<S>::UintT;
				using T = SIMD_Vector<U, N>;
				T ret;
				for (size_t i = 0; i < N; ++i)
				{
					U acc = 0;
					for (size_t j = 0; j < i; ++j)
					{
						if (a[i] == a[j]) acc |= U(1) << j;
					}
					ret[i] = acc;
				}
				return ret;
			}

			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_sub>)
			static SIMD_Vector<S, N> eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = a[i] - b[i];
				return ret;
			}
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_mul>)
			static SIMD_Vector<S, N> eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = a[i] * b[i];
				return ret;
			}
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_div>)
			static SIMD_Vector<S, N> eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = a[i] / b[i];
				return ret;
			}
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_mod>)
			static SIMD_Vector<S, N> eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i)
					if constexpr (meta::any_int<S>) ret[i] = a[i] % b[i];
					else ret[i] = std::fmod(a[i], b[i]); //TODO: should this even exist?
				return ret;
			}


			//TODO: limit bitwise operations to int types?
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_or>)
			static SIMD_Vector<S, N> eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				using T = typename meta::ScalarTraits<S>::UintT;
				for (size_t i = 0; i < N; ++i) ret[i] = std::bit_cast<S>(std::bit_cast<T>(a[i]) | std::bit_cast<T>(b[i]));
				return ret;
			}
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_and>)
			static SIMD_Vector<S, N> eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				using T = typename meta::ScalarTraits<S>::UintT;
				for (size_t i = 0; i < N; ++i) ret[i] = std::bit_cast<S>(std::bit_cast<T>(a[i]) & std::bit_cast<T>(b[i]));
				return ret;
			}
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_xor>)
			static SIMD_Vector<S, N> eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				using T = typename meta::ScalarTraits<S>::UintT;
				for (size_t i = 0; i < N; ++i) ret[i] = std::bit_cast<S>(S(std::bit_cast<T>(a[i]) ^ std::bit_cast<T>(b[i])));
				return ret;
			}
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_not>)
			static SIMD_Vector<S, N> eval(const SIMD_Vector<S, N>& a)
			{
				scream();
				SIMD_Vector<S, N> ret;
				using T = typename meta::ScalarTraits<S>::UintT;
				for (size_t i = 0; i < N; ++i) ret[i] = std::bit_cast<S>(~std::bit_cast<T>(a[i]));
				return ret;
			}


			template<typename Op, typename S, size_t N, typename I>
				requires (std::same_as<Op, op_shl> && meta::any_int<S>&& meta::any_int<I>)
			static SIMD_Vector<S, N> eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				//using T = typename concepts::same_size_uint_t<S>::type;
				for (size_t i = 0; i < N; ++i) ret[i] = a[i] << b[i];
				return ret;
			}
			template<typename Op, typename S, size_t N, typename I>
				requires (std::same_as<Op, op_shr> && meta::any_int<S>&& meta::any_int<I>)
			static SIMD_Vector<S, N> eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				//using T = typename concepts::same_size_uint_t<S>::type;
				for (size_t i = 0; i < N; ++i) ret[i] = a[i] >> b[i];
				return ret;
			}

			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_sqrtf>)
			static SIMD_Vector<float, N> eval(const SIMD_Vector<S, N>& a)
			{
				scream();
				SIMD_Vector<float, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = std::sqrt(float(a[i]));
				return ret;
			}
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_sqrtd>)
			static SIMD_Vector<double, N> eval(const SIMD_Vector<S, N>& a)
			{
				scream();
				SIMD_Vector<double, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = std::sqrt(double(a[i]));
				return ret;
			}

			template<typename Op, typename S, size_t N, typename I>
				requires (meta::any_int<I>&& std::same_as<Op, op_permx>)
			static SIMD_Vector<S, N> eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& ind)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = a[ind[i] & (N - 1)];
				return ret;
			}
			template<typename Op, typename S, size_t N, typename I>
				requires (meta::any_int<I>&& std::same_as<Op, op_permx2>)
			static SIMD_Vector<S, N> eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b, const SIMD_Vector<I, N>& ind)
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

			template<typename Op, typename S, size_t N>
				requires (std::is_floating_point_v<S>&& std::same_as<Op, op_floor>)
			static SIMD_Vector<S, N> eval(const SIMD_Vector<S, N>& a)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = std::floor(a[i]);
				return ret;
			}
			template<typename Op, typename S, size_t N>
				requires (std::is_floating_point_v<S>&& std::same_as<Op, op_ceil>)
			static SIMD_Vector<S, N> eval(const SIMD_Vector<S, N>& a)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = std::ceil(a[i]);
				return ret;
			}


			template<typename Op>
				requires (meta::IsLoadOp<Op>)
			static SIMD_Vector<typename Op::S, Op::N> eval(const void* p, const typename SIMD_Vector<typename Op::S, Op::N>::MaskT& mask, const SIMD_Vector<typename Op::S, Op::N>& src)
			{
				scream();
				using S = typename Op::S;
				constexpr auto N = Op::N;

				SIMD_Vector<S, N> ret;
				const S* sp = static_cast<const S*>(p);
				for (size_t i = 0; i < N; ++i) ret[i] = mask[i] ? sp[i] : src[i];
				return ret;
			}
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_store>)
			static auto eval(SIMD_Vector<S, N> vec, void* p, const typename SIMD_Vector<S, N>::MaskT& mask)
			{
				scream();
				S* sp = static_cast<S*>(p);
				for (size_t i = 0; i < N; ++i) if (mask[i]) sp[i] = vec[i];
				return success_ack_t{};
			}
			template<typename Op, typename I>
				requires (meta::any_int<I>&& meta::IsGatherOp<Op>)
			static SIMD_Vector<typename Op::S, Op::N> eval(const void* base, const SIMD_Vector<I, Op::N>& ind, const typename SIMD_Vector<typename Op::S, Op::N>::MaskT& mask, const SIMD_Vector<typename Op::S, Op::N>& src = 0)
			{
				scream();
				SIMD_Vector<typename Op::S, Op::N> ret;
				size_t addr = size_t(base);
				for (size_t i = 0; i < Op::N; ++i) ret[i] = mask[i] ? *(const typename Op::S*)(addr + Op::Scale * ind[i]) : src[i];
				return ret;
			}
			template<typename Op, typename S, size_t N, typename I>
				requires (meta::any_int<I>&& meta::IsScatterOp<Op>)
			static auto eval(const SIMD_Vector<S, N>& v, void* base, const SIMD_Vector<I, N>& ind, const typename SIMD_Vector<S, N>::MaskT& mask)
			{
				scream();
				size_t addr = size_t(base);
				for (size_t i = 0; i < N; ++i) if (mask[i]) *(S*)(addr + Op::Scale * ind[i]) = v[i];
				return success_ack_t{};
			}


			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_compress>)
			static SIMD_Vector<S, N> eval(const typename SIMD_Vector<S, N>::MaskT& mask, const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& src = 0)
			{
				scream();
				SIMD_Vector<S, N> ret;
				size_t j = 0;
				for (size_t i = 0; i < N; ++i) if (mask[i]) ret[j++] = a[i];
				for (; j < N; ++j) ret[j] = src[j];
				return ret;
			}

			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_mask_mov>)
			static SIMD_Vector<S, N> eval(const SIMD_Vector<S, N>& ifBitClear, const typename SIMD_Vector<S, N>::MaskT& mask, const SIMD_Vector<S, N>& ifBitSet)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = mask[i] ? ifBitSet[i] : ifBitClear[i];
				return ret;
			}

			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_unpacklo>)
			static SIMD_Vector<S, N> eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				return unpack_base<S, N, true>(a, b);
			}
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_unpackhi>)
			static SIMD_Vector<S, N> eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				return unpack_base<S, N, false>(a, b);
			}

			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_cmpeq>)
			static typename SIMD_Vector<S, N>::MaskT eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				typename SIMD_Vector<S, N>::MaskT ret = 0;
				for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] == b[i]);
				return ret;
			}
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_cmpneq>)
			static typename SIMD_Vector<S, N>::MaskT eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				typename SIMD_Vector<S, N>::MaskT ret = 0;
				for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] != b[i]);
				return ret;
			}
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_cmplt>)
			static typename SIMD_Vector<S, N>::MaskT eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				typename SIMD_Vector<S, N>::MaskT ret = 0;
				for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] < b[i]);
				return ret;
			}
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_cmple>)
			static typename SIMD_Vector<S, N>::MaskT eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				typename SIMD_Vector<S, N>::MaskT ret = 0;
				for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] <= b[i]);
				return ret;
			}
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_cmpgt>)
			static typename SIMD_Vector<S, N>::MaskT eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				typename SIMD_Vector<S, N>::MaskT ret = 0;
				for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] > b[i]);
				return ret;
			}
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_cmpge>)
			static typename SIMD_Vector<S, N>::MaskT eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				typename SIMD_Vector<S, N>::MaskT ret = 0;
				for (size_t i = 0; i < N; ++i) ret.setBit(i, a[i] >= b[i]);
				return ret;
			}


			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_abs>)
			static SIMD_Vector<S, N> eval(const SIMD_Vector<S, N>& a)
			{
				scream();
				if constexpr (std::is_unsigned_v<S>) return a;
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = std::abs(a[i]);
				return ret;
			}
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_min>)
			static SIMD_Vector<S, N> eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = std::min(a[i], b[i]);
				return ret;
			}
			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_max>)
			static SIMD_Vector<S, N> eval(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = std::max(a[i], b[i]);
				return ret;
			}

			template<typename Op, meta::ScalarSizeClassEnum C, size_t N>
				requires (meta::IsMovmOp<Op>)
			static SIMD_Vector<typename Op::S, N> eval(const SIMD_Mask<C, N>& mask)
			{
				scream();
				SIMD_Vector<typename Op::S, N> ret;
				using Tr = meta::ScalarTraits<typename Op::S>;
				for (size_t i = 0; i < N; ++i)
				{
					typename Tr::UintT u = mask[i] ? Tr::AllOnesUint : 0;
					ret[i] = std::bit_cast<typename Op::S>(u);
				}
				return ret;
			}

			template<typename Op, typename S, size_t N>
				requires (std::same_as<Op, op_movemask>)
			static mask_t<S, N> eval(const SIMD_Vector<S, N>& vec)
			{
				mask_t<S, N> ret;
				using Tr = meta::ScalarTraits<S>;
				using U = Tr::UintT;
				for (size_t i = 0; i < N; ++i)
				{
					U sb = std::bit_cast<U>(vec[i]) & Tr::SignMask;
					ret.setBit(i, sb);
				}
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
#endif
	}
}