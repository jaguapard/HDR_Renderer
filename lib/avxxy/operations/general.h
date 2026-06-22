#pragma once
#include "shared.h"
namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		struct op_conflict : OperationBase
		{
			template<typename S, size_t N>
				requires (sizeof(S) * 8 >= N)
			static SIMD_Vector<typename ScalarTraits<S>::UintT, N> run(const SIMD_Vector<S, N>& a)
			{
				using U = ScalarTraits<S>::UintT;
				using T = SIMD_Vector<U, N>;
				T ret;
				scream();
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
		};

		struct op_div : OperationBase
		{
			template<typename S, size_t N>
			static SIMD_Vector<S, N> run(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = a[i] / b[i];
				return ret;
			}
		};

		struct op_mask_mov : OperationBase
		{
			template<typename S, size_t N>
			static SIMD_Vector<S, N> run(const SIMD_Vector<S, N>& ifBitClear, const mask_t<S,N>& mask, const SIMD_Vector<S, N>& ifBitSet)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = mask[i] ? ifBitSet[i] : ifBitClear[i];
				return ret;
			}
		};

		struct op_unpacklo : OperationBase
		{
			template<typename S, size_t N>
			static SIMD_Vector<S, N> run(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				return scalar_unpack_base<S, N, true>(a, b);
			}
		};

		struct op_unpackhi : OperationBase
		{
			template<typename S, size_t N>
			static SIMD_Vector<S, N> run(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				return scalar_unpack_base<S, N, false>(a, b);
			}
		};

		struct op_gather : OperationBase
		{
			template<typename S, size_t N, size_t Scale, typename I>
				requires (meta::any_int<I>)
			static SIMD_Vector<S, N> run(const void* base, const SIMD_Vector<I, N>& ind, const typename SIMD_Vector<S, N>::MaskT& mask, const SIMD_Vector<S, N>& src = 0)
			{
				scream();
				SIMD_Vector<S, N> ret;
				size_t addr = size_t(base);
				for (size_t i = 0; i < N; ++i) ret[i] = mask[i] ? *(const S*)(addr + Scale * ind[i]) : src[i];
				return ret;
			}
		};

		struct op_load : OperationBase
		{
			template<typename S, size_t N>
			static SIMD_Vector<S, N> run(const void* p, const mask_t<S, N>& mask, const SIMD_Vector<S, N>& src)
			{
				scream();
				SIMD_Vector<S, N> ret;
				const S* sp = static_cast<const S*>(p);
				for (size_t i = 0; i < N; ++i) ret[i] = mask[i] ? sp[i] : src[i];
				return ret;
			}
		};

		struct op_max : OperationBase
		{
			template<typename S, size_t N>
			static SIMD_Vector<S, N> run(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = std::max(a[i], b[i]);
				return ret;
			}
		};
		struct op_min : OperationBase
		{
			template<typename S, size_t N>
			static SIMD_Vector<S, N> run(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = std::min(a[i], b[i]);
				return ret;
			}
		};
		struct op_mul : OperationBase
		{
			template<typename S, size_t N>
			static SIMD_Vector<S, N> run(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = a[i] * b[i];
				return ret;
			}
		};
		struct op_not : OperationBase
		{
			template<typename S, size_t N>
			static SIMD_Vector<S, N> run(const SIMD_Vector<S, N>& a)
			{
				scream();
				SIMD_Vector<S, N> ret;
				using T = meta::ScalarTraits<S>::UintT;
				for (size_t i = 0; i < N; ++i) ret[i] = std::bit_cast<S>(~std::bit_cast<T>(a[i]));
				return ret;
			}
		};
		struct op_or : OperationBase
		{
			template<typename S, size_t N>
			static SIMD_Vector<S, N> run(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				using T = meta::ScalarTraits<S>::UintT;
				for (size_t i = 0; i < N; ++i) ret[i] = std::bit_cast<S>(std::bit_cast<T>(a[i]) | std::bit_cast<T>(b[i]));
				return ret;
			}
		};

		struct op_permx : OperationBase
		{
			template<typename S, size_t N, typename I>
				requires (meta::any_int<I>)
			static SIMD_Vector<S, N> run(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& ind)
			{
				scream();
				SIMD_Vector<S, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = a[ind[i] & (N - 1)];
				return ret;
			}
		};

		struct op_permx2 : OperationBase
		{
			template<typename S, size_t N, typename I>
				requires (meta::any_int<I>)
			static SIMD_Vector<S, N> run(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b, const SIMD_Vector<I, N>& ind)
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
		};

		struct op_scatter : OperationBase
		{
			template<typename S, size_t N, size_t Scale, typename I>
				requires (meta::any_int<I>)
			static void run(const SIMD_Vector<S, N>& v, void* base, const SIMD_Vector<I, N>& ind, const mask_t<S, N>& mask)
			{
				scream();
				size_t addr = size_t(base);
				for (size_t i = 0; i < N; ++i) if (mask[i]) *(S*)(addr + Scale * ind[i]) = v[i];
			}
		};

		struct op_shl : OperationBase
		{
			template<typename S, size_t N, typename I>
				requires (meta::any_int<S> && meta::any_int<I>)
			static SIMD_Vector<S, N> run(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				//using T = typename concepts::same_size_uint_t<S>::type;
				for (size_t i = 0; i < N; ++i) ret[i] = a[i] << b[i];
				return ret;
			}
		};
		struct op_shr : OperationBase
		{
			template<typename S, size_t N, typename I>
				requires (meta::any_int<S> && meta::any_int<I>)
			static SIMD_Vector<S, N> run(const SIMD_Vector<S, N>& a, const SIMD_Vector<I, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				//using T = typename concepts::same_size_uint_t<S>::type;
				for (size_t i = 0; i < N; ++i) ret[i] = a[i] >> b[i];
				return ret;
			}
		};

		struct op_sqrtf : OperationBase
		{
			template<typename S, size_t N>
			static SIMD_Vector<float, N> run(const SIMD_Vector<S, N>& a)
			{
				scream();
				SIMD_Vector<float, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = std::sqrt(float(a[i]));
				return ret;
			}
		};
		struct op_sqrtd : OperationBase
		{
			template<typename S, size_t N>
			static SIMD_Vector<float, N> run(const SIMD_Vector<S, N>& a)
			{
				scream();
				SIMD_Vector<float, N> ret;
				for (size_t i = 0; i < N; ++i) ret[i] = std::sqrt(double(a[i]));
				return ret;
			}
		};

		struct op_store : OperationBase
		{
			template<typename S, size_t N>
			static void run(SIMD_Vector<S, N> vec, void* p, const mask_t<S,N>& mask)
			{
				scream();
				S* sp = static_cast<S*>(p);
				for (size_t i = 0; i < N; ++i) if (mask[i]) sp[i] = vec[i];
			}
		};

		struct op_xor : OperationBase
		{
			template<typename S, size_t N>
			static SIMD_Vector<S, N> run(const SIMD_Vector<S, N>& a, const SIMD_Vector<S, N>& b)
			{
				scream();
				SIMD_Vector<S, N> ret;
				using T = typename meta::ScalarTraits<S>::UintT;
				for (size_t i = 0; i < N; ++i) ret[i] = std::bit_cast<S>(S(std::bit_cast<T>(a[i]) ^ std::bit_cast<T>(b[i])));
				return ret;
			}
		};

		struct op_movm : OperationBase
		{
			template<typename S, size_t N, meta::ScalarSizeClassEnum C>
			static SIMD_Vector<S, N> run(const SIMD_Mask<C, N>& mask)
			{
				scream();
				SIMD_Vector<S, N> ret;
				using Tr = ScalarTraits<S>;
				for (size_t i = 0; i < N; ++i)
				{
					typename Tr::UintT u = mask[i] ? Tr::AllOnesUint : 0;
					ret[i] = std::bit_cast<S>(u);
				}
				return ret;
			}
			//mask_t<S, N> movemask(const SIMD_Vector<S, N>& v)
		};

		struct op_movemask : OperationBase
		{
			template<typename S, size_t N>
			static mask_t<S, N> run(const SIMD_Vector<S, N>& vec)
			{
				mask_t<S, N> ret;
				using Tr = ScalarTraits<S>;
				using U = Tr::UintT;
				for (size_t i = 0; i < N; ++i)
				{
					U sb = std::bit_cast<U>(vec[i]) & Tr::SignMask;
					ret.setBit(i, sb);
				}
				return ret;
			}
		};
	}
}