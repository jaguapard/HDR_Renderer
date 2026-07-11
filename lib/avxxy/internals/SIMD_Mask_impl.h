#pragma once
#include "SIMD_Mask.h"
#include <iostream>
//#include "funcs.h"
#include "../SIMD_Vector.h"
namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		//returns `count` bits of uint64 starting from `start` shifted to lowest bits of output. Upper bits are zeroed-out
		template<uint64_t start, uint64_t count>
		constexpr uint64_t extract_u64_bits(uint64_t a)
		{
			uint64_t i = (a >> start);
			if constexpr (count == 64) return i;
			else
			{
				uint64_t m = (1ull << count) - 1;
				return i & m;
			}
		}

		//sets corresponding elements to all ones if corresponding mask bit is 1 or zero otherwise
		template<typename S, size_t N>
		SIMD_Vector<S, N> _movm_raw(uint64_t mask)
		{
			static_assert(IsValid_SIMD_Mask<N>);
			//static_assert(N >= 2 && N <= 64 && meta::isPowerOf2(N));
			using namespace meta;
			using T = SIMD_Vector<S, N>;
			using U = typename ScalarTraits<S>::UintT;
			if constexpr (!is_f32<S> && !is_f64<S> && !any_int<S>) return vcast<S>(_movm_raw<U>(mask));

			else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && any_i8<S>) return _mm512_movm_epi8(mask);
			else if constexpr (FS.has(AVX512_BW) && zmm_sized<T> && any_i16<S>) return _mm512_movm_epi16(mask);
			else if constexpr (FS.has(AVX512_DQ) && zmm_sized<T> && any_i32<S>) return _mm512_movm_epi32(mask);
			else if constexpr (FS.has(AVX512_DQ) && zmm_sized<T> && any_i64<S>) return _mm512_movm_epi64(mask);

			else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && ymm_sized<T> && any_i8<S>) return _mm256_movm_epi8(mask);
			else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && ymm_sized<T> && any_i16<S>) return _mm256_movm_epi16(mask);
			else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_DQ) && ymm_sized<T> && any_i32<S>) return _mm256_movm_epi32(mask);
			else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_DQ) && ymm_sized<T> && any_i64<S>) return _mm256_movm_epi64(mask);

			else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && xmm_sized<T> && any_i8<S>) return _mm_movm_epi8(mask);
			else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_BW) && xmm_sized<T> && any_i16<S>) return _mm_movm_epi16(mask);
			else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_DQ) && xmm_sized<T> && any_i32<S>) return _mm_movm_epi32(mask);
			else if constexpr (FS.has(AVX512_VL) && FS.has(AVX512_DQ) && xmm_sized<T> && any_i64<S>) return _mm_movm_epi64(mask);

			else if constexpr (FS.has(AVX2) && ymm_sized<T> && sizeof(S) == 8)
			{
				__m256i broadcasted = _mm256_set1_epi64x(mask);
				__m256i x = _mm256_andnot_si256(broadcasted, _mm256_setr_epi64x(1, 2, 4, 8));
				return T::fromBits(_mm256_cmpeq_epi64(x, _mm256_setzero_si256()));
			}
			else if constexpr (FS.has(AVX2) && ymm_sized<T> && sizeof(S) == 4)
			{
				__m256i broadcasted = _mm256_set1_epi32(mask);
				__m256i x = _mm256_andnot_si256(broadcasted, _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128));
				return T::fromBits(_mm256_cmpeq_epi32(x, _mm256_setzero_si256()));
			}
			else if constexpr (FS.has(AVX2) && ymm_sized<T> && sizeof(S) == 2)
			{
				__m256i broadcasted = _mm256_set1_epi16(mask);
				__m256i x = _mm256_andnot_si256(broadcasted, _mm256_setr_epi16(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 0x8000));
				return T::fromBits(_mm256_cmpeq_epi16(x, _mm256_setzero_si256()));
			}
			else if constexpr (FS.has(AVX2) && ymm_sized<T> && sizeof(S) == 1) //TODO test this and 128 bit version too
			{
				__m256i broadcasted = _mm256_set1_epi32(mask);
				broadcasted = _mm256_shuffle_epi8(broadcasted, _mm256_setr_epi64x(0, 0x0101010101010101, 0x0202020202020202, 0x0303030303030303));
				__m256i x = _mm256_andnot_si256(broadcasted, _mm256_setr_epi8(1, 2, 4, 8, 16, 32, 64, 0x7F, 1, 2, 4, 8, 16, 32, 64, 0x7F, 1, 2, 4, 8, 16, 32, 64, 0x7F, 1, 2, 4, 8, 16, 32, 64, 0x7F));
				return T::fromBits(_mm256_cmpeq_epi8(x, _mm256_setzero_si256()));
				//https://stackoverflow.com/questions/21622212/how-to-perform-the-inverse-of-mm256-movemask-epi8-vpmovmskb
				/*__m256i vmask(_mm256_set1_epi32(mask));
				const __m256i shuffle(_mm256_setr_epi64x(0x0000000000000000,
					0x0101010101010101, 0x0202020202020202, 0x0303030303030303));
				vmask = _mm256_shuffle_epi8(vmask, shuffle);
				const __m256i bit_mask(_mm256_set1_epi64x(0x7fbfdfeff7fbfdfe));
				vmask = _mm256_or_si256(vmask, bit_mask);
				return T::fromBits(_mm256_cmpeq_epi8(vmask, _mm256_set1_epi64x(-1)));*/
			}

			//TODO: emulations for older
			else if constexpr (FS.has(SSE41) && xmm_sized<T> && sizeof(S) == 8)
			{
				__m128i broadcasted = _mm_set1_epi64x(mask);
				__m128i x = _mm_andnot_si128(broadcasted, _mm_set_epi64x(2, 1));
				return T::fromBits(_mm_cmpeq_epi64(x, _mm_setzero_si128()));
			}
			else if constexpr (FS.has(SSE2) && xmm_sized<T> && sizeof(S) == 4)
			{
				__m128i broadcasted = _mm_set1_epi32(mask);
				__m128i x = _mm_andnot_si128(broadcasted, _mm_setr_epi32(1, 2, 4, 8));
				return T::fromBits(_mm_cmpeq_epi32(x, _mm_setzero_si128()));
			}
			else if constexpr (FS.has(SSE2) && xmm_sized<T> && sizeof(S) == 2) //only 8 bits can fit into register
			{
				__m128i broadcasted = _mm_set1_epi16(mask);
				__m128i x = _mm_andnot_si128(broadcasted, _mm_setr_epi16(1, 2, 4, 8, 16, 32, 64, 128));
				return T::fromBits(_mm_cmpeq_epi16(x, _mm_setzero_si128()));
			}
			else if constexpr (FS.has(SSSE3) && xmm_sized<T> && sizeof(S) == 1) //only 16 elements can fit into register
			{
				__m128i broadcasted = _mm_set1_epi16(mask);
				broadcasted = _mm_shuffle_epi8(broadcasted, _mm_setr_epi32(0, 0, 0x01010101, 0x01010101));
				__m128i x = _mm_andnot_si128(broadcasted, _mm_setr_epi8(1, 2, 4, 8, 16, 32, 64, 0x7F, 1, 2, 4, 8, 16, 32, 64, 0x7F));
				return T::fromBits(_mm_cmpeq_epi8(x, _mm_setzero_si128()));
			}
			else if constexpr (sizeof(T) > 16)
			{
				static_assert(N % 2 == 0);
				return T{ _movm_raw<S,N / 2>(extract_u64_bits<0,N / 2>(mask)), _movm_raw<S,N / 2>(extract_u64_bits<N / 2,N / 2>(mask)) };
			}
			else
			{
				//internals::scream();
				SIMD_Vector<S, N> ret;
				using Tr = meta::ScalarTraits<S>;
				for (size_t i = 0; i < N; ++i)
				{
					typename Tr::UintT u = (mask & (1ull << i)) ? Tr::AllOnesUint : 0;
					ret[i] = std::bit_cast<S>(u);
				}
				return ret;
			}
		}

		template<typename S, size_t N>
		uint64_t _movemask_raw(const SIMD_Vector<S, N>& a)
		{
			static_assert(N <= 64);
			using namespace meta;
			constexpr bool zmm_eligible = (FS.has(AVX512_F) && sizeof(S) >= 4) || (FS.has(AVX512_BW));
			constexpr bool xmm_ymm_eligible = zmm_eligible && FS.has(AVX512_VL);
			using T = SIMD_Vector<S, N>;
			using I = typename ScalarTraits<S>::IntT;
			constexpr bool bmask = mask_t<S, N>::IsBitMask;
			//no movemask, but comparison with 0 already acts like it. bmask to prevent recursion loops, since comparison will try to return mask vector and crush it to uint
			if constexpr (bmask && ((sizeof(T) > 32 && zmm_eligible) || (sizeof(T) <= 32 && xmm_ymm_eligible))) return (vcast<I>(a) < 0);
			else if constexpr (FS.has(AVX2) && ymm_sized<T> && sizeof(S) == 1) return _mm256_movemask_epi8(vcast<__m256i>(a));
			else if constexpr (FS.has(AVX) && ymm_sized<T> && sizeof(S) == 4) return _mm256_movemask_ps(vcast<__m256>(a));
			else if constexpr (FS.has(AVX2) && ymm_sized<T> && sizeof(S) == 2)
			{
				//AVX2 has no movemask_epi16 intrinsic
				// low bits -> upper bits go to the right, opposite to shifts
				//Post-shuffle layout: |01234567xxxxxxxx|xxxxxxxx89abcdef|, where x are always 0 and 0,1,..f are upper bytes of words
				//Extracting the mask, shifting and oring (| = byte boundary, 0..f = sign bits of words):
				//|01234567|xxxxxxxx|xxxxxxxx|89abcdef| OR
				//|xxxxxxxx|89abcdef|xxxxxxxx|xxxxxxxx|
				//|========|========|========|========|
				//|01234567|89abcdef|xxxxxxxx|89abcdef|
				//upper 16 bits are discarded
				__m256i shuf = _mm256_shuffle_epi8(vcast<__m256i>(a), _mm256_setr_epi8(
					1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1,
					-1, -1, -1, -1, -1, -1, -1, -1, 1, 3, 5, 7, 9, 11, 13, 15));
				uint32_t msk = _mm256_movemask_epi8(shuf);
				return uint16_t(msk | (msk >> 16));
			}

			else if constexpr (FS.has(SSE2) && xmm_sized<T> && sizeof(S) == 1) return _mm_movemask_epi8(vcast<__m128i>(a));
			else if constexpr (FS.has(SSSE3) && xmm_sized<T> && sizeof(S) == 2)
			{
				//move upper bytes of each word to lower 64 bits and zero out upper 64 bits
				__m128i shuf = _mm_shuffle_epi8(vcast<__m128i>(a), _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1));
				return _mm_movemask_epi8(shuf);
			}
			else if constexpr (FS.has(SSE) && xmm_sized<T> && sizeof(S) == 4) return _mm_movemask_ps(vcast<__m128d>(a));
			else if constexpr (FS.has(SSE2) && xmm_sized<T> && sizeof(S) == 8) return _mm_movemask_pd(vcast<__m128d>(a));
			else if constexpr (FS.has(SSE2) && xmm_sized<T> && sizeof(S) == 2)
			{
				__m128i ia = vcast<__m128i>(a);
				__m128i dup_lo = _mm_unpacklo_epi16(ia, ia); //duplicate each 16 bit element in low  half of a, so |abcdefgh| becomes |aabbccdd|
				__m128i dup_hi = _mm_unpackhi_epi16(ia, ia); //duplicate each 16 bit element in high half of a, so |abcdefgh| becomes |eeffgghh|
				int32_t interm1 = _mm_movemask_ps(_mm_castsi128_ps(dup_lo));
				int32_t interm2 = _mm_movemask_ps(_mm_castsi128_ps(dup_hi));
				return interm1 | (interm2 << 4);
			}

			else if constexpr (sizeof(T) > 16) return _movemask_raw(a.lo()) | (_movemask_raw(a.hi()) << N / 2);

			else
			{
				uint64_t ret = 0;
				for (size_t i = 0; i < N; ++i)
				{
					if (std::bit_cast<I>(a[i]) < 0) ret |= 1ull << i;
				}
				return ret;
			}
		}

		template<typename S, size_t N>
		SIMD_Vector<S, N> clean_mask_vector(const SIMD_Vector<S, N>& a)
		{
			using namespace meta;
			using T = SIMD_Vector<S, N>;
			if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i64<S>) return T::fromBits(_mm256_cmpgt_epi64(_mm256_setzero_si256(), vcast<__m256i>(a)));
			else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i32<S>) return T::fromBits(_mm256_cmpgt_epi32(_mm256_setzero_si256(), vcast<__m256i>(a)));
			else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i16<S>) return T::fromBits(_mm256_cmpgt_epi16(_mm256_setzero_si256(), vcast<__m256i>(a)));
			else if constexpr (FS.has(AVX2) && ymm_sized<T> && any_i8<S>) return T::fromBits(_mm256_cmpgt_epi8(_mm256_setzero_si256(), vcast<__m256i>(a)));
			else if constexpr (FS.has(SSE42) && xmm_sized<T> && any_i64<S>) return T::fromBits(_mm_cmpgt_epi64(_mm_setzero_si128(), vcast<__m128i>(a)));
			else if constexpr (FS.has(SSE2) && xmm_sized<T> && any_i32<S>) return T::fromBits(_mm_cmpgt_epi32(_mm_setzero_si128(), vcast<__m128i>(a)));
			else if constexpr (FS.has(SSE2) && xmm_sized<T> && any_i16<S>) return T::fromBits(_mm_cmpgt_epi16(_mm_setzero_si128(), vcast<__m128i>(a)));
			else if constexpr (FS.has(SSE2) && xmm_sized<T> && any_i8<S>) return T::fromBits(_mm_cmpgt_epi8(_mm_setzero_si128(), vcast<__m128i>(a)));
			else if constexpr (sizeof(T) > 16) return { clean_mask_vector(a.lo()), clean_mask_vector(a.hi()) };
			else
			{
				T ret;
				using I = ScalarTraits<S>::IntT;
				const I* p = (const I*)&a;
				for (size_t i = 0; i < N; ++i)
				{
					if (p[i] < 0) ret[i] = std::bit_cast<S>(ScalarTraits<S>::AllOnesUint);
					else ret[i] = std::bit_cast<S>(I(0));
				}
				return ret;
			}
		}
	}

	namespace internals
	{
		template<meta::ScalarSizeClassEnum LS, size_t N> requires IsValid_SIMD_Mask<N>
		inline SIMD_Mask<LS, N>::SIMD_Mask(BitsUintT bits)
		{
			if constexpr (IsBitMask) underlying = bits & AllOnesUint;
			else underlying = internals::_movm_raw<VecIntT, N>(bits & AllOnesUint);
		}

		template<meta::ScalarSizeClassEnum LS, size_t N> requires IsValid_SIMD_Mask<N>
		template<typename T>
		inline SIMD_Mask<LS, N>::SIMD_Mask(const SIMD_Vector<T, N>& vec)
		{
			*this = movemask(vec);
		}

		template<meta::ScalarSizeClassEnum LS, size_t N>  requires IsValid_SIMD_Mask<N>
		template<size_t N2> requires (N2 * 2 == N)
			inline SIMD_Mask<LS, N>::SIMD_Mask(const SIMD_Mask<LS, N2>& lo, const SIMD_Mask<LS, N2>& hi)
		{
			if constexpr (IsBitMask) underlying = BitsUintT(lo) | (BitsUintT(hi) << (N / 2));
			else underlying = { lo.underlying, hi.underlying };
		}

		template<meta::ScalarSizeClassEnum LS, size_t N> requires IsValid_SIMD_Mask<N>
		inline bool SIMD_Mask<LS, N>::operator[](size_t i) const
		{
			return BitsUintT(*this) & (BitsUintT(1) << i);
		}

		template<meta::ScalarSizeClassEnum LS, size_t N>  requires IsValid_SIMD_Mask<N>
		inline void SIMD_Mask<LS, N>::setBit(size_t i, bool value)
		{
			BitsUintT u = *this;
			u &= ~(BitsUintT(1) << i);
			*this = u | BitsUintT(value) << i;
		}

		template<meta::ScalarSizeClassEnum LS, size_t N> requires IsValid_SIMD_Mask<N>
		inline auto SIMD_Mask<LS, N>::lo() const requires (N >= 2)
		{
			static_assert(N % 2 == 0);
			SIMD_Mask<LS, N / 2> ret;
			if constexpr (IsBitMask) ret.underlying = underlying;
			else ret.underlying = underlying.lo();
			return ret;
		}
		template<meta::ScalarSizeClassEnum LS, size_t N> requires IsValid_SIMD_Mask<N>
		inline auto SIMD_Mask<LS, N>::hi() const requires (N >= 2)
		{
			static_assert(N % 2 == 0);
			SIMD_Mask<LS, N / 2> ret;
			if constexpr (IsBitMask) ret.underlying = underlying >> (N / 2);
			else ret.underlying = underlying.hi();
			return ret;
		}

		template<meta::ScalarSizeClassEnum LS, size_t N> requires IsValid_SIMD_Mask<N>
		inline SIMD_Mask<LS, N>::operator BitsUintT() const
		{
			if constexpr (IsBitMask) return underlying & AllOnesUint;
			else return internals::_movemask_raw(underlying) & AllOnesUint;
		}

		template<meta::ScalarSizeClassEnum LS, size_t N> requires IsValid_SIMD_Mask<N>
		template<typename T> requires (meta::IsIntrinsicVector<T> && (meta::ScalarSizeTraits<LS>::ByteSize* N == sizeof(T)))
			inline SIMD_Mask<LS, N>::operator T() const
		{
			if constexpr (IsBitMask) return vcast<T>(movm<VecIntT>(*this));
			else
			{
				return vcast<T>(internals::clean_mask_vector(underlying));
			}
		}

		template<meta::ScalarSizeClassEnum LS, size_t N> requires IsValid_SIMD_Mask<N>
		inline SIMD_Mask<LS, N> SIMD_Mask<LS, N>::operator&(const SIMD_Mask<LS, N>& other) const
		{
			SIMD_Mask<LS, N> ret;
			ret.underlying = underlying & other.underlying;
			return ret;
		}
		template<meta::ScalarSizeClassEnum LS, size_t N> requires IsValid_SIMD_Mask<N>
		inline SIMD_Mask<LS, N> SIMD_Mask<LS, N>::operator|(const SIMD_Mask<LS, N>& other) const
		{
			SIMD_Mask<LS, N> ret;
			ret.underlying = underlying | other.underlying;
			return ret;
		}
		template<meta::ScalarSizeClassEnum LS, size_t N> requires IsValid_SIMD_Mask<N>
		inline SIMD_Mask<LS, N> SIMD_Mask<LS, N>::operator^(const SIMD_Mask<LS, N>& other) const
		{
			SIMD_Mask<LS, N> ret;
			ret.underlying = underlying ^ other.underlying;
			return ret;
		}

		template<meta::ScalarSizeClassEnum LS, size_t N>  requires IsValid_SIMD_Mask<N>
		inline SIMD_Mask<LS, N> SIMD_Mask<LS, N>::operator~() const
		{
			SIMD_Mask<LS, N> ret;
			ret.underlying = ~underlying;
			return ret;
		}

		template<meta::ScalarSizeClassEnum LS, size_t N>  requires IsValid_SIMD_Mask<N>
		inline SIMD_Mask<LS, N>& SIMD_Mask<LS, N>::operator&=(const SIMD_Mask<LS, N>& other)
		{
			*this = *this & other;
			return *this;
		}
		template<meta::ScalarSizeClassEnum LS, size_t N>  requires IsValid_SIMD_Mask<N>
		inline SIMD_Mask<LS, N>& SIMD_Mask<LS, N>::operator|=(const SIMD_Mask<LS, N>& other)
		{
			*this = *this | other;
			return *this;
		}
		template<meta::ScalarSizeClassEnum LS, size_t N>  requires IsValid_SIMD_Mask<N>
		inline SIMD_Mask<LS, N>& SIMD_Mask<LS, N>::operator^=(const SIMD_Mask<LS, N>& other)
		{
			*this = *this ^ other;
			return *this;
		}

		template<meta::ScalarSizeClassEnum LS, size_t N> requires IsValid_SIMD_Mask<N>
		template<meta::ScalarSizeClassEnum LS2, size_t N2> requires (N >= N2)
			inline SIMD_Mask<LS, N>::SIMD_Mask(const SIMD_Mask<LS2, N2>& other)
		{
			if constexpr (IsBitMask) underlying = other;
			else underlying = other.underlying;
		}

		template<meta::ScalarSizeClassEnum LS, size_t N> requires IsValid_SIMD_Mask<N>
		template<typename T> requires (meta::IsIntrinsicVector<T>&& meta::SameSizeClasses<(sizeof(typename SIMD_Mask<LS, N>::VecT)), (sizeof(T))>)
			inline SIMD_Mask<LS, N>::SIMD_Mask(const T& intrinsicVec)
		{
			auto v = VecT::fromBits(intrinsicVec);
			if constexpr (IsBitMask) underlying = internals::_movemask_raw(v);
			else underlying = internals::clean_mask_vector(v);
		}

		template<meta::ScalarSizeClassEnum LS, size_t N>
		std::ostream& operator<<(std::ostream& os, const SIMD_Mask<LS, N>& mask)
		{
			for (size_t i = 0; i < mask; ++i)
			{
				os << int(mask[i]);
				if (i < N - 1) os << ",";
			}
			return os;
		}
	}
}