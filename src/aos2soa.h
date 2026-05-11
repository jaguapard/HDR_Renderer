#pragma once
#include "helpers.h"
//TODO: all these are not tested rigorously, should test!
/**
@brief Loads and transposes up to 16 packs of FieldCount contigious 4-byte elements and trasposes them to SoA layout
@note This functions assumes no guarantees from the input and thus has lower performance. Consider using other versions.
@details
Pseudocode:
for i in [0,15]:
	if mask[i]:
		tmp = load contigious (FieldCount*4) bytes starting at byte uintptr_t(base) + ind[i]*(FieldCount*4)
		for j in [0, FieldCount-1]:
			ret[j][i*4..i*4+3] = tmp[j*4..j*4+3]

@param base: the pointer to the start of collection that indices are defined relative to
@param ind: 32-bit indices of the packs to be gathered and transposed
@param mask: mask with bits set for active packs. Packs at inactive indices are not read from memory and have undefined values
@returns Array of FieldCount 512-bit vectors, with values transposed to SoA layout
*/
template<typename ReturnType, uint32_t FieldCount>
__forceinline std::array<ReturnType, FieldCount> aos2soa_gather_and_transpose_safe(const void* base, __m512i ind, __mmask16 mask)
	requires (sizeof(ReturnType) == 64)
{
	std::array<ReturnType, FieldCount> ret;
	const float* p = (const float*)base;
#ifdef VS_CLANG //Specialized versions below rely very heavily on Clang optimizations, and if they fail, that will become horrible garbage code, probably slower than gathers anyway.
	//at small sizes, transpose overhead is too high relative to gatherless-gain (?). At big, there's just too much shuffling?
	//ind can overflow if multiplied in place, causing silent corruption of the results. Thus, extend and multiply
	uint64_t offsets[16];
	const uint64_t rawBase = (const uint64_t)(base);
	__m512i indLo = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(ind, 0));
	__m512i indHi = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(ind, 1));
	__m512i offsetLo = _mm512_mullo_epi64(indLo, _mm512_set1_epi64(FieldCount * 4));
	__m512i offsetHi = _mm512_mullo_epi64(indHi, _mm512_set1_epi64(FieldCount * 4));
	_mm512_storeu_si512(&offsets[0], offsetLo); //can use *4 for easier addressing modes for free, since multiplication and extension is already required
	_mm512_storeu_si512(&offsets[8], offsetHi);

	if constexpr (FieldCount == 4) //can probaly expand it to 2 as well. 3 requires ps masked loads
	{
		float r0[16], r1[16], r2[16], r3[16];
		__mmask32 m = duplicate_mmask_bits_16_to_32(mask);
		for (int i = 0; i < 16; i += 4)
		{
			__m128 v0 = _mm_castpd_ps(_mm_maskz_loadu_pd(m >> (i * 2), (const void*)(rawBase + offsets[i])));
			__m128 v1 = _mm_castpd_ps(_mm_maskz_loadu_pd(m >> (i * 2 + 2), (const void*)(rawBase + offsets[i + 1])));
			__m128 v2 = _mm_castpd_ps(_mm_maskz_loadu_pd(m >> (i * 2 + 4), (const void*)(rawBase + offsets[i + 2])));
			__m128 v3 = _mm_castpd_ps(_mm_maskz_loadu_pd(m >> (i * 2 + 6), (const void*)(rawBase + offsets[i + 3])));

			_mm_storeu_ps(&r0[i], v0); //r0 = abcd0,abcd4,abcd8,abcd12
			_mm_storeu_ps(&r1[i], v1); //r1 = abcd1,abcd5,abcd9,abcd13
			_mm_storeu_ps(&r2[i], v2); //r2 = abcd2,abcd6,abcd10,abcd14
			_mm_storeu_ps(&r3[i], v3); //r3 = abcd3,abcd7,abcd11,abcd15
		}

		__m512 aabb01 = _mm512_unpacklo_ps(_mm512_loadu_ps(r0), _mm512_loadu_ps(r1));
		__m512 aabb23 = _mm512_unpacklo_ps(_mm512_loadu_ps(r2), _mm512_loadu_ps(r3));
		__m512 ccdd01 = _mm512_unpackhi_ps(_mm512_loadu_ps(r0), _mm512_loadu_ps(r1));
		__m512 ccdd23 = _mm512_unpackhi_ps(_mm512_loadu_ps(r2), _mm512_loadu_ps(r3));
		_mm512_storeu_pd(&ret[0], _mm512_unpacklo_pd(_mm512_castps_pd(aabb01), _mm512_castps_pd(aabb23)));
		_mm512_storeu_pd(&ret[1], _mm512_unpackhi_pd(_mm512_castps_pd(aabb01), _mm512_castps_pd(aabb23)));
		_mm512_storeu_pd(&ret[2], _mm512_unpacklo_pd(_mm512_castps_pd(ccdd01), _mm512_castps_pd(ccdd23)));
		_mm512_storeu_pd(&ret[3], _mm512_unpackhi_pd(_mm512_castps_pd(ccdd01), _mm512_castps_pd(ccdd23)));
		return ret;
	}
	if constexpr (FieldCount == 6 || FieldCount == 8) //odd counts can't use pd loads and mask management becomes harder //else if constexpr (FieldCount > 4 && FieldCount <= 8) //can be expanded to >4 && <= 8 fields, algorithm already supports it, only mask adjustment is needed
	{
		__mmask64 m = duplicate_mmask_bits_16_to_64(mask);
		if constexpr (FieldCount == 6) m &= 0x7777777777777777; //mask off MSB of each nibble, emulating limiting the following masked loads to 6 contigious 4-byte elements
		for (int i = 0; i < 16; i += 4)
		{
			__m256d v0 = _mm256_maskz_loadu_pd(m >> (i * 4), (const void*)(rawBase + offsets[i]));          //|abcd|efgh|_0/4/8/12
			__m256d v1 = _mm256_maskz_loadu_pd(m >> (i * 4 + 4), (const void*)(rawBase + offsets[i + 1]));  //|abcd|efgh|_1/5/9/13
			__m256d v2 = _mm256_maskz_loadu_pd(m >> (i * 4 + 8), (const void*)(rawBase + offsets[i + 2]));  //|abcd|efgh|_2/6/10/14
			__m256d v3 = _mm256_maskz_loadu_pd(m >> (i * 4 + 12), (const void*)(rawBase + offsets[i + 3])); //|abcd|efgh|_3/7/11/15

			__m256 tmp0 = _mm256_unpacklo_ps(v0, v1); //|a0,a1,b0,b1|e0,e1,f0,f1|
			__m256 tmp1 = _mm256_unpacklo_ps(v2, v3); //|a2,a3,b2,b3|e2,e3,f2,f3|
			__m256 tmp2 = _mm256_unpackhi_ps(v0, v1); //|c0,c1,d0,d1|g0,g1,h0,h1|
			__m256 tmp3 = _mm256_unpackhi_ps(v2, v3); //|c2,c3,d2,d3|g2,g3,h2,h3|
			__m256d ae = _mm256_unpacklo_pd(tmp0, tmp1); //|a0,a1,a2,a3|e0,e1,e2,e3|
			__m256d bf = _mm256_unpackhi_pd(tmp0, tmp1); //|b0,b1,b2,b3|f0,f1,f2,f3|
			__m256d cg = _mm256_unpacklo_pd(tmp2, tmp3); //|c0,c1,c2,c3|g0,g1,g2,g3|
			__m256d dh = _mm256_unpackhi_pd(tmp2, tmp3); //|d0,d1,d2,d3|h0,h1,h2,h3|

			_mm_storeu_ps((float*)&ret[0] + i, _mm256_extractf128_ps(ae, 0));
			_mm_storeu_ps((float*)&ret[1] + i, _mm256_extractf128_ps(bf, 0));
			_mm_storeu_ps((float*)&ret[2] + i, _mm256_extractf128_ps(cg, 0));
			_mm_storeu_ps((float*)&ret[3] + i, _mm256_extractf128_ps(dh, 0));
			_mm_storeu_ps((float*)&ret[4] + i, _mm256_extractf128_ps(ae, 1));
			if constexpr (FieldCount > 5) _mm_storeu_ps((float*)&ret[5] + i, _mm256_extractf128_ps(bf, 1));
			if constexpr (FieldCount > 6) _mm_storeu_ps((float*)&ret[6] + i, _mm256_extractf128_ps(cg, 1));
			if constexpr (FieldCount > 7) _mm_storeu_ps((float*)&ret[7] + i, _mm256_extractf128_ps(dh, 1));
		}
		return ret;
	}
#endif

	//if no specialized version available, then just gather as normal.
	const float* fp = (const float*)base;
	for (int i = 0; i < FieldCount; ++i)
	{
		__m256 tmp1 = _mm512_mask_i64gather_ps(_mm256_setzero_ps(), mask, offsetLo, fp + i, 1); //1 scale, since offsets are already calculated in bytes, not floats
		__m256 tmp2 = _mm512_mask_i64gather_ps(_mm256_setzero_ps(), mask >> 8, offsetHi, fp + i, 1);
		__m512 tmp = _mm512_insertf32x8(_mm512_castps256_ps512(tmp1), tmp2, 1);
		_mm512_storeu_ps(&ret[i], tmp);
	}
	return ret;
}


/**
@brief Loads and transposes up to 16 packs of FieldCount contigious 4-byte elements and trasposes them to SoA layout.
@brief This function assumes that the mask argument is non-zero and has undefined behavior if the mask is zero
@details
Pseudocode:
for i in [0,15]:
	if mask[i]:
		tmp = load contigious (FieldCount*4) bytes starting at byte uintptr_t(base) + ind[i]*(FieldCount*4)
		for j in [0, FieldCount-1]:
			ret[j][i*4..i*4+3] = tmp[j*4..j*4+3]

@param base: the pointer to the start of collection that indices are defined relative to
@param ind: 32-bit indices of the packs to be gathered and transposed
@param mask: mask with bits set for active packs. Packs at inactive indices are not read from memory and have undefined values
@returns Array of FieldCount 512-bit vectors, with values transposed to SoA layout
*/
template<typename ReturnType, uint32_t FieldCount>
__forceinline std::array<ReturnType, FieldCount> aos2soa_gather_and_transpose_nonzero_mask(const void* base, __m512i ind, __mmask16 mask)
	requires (sizeof(ReturnType) == 64)
{
	assert(mask != 0);
	std::array<ReturnType, FieldCount> ret;
	__m512i compressedInd = _mm512_maskz_compress_epi32(mask, ind);
	ind = _mm512_mask_mov_epi32(_mm512_broadcastd_epi32(_mm512_castsi512_si128(compressedInd)), mask, ind); //unmasked elements will use safe index for dummy load (first found valid ind)

	//ind can overflow if multiplied in place, causing silent corruption of the results. Thus, extend and multiply
	uint64_t offsets[16];
	const uint64_t rawBase = (const uint64_t)(base);
	__m512i indLo = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(ind, 0));
	__m512i indHi = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(ind, 1));
	__m512i offsetLo = _mm512_mullo_epi64(indLo, _mm512_set1_epi64(FieldCount * 4));
	__m512i offsetHi = _mm512_mullo_epi64(indHi, _mm512_set1_epi64(FieldCount * 4));
	_mm512_storeu_si512(&offsets[0], offsetLo); //can use *4 for easier addressing modes for free, since multiplication and extension is already required
	_mm512_storeu_si512(&offsets[8], offsetHi);
	constexpr uint64_t packLoadMask = (1ull << FieldCount) - 1; //avoid touching OOB for tails. Load only FieldCount lower floats

#ifdef VS_CLANG //Specialized versions below rely very heavily on Clang optimizations, and if they fail, that will become horrible garbage code, probably slower than gathers anyway.
	if constexpr (FieldCount > 2 && FieldCount <= 4)
	{
		float r0[16], r1[16], r2[16], r3[16];
		for (int i = 0; i < 16; i += 4)
		{
			__m128 v0 = _mm_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[i]));
			__m128 v1 = _mm_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[i + 1]));
			__m128 v2 = _mm_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[i + 2]));
			__m128 v3 = _mm_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[i + 3]));

			_mm_storeu_ps(&r0[i], v0); //r0 = abcd0,abcd4,abcd8,abcd12
			_mm_storeu_ps(&r1[i], v1); //r1 = abcd1,abcd5,abcd9,abcd13
			_mm_storeu_ps(&r2[i], v2); //r2 = abcd2,abcd6,abcd10,abcd14
			_mm_storeu_ps(&r3[i], v3); //r3 = abcd3,abcd7,abcd11,abcd15
		}

		__m512 aabb01 = _mm512_unpacklo_ps(_mm512_loadu_ps(r0), _mm512_loadu_ps(r1));
		__m512 aabb23 = _mm512_unpacklo_ps(_mm512_loadu_ps(r2), _mm512_loadu_ps(r3));
		__m512 ccdd01 = _mm512_unpackhi_ps(_mm512_loadu_ps(r0), _mm512_loadu_ps(r1));
		__m512 ccdd23 = _mm512_unpackhi_ps(_mm512_loadu_ps(r2), _mm512_loadu_ps(r3));
		_mm512_storeu_pd(&ret[0], _mm512_unpacklo_pd(_mm512_castps_pd(aabb01), _mm512_castps_pd(aabb23)));
		_mm512_storeu_pd(&ret[1], _mm512_unpackhi_pd(_mm512_castps_pd(aabb01), _mm512_castps_pd(aabb23)));
		_mm512_storeu_pd(&ret[2], _mm512_unpacklo_pd(_mm512_castps_pd(ccdd01), _mm512_castps_pd(ccdd23)));
		_mm512_storeu_pd(&ret[3], _mm512_unpackhi_pd(_mm512_castps_pd(ccdd01), _mm512_castps_pd(ccdd23)));
		return ret;
	}
	if constexpr (FieldCount > 4 && FieldCount <= 8)
	{
		for (int i = 0; i < 16; i += 4)
		{
			__m256 v0 = _mm256_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[i]));    //|abcd|efgh|_0/4/8/12
			__m256 v1 = _mm256_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[i + 1]));  //|abcd|efgh|_1/5/9/13
			__m256 v2 = _mm256_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[i + 2]));  //|abcd|efgh|_2/6/10/14
			__m256 v3 = _mm256_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[i + 3]));  //|abcd|efgh|_3/7/11/15

			__m256 tmp0 = _mm256_unpacklo_ps(v0, v1); //|a0,a1,b0,b1|e0,e1,f0,f1|
			__m256 tmp1 = _mm256_unpacklo_ps(v2, v3); //|a2,a3,b2,b3|e2,e3,f2,f3|
			__m256 tmp2 = _mm256_unpackhi_ps(v0, v1); //|c0,c1,d0,d1|g0,g1,h0,h1|
			__m256 tmp3 = _mm256_unpackhi_ps(v2, v3); //|c2,c3,d2,d3|g2,g3,h2,h3|
			__m256d ae = _mm256_unpacklo_pd(tmp0, tmp1); //|a0,a1,a2,a3|e0,e1,e2,e3|
			__m256d bf = _mm256_unpackhi_pd(tmp0, tmp1); //|b0,b1,b2,b3|f0,f1,f2,f3|
			__m256d cg = _mm256_unpacklo_pd(tmp2, tmp3); //|c0,c1,c2,c3|g0,g1,g2,g3|
			__m256d dh = _mm256_unpackhi_pd(tmp2, tmp3); //|d0,d1,d2,d3|h0,h1,h2,h3|

			_mm_storeu_ps((float*)&ret[0] + i, _mm256_extractf128_ps(ae, 0));
			_mm_storeu_ps((float*)&ret[1] + i, _mm256_extractf128_ps(bf, 0));
			_mm_storeu_ps((float*)&ret[2] + i, _mm256_extractf128_ps(cg, 0));
			_mm_storeu_ps((float*)&ret[3] + i, _mm256_extractf128_ps(dh, 0));
			_mm_storeu_ps((float*)&ret[4] + i, _mm256_extractf128_ps(ae, 1));
			if constexpr (FieldCount > 5) _mm_storeu_ps((float*)&ret[5] + i, _mm256_extractf128_ps(bf, 1));
			if constexpr (FieldCount > 6) _mm_storeu_ps((float*)&ret[6] + i, _mm256_extractf128_ps(cg, 1));
			if constexpr (FieldCount > 7) _mm_storeu_ps((float*)&ret[7] + i, _mm256_extractf128_ps(dh, 1));
		}
		/* seems slower
		for (int i = 0; i < 16; i += 8)
		{
			__m256 v0 = _mm256_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[i]));    //|abcd|efgh|_0/8
			__m256 v1 = _mm256_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[i + 1]));  //|abcd|efgh|_1/9
			__m256 v2 = _mm256_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[i + 2]));  //|abcd|efgh|_2/10
			__m256 v3 = _mm256_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[i + 3]));  //|abcd|efgh|_3/11
			__m256 v4 = _mm256_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[i + 4]));  //|abcd|efgh|_4/12
			__m256 v5 = _mm256_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[i + 5]));  //|abcd|efgh|_5/13
			__m256 v6 = _mm256_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[i + 6]));  //|abcd|efgh|_6/14
			__m256 v7 = _mm256_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[i + 7]));  //|abcd|efgh|_7/15

			__m512 ins0 = _mm512_insertf32x8(_mm512_castps256_ps512(v0), v4, 1); //|abcd0|efgh0|abcd4|efgh4|
			__m512 ins1 = _mm512_insertf32x8(_mm512_castps256_ps512(v1), v5, 1); //|abcd1|efgh1|abcd5|efgh5|
			__m512 ins2 = _mm512_insertf32x8(_mm512_castps256_ps512(v2), v6, 1); //|abcd2|efgh2|abcd6|efgh6|
			__m512 ins3 = _mm512_insertf32x8(_mm512_castps256_ps512(v3), v7, 1); //|abcd3|efgh3|abcd7|efgh7|

			__m512 tmp0 = _mm512_unpacklo_ps(ins0, ins1); //|a0,a1,b0,b1|e0,e1,f0,f1|a4,a5,b4,b5|e4,e5,f4,f5|
			__m512 tmp1 = _mm512_unpacklo_ps(ins2, ins3); //|a2,a3,b2,b3|e2,e3,f2,f3|a6,a7,b6,b7|e6,e7,f6,f7|
			__m512 tmp2 = _mm512_unpackhi_ps(ins0, ins1); //|c0,c1,d0,d1|g0,g1,h0,h1|c4,c5,d4,d5|g4,g5,h4,h5|
			__m512 tmp3 = _mm512_unpackhi_ps(ins2, ins3); //|c2,c3,d2,d3|g2,g3,h2,h3|c6,c7,d6,d7|g6,g7,h6,h7|

			__m512d aeae = _mm512_unpacklo_pd(tmp0, tmp1); //|a0,a1,a2,a3|e0,e1,e2,e3|a4,a5,a6,a7|e4,e5,e6,e7|
			__m512d bfbf = _mm512_unpackhi_pd(tmp0, tmp1); //|b0,b1,b2,b3|f0,f1,f2,f3|b4,b5,b6,b7|f4,f5,f6,f7|
			__m512d cgcg = _mm512_unpacklo_pd(tmp2, tmp3); //|c0,c1,c2,c3|g0,g1,g2,g3|c4,c5,c6,c7|g4,g5,g6,g7|
			__m512d dhdh = _mm512_unpackhi_pd(tmp2, tmp3); //|d0,d1,d2,d3|h0,h1,h2,h3|d4,d5,d6,d7|h4,h5,h6,h7|

			_mm_storeu_ps((float*)&ret[0] + i, _mm512_extractf32x4_ps(aeae, 0));
			_mm_storeu_ps((float*)&ret[0] + i + 4, _mm512_extractf32x4_ps(aeae, 2));
			_mm_storeu_ps((float*)&ret[1] + i, _mm512_extractf32x4_ps(bfbf, 0));
			_mm_storeu_ps((float*)&ret[1] + i + 4, _mm512_extractf32x4_ps(bfbf, 2));
			_mm_storeu_ps((float*)&ret[2] + i, _mm512_extractf32x4_ps(cgcg, 0));
			_mm_storeu_ps((float*)&ret[2] + i + 4, _mm512_extractf32x4_ps(cgcg, 2));
			_mm_storeu_ps((float*)&ret[3] + i, _mm512_extractf32x4_ps(dhdh, 0));
			_mm_storeu_ps((float*)&ret[3] + i + 4, _mm512_extractf32x4_ps(dhdh, 2));
			_mm_storeu_ps((float*)&ret[4] + i, _mm512_extractf32x4_ps(aeae, 1));
			_mm_storeu_ps((float*)&ret[4] + i + 4, _mm512_extractf32x4_ps(aeae, 3));
			if constexpr (FieldCount > 5) {
				_mm_storeu_ps((float*)&ret[5] + i, _mm512_extractf32x4_ps(bfbf, 1));
				_mm_storeu_ps((float*)&ret[5] + i + 4, _mm512_extractf32x4_ps(bfbf, 3));
			}
			if constexpr (FieldCount > 6) {
				_mm_storeu_ps((float*)&ret[6] + i, _mm512_extractf32x4_ps(cgcg, 1));
				_mm_storeu_ps((float*)&ret[6] + i + 4, _mm512_extractf32x4_ps(cgcg, 3));
			}
			if constexpr (FieldCount > 7) {
				_mm_storeu_ps((float*)&ret[7] + i, _mm512_extractf32x4_ps(dhdh, 1));
				_mm_storeu_ps((float*)&ret[7] + i + 4, _mm512_extractf32x4_ps(dhdh, 3));
			}
		}
		*/
		return ret;
	}
	if constexpr (FieldCount > 8 && FieldCount <= 16) //TODO: this version is untested
	{
		for (int i = 0; i < 16; i += 4)
		{
			__m512 v0 = _mm512_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[i]));      //|abcd|efgh|ijkl|mnoq|_0/4/8/12
			__m512 v1 = _mm512_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[i + 1]));  //|abcd|efgh|ijkl|mnoq|_1/5/9/13
			__m512 v2 = _mm512_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[i + 2]));  //|abcd|efgh|ijkl|mnoq|_2/6/10/14
			__m512 v3 = _mm512_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[i + 3]));  //|abcd|efgh|ijkl|mnoq|_3/7/11/15

			__m512 tmp0 = _mm512_unpacklo_ps(v0, v1); //|a0,a1,b0,b1|e0,e1,f0,f1|i0,i1,j0,j1|m0,m1,n0,n1|
			__m512 tmp1 = _mm512_unpacklo_ps(v2, v3); //|a2,a3,b2,b3|e2,e3,f2,f3|i2,i3,j2,j3|m2,m3,n2,n3|
			__m512 tmp2 = _mm512_unpackhi_ps(v0, v1); //|c0,c1,d0,d1|g0,g1,h0,h1|k0,k1,l0,l1|o0,o1,q0,q1|
			__m512 tmp3 = _mm512_unpackhi_ps(v2, v3); //|c2,c3,d2,d3|g2,g3,h2,h3|k2,k3,l2,l3|o2,o3,q2,q3|
			__m512d aeim = _mm512_unpacklo_pd(tmp0, tmp1); //|a0,a1,a2,a3|e0,e1,e2,e3|i0,i1,i2,i3|m0,m1,m2,m3|
			__m512d bfjn = _mm512_unpackhi_pd(tmp0, tmp1); //|b0,b1,b2,b3|f0,f1,f2,f3|j0,j1,j2,j3|n0,n1,n2,n3|
			__m512d cgko = _mm512_unpacklo_pd(tmp2, tmp3); //|c0,c1,c2,c3|g0,g1,g2,g3|k0,k1,k2,k3|o0,o1,o2,o3|
			__m512d dhlq = _mm512_unpackhi_pd(tmp2, tmp3); //|d0,d1,d2,d3|h0,h1,h2,h3|l0,l1,l2,l3|q0,q1,q2,q3|

			_mm_storeu_ps((float*)&ret[0] + i, _mm512_extractf32x4_ps(aeim, 0)); //a
			_mm_storeu_ps((float*)&ret[1] + i, _mm512_extractf32x4_ps(bfjn, 0)); //b
			_mm_storeu_ps((float*)&ret[2] + i, _mm512_extractf32x4_ps(cgko, 0)); //c
			_mm_storeu_ps((float*)&ret[3] + i, _mm512_extractf32x4_ps(dhlq, 0)); //d
			_mm_storeu_ps((float*)&ret[4] + i, _mm512_extractf32x4_ps(aeim, 1)); //e
			_mm_storeu_ps((float*)&ret[5] + i, _mm512_extractf32x4_ps(bfjn, 1)); //f
			_mm_storeu_ps((float*)&ret[6] + i, _mm512_extractf32x4_ps(cgko, 1)); //g
			_mm_storeu_ps((float*)&ret[7] + i, _mm512_extractf32x4_ps(dhlq, 1)); //h
			_mm_storeu_ps((float*)&ret[8] + i, _mm512_extractf32x4_ps(aeim, 2)); //i
			if constexpr (FieldCount > 9) _mm_storeu_ps((float*)&ret[9] + i, _mm512_extractf32x4_ps(bfjn, 2)); //j
			if constexpr (FieldCount > 10) _mm_storeu_ps((float*)&ret[10] + i, _mm512_extractf32x4_ps(cgko, 2)); //k
			if constexpr (FieldCount > 11) _mm_storeu_ps((float*)&ret[11] + i, _mm512_extractf32x4_ps(dhlq, 2)); //l
			if constexpr (FieldCount > 12) _mm_storeu_ps((float*)&ret[12] + i, _mm512_extractf32x4_ps(aeim, 3)); //m
			if constexpr (FieldCount > 13) _mm_storeu_ps((float*)&ret[13] + i, _mm512_extractf32x4_ps(bfjn, 3)); //n
			if constexpr (FieldCount > 14) _mm_storeu_ps((float*)&ret[14] + i, _mm512_extractf32x4_ps(cgko, 3)); //o
			if constexpr (FieldCount > 15) _mm_storeu_ps((float*)&ret[15] + i, _mm512_extractf32x4_ps(dhlq, 3)); //q
		}
		return ret;
	}
#endif

	//if no specialized version available, then just gather as normal.
	const float* fp = (const float*)base;
	for (int i = 0; i < FieldCount; ++i)
	{
		__m256 tmp1 = _mm512_mask_i64gather_ps(_mm256_setzero_ps(), mask, offsetLo, fp + i, 1); //1 scale, since offsets are already calculated in bytes, not floats
		__m256 tmp2 = _mm512_mask_i64gather_ps(_mm256_setzero_ps(), mask >> 8, offsetHi, fp + i, 1);
		__m512 tmp = _mm512_insertf32x8(_mm512_castps256_ps512(tmp1), tmp2, 1);
		_mm512_storeu_ps(&ret[i], tmp);
	}
	return ret;
}