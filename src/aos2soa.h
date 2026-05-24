#pragma once
#include "helpers.h"

/**
@brief Loads up to 16 packs of FieldCount contigious 4-byte elements and trasposes them to SoA layout.
@details
Pseudocode:
for i in [0,15]:
	if mask[i]:
		tmp = load contigious (FieldCount*4) bytes starting at byte uint64_t(base) + ind[i]*(FieldCount*4)
		for j in [0, FieldCount-1]:
			ret[j][i*4..i*4+3] = tmp[j*4..j*4+3]

@param base: the pointer to the start of collection that indices are defined relative to
@param ind: 32-bit indices of the packs to be gathered and transposed
@param mask: mask with bits set for active packs. Inactive indices are not read from memory and have undefined values
@returns Array of FieldCount 512-bit vectors, with values transposed to SoA layout
*/
template<typename ReturnType, uint32_t FieldCount>
__forceinline std::array<ReturnType, FieldCount> aos2soa_gather_and_transpose(const void* base, __m512i ind, __mmask16 mask)
	requires (sizeof(ReturnType) == 64 && FieldCount >= 1)
{
	//Unmasked elements use safe dummy index for load (first valid index is broadcasted to all lanes and replaces unmasked ones).
	//This is done to avoid branching in a loop or mask calculation frenzy for masked loads. 
	//The algorithm will break if all elements are invalid, but then the function can't do anything anyway, 
	// so instead of forcing users to sanitize the mask, we do it ourselves. Function contract already says that values are undefined, so it's OK
	std::array<ReturnType, FieldCount> ret;
	if (!mask) [[unlikely]] return ret;
	__m512i compressedInd = _mm512_maskz_compress_epi32(mask, ind);
	ind = _mm512_mask_mov_epi32(_mm512_broadcastd_epi32(_mm512_castsi512_si128(compressedInd)), mask, ind);

	//ind can overflow if multiplied in place, causing silent corruption of the results. Thus, extend and multiply
	uint64_t offsets[16];
	const uint64_t rawBase = (const uint64_t)(base);
	__m512i indLo = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(ind, 0));
	__m512i indHi = _mm512_cvtepi32_epi64(_mm512_extracti32x8_epi32(ind, 1));
	//can use extra *4 for easier addressing modes. It's unlikely to matter much, but since it free, why not. 
	//Multiplication and extension is already required due to indices being struct indices, not element indices
	// and this function promises to load all 32-bit indices properly.
	__m512i offsetLo = _mm512_mullo_epi64(indLo, _mm512_set1_epi64(FieldCount * 4));
	__m512i offsetHi = _mm512_mullo_epi64(indHi, _mm512_set1_epi64(FieldCount * 4));
	_mm512_storeu_si512(&offsets[0], offsetLo);
	_mm512_storeu_si512(&offsets[8], offsetHi);
	constexpr uint64_t packLoadMask = (1ull << FieldCount) - 1; //avoid touching OOB for tails. Load only FieldCount lower floats for all loads

	if constexpr (FieldCount > 2 && FieldCount <= 4)
	{
		//r0 = abcd0,abcd4,abcd8,abcd12
		__m512 r0 = xmm_x4_to_zmm( 
			_mm_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[0])),
			_mm_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[4])),
			_mm_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[8])),
			_mm_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[12]))
		);
		//r1 = abcd1,abcd5,abcd9,abcd13
		__m512 r1 = xmm_x4_to_zmm(
			_mm_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[1])),
			_mm_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[5])),
			_mm_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[9])),
			_mm_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[13]))
		);
		//r2 = abcd2,abcd6,abcd10,abcd14
		__m512 r2 = xmm_x4_to_zmm(
			_mm_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[2])),
			_mm_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[6])),
			_mm_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[10])),
			_mm_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[14]))
		);
		//r3 = abcd3,abcd7,abcd11,abcd15
		__m512 r3 = xmm_x4_to_zmm(
			_mm_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[3])),
			_mm_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[7])),
			_mm_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[11])),
			_mm_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[15]))
		);

		__m512 aabb01 = _mm512_unpacklo_ps(r0, r1);
		__m512 aabb23 = _mm512_unpacklo_ps(r2, r3);
		__m512 ccdd01 = _mm512_unpackhi_ps(r0, r1);
		__m512 ccdd23 = _mm512_unpackhi_ps(r2, r3);
		if constexpr (FieldCount > 0) _mm512_storeu_pd(&ret[0], _mm512_unpacklo_pd(_mm512_castps_pd(aabb01), _mm512_castps_pd(aabb23)));
		if constexpr (FieldCount > 1) _mm512_storeu_pd(&ret[1], _mm512_unpackhi_pd(_mm512_castps_pd(aabb01), _mm512_castps_pd(aabb23)));
		if constexpr (FieldCount > 2) _mm512_storeu_pd(&ret[2], _mm512_unpacklo_pd(_mm512_castps_pd(ccdd01), _mm512_castps_pd(ccdd23)));
		if constexpr (FieldCount > 3) _mm512_storeu_pd(&ret[3], _mm512_unpackhi_pd(_mm512_castps_pd(ccdd01), _mm512_castps_pd(ccdd23)));
		return ret;
	}

	if constexpr (FieldCount > 4 && FieldCount <= 8)
	{
		__m256 struct0 = _mm256_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[0]));
		__m256 struct1 = _mm256_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[1]));
		__m256 struct2 = _mm256_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[2]));
		__m256 struct3 = _mm256_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[3]));
		__m256 struct4 = _mm256_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[4]));
		__m256 struct5 = _mm256_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[5]));
		__m256 struct6 = _mm256_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[6]));
		__m256 struct7 = _mm256_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[7]));
		__m256 struct8 = _mm256_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[8]));
		__m256 struct9 = _mm256_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[9]));
		__m256 struct10 = _mm256_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[10]));
		__m256 struct11 = _mm256_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[11]));
		__m256 struct12 = _mm256_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[12]));
		__m256 struct13 = _mm256_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[13]));
		__m256 struct14 = _mm256_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[14]));
		__m256 struct15 = _mm256_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[15]));
		
		//64-bit packs are brought into proper order by these unpacks
		__m256d tmp0 = _mm256_castps_pd(_mm256_unpacklo_ps(struct0, struct1)); //|a0,a1,b0,b1|e0,e1,f0,f1|
		__m256d tmp1 = _mm256_castps_pd(_mm256_unpacklo_ps(struct2, struct3)); //|a2,a3,b2,b3|e2,e3,f2,f3|
		__m256d tmp2 = _mm256_castps_pd(_mm256_unpacklo_ps(struct4, struct5)); //|a4,a5,b4,b5|e4,e5,f4,f5|
		__m256d tmp3 = _mm256_castps_pd(_mm256_unpacklo_ps(struct6, struct7)); //|a6,a7,b6,b7|e6,e7,f6,f7|
		__m256d tmp4 = _mm256_castps_pd(_mm256_unpacklo_ps(struct8, struct9)); //|a8,a9,b8,b9|e8,e9,f8,f9|
		__m256d tmp5 = _mm256_castps_pd(_mm256_unpacklo_ps(struct10, struct11)); //|a10,a11,b10,b11|e10,e11,f10,f11|
		__m256d tmp6 = _mm256_castps_pd(_mm256_unpacklo_ps(struct12, struct13)); //|a12,a13,b12,b13|e12,e13,f12,f13|
		__m256d tmp7 = _mm256_castps_pd(_mm256_unpacklo_ps(struct14, struct15)); //|a14,a15,b14,b15|e14,e15,f14,f15|
		__m256d tmp8 = _mm256_castps_pd(_mm256_unpackhi_ps(struct0, struct1)); //|c0,c1,d0,d1|g0,g1,h0,h1|
		__m256d tmp9 = _mm256_castps_pd(_mm256_unpackhi_ps(struct2, struct3)); //|c2,c3,d2,d3|g2,g3,h2,h3|
		__m256d tmp10 = _mm256_castps_pd(_mm256_unpackhi_ps(struct4, struct5)); //|c4,c5,d4,d5|g4,g5,h4,h5|
		__m256d tmp11 = _mm256_castps_pd(_mm256_unpackhi_ps(struct6, struct7)); //|c6,c7,d6,d7|g6,g7,h6,h7|
		__m256d tmp12 = _mm256_castps_pd(_mm256_unpackhi_ps(struct8, struct9)); //|c8,c9,d8,d9|g8,g9,h8,h9|
		__m256d tmp13 = _mm256_castps_pd(_mm256_unpackhi_ps(struct10, struct11)); //|c10,c11,d10,d11|g10,g11,h10,h11|
		__m256d tmp14 = _mm256_castps_pd(_mm256_unpackhi_ps(struct12, struct13)); //|c12,c13,d12,d13|g12,g13,h12,h13|
		__m256d tmp15 = _mm256_castps_pd(_mm256_unpackhi_ps(struct14, struct15)); //|c14,c15,d14,d15|g14,g15,h14,h15|

		//128-bit packs are brought into proper order by these unpacks
		__m256d xmm0 = _mm256_unpacklo_pd(tmp0, tmp1); //|a0,a1,a2,a3|e0,e1,e2,e3|
		__m256d xmm1 = _mm256_unpackhi_pd(tmp0, tmp1); //|b0,b1,b2,b3|f0,f1,f2,f3|
		__m256d xmm2 = _mm256_unpacklo_pd(tmp2, tmp3); //|a4,a5,a6,a7|e4,e5,e6,e7|
		__m256d xmm3 = _mm256_unpackhi_pd(tmp2, tmp3); //|b4,b5,b6,b7|f4,f5,f6,f7|
		__m256d xmm4 = _mm256_unpacklo_pd(tmp4, tmp5); //|a8,a9,a10,a11|e8,e9,e10,e11|
		__m256d xmm5 = _mm256_unpackhi_pd(tmp4, tmp5); //|b8,b9,b10,b11|f8,f9,f10,f11|
		__m256d xmm6 = _mm256_unpacklo_pd(tmp6, tmp7); //|a12,a13,a14,a15|e12,e13,e14,e15|
		__m256d xmm7 = _mm256_unpackhi_pd(tmp6, tmp7); //|b12,b13,b14,b15|f12,f13,f14,f15|
		__m256d xmm8 = _mm256_unpacklo_pd(tmp8, tmp9); //|c0,c1,c2,c3|g0,g1,g2,g3|
		__m256d xmm9 = _mm256_unpackhi_pd(tmp8, tmp9); //|d0,d1,d2,d3|h0,h1,h2,h3|
		__m256d xmm10 = _mm256_unpacklo_pd(tmp10, tmp11); //|c4,c5,c6,c7|g4,g5,g6,g7|
		__m256d xmm11 = _mm256_unpackhi_pd(tmp10, tmp11); //|d4,d5,d6,d7|h4,h5,h6,h7|
		__m256d xmm12 = _mm256_unpacklo_pd(tmp12, tmp13); //|c8,c9,c10,c11|g8,g9,g10,g11|
		__m256d xmm13 = _mm256_unpackhi_pd(tmp12, tmp13); //|d8,d9,d10,d11|h8,h9,h10,h11|
		__m256d xmm14 = _mm256_unpacklo_pd(tmp14, tmp15); //|c12,c13,c14,c15|g12,g13,g14,g15|
		__m256d xmm15 = _mm256_unpackhi_pd(tmp14, tmp15); //|d12,d13,d14,d15|h12,h13,h14,h15|

		//256-bit packs are brought into proper order by these unpacks
		__m256d a0_7 = _mm256_insertf128_pd(xmm0, _mm256_castpd256_pd128(xmm2), 1);
		__m256d b0_7 = _mm256_insertf128_pd(xmm1, _mm256_castpd256_pd128(xmm3), 1);
		__m256d c0_7 = _mm256_insertf128_pd(xmm8, _mm256_castpd256_pd128(xmm10), 1);
		__m256d d0_7 = _mm256_insertf128_pd(xmm9, _mm256_castpd256_pd128(xmm11), 1);
		__m256d e0_7 = _mm256_permute2f128_pd(xmm0, xmm2, 1 | (3 << 4));
		__m256d f0_7 = _mm256_permute2f128_pd(xmm1, xmm3, 1 | (3 << 4));
		__m256d g0_7 = _mm256_permute2f128_pd(xmm8, xmm10, 1 | (3 << 4));
		__m256d h0_7 = _mm256_permute2f128_pd(xmm9, xmm11, 1 | (3 << 4));
		__m256d a8_15 = _mm256_insertf128_pd(xmm4, _mm256_castpd256_pd128(xmm6), 1);
		__m256d b8_15 = _mm256_insertf128_pd(xmm5, _mm256_castpd256_pd128(xmm7), 1);
		__m256d c8_15 = _mm256_insertf128_pd(xmm12, _mm256_castpd256_pd128(xmm14), 1);
		__m256d d8_15 = _mm256_insertf128_pd(xmm13, _mm256_castpd256_pd128(xmm15), 1);
		__m256d e8_15 = _mm256_permute2f128_pd(xmm4, xmm6, 1 | (3 << 4));
		__m256d f8_15 = _mm256_permute2f128_pd(xmm5, xmm7, 1 | (3 << 4));
		__m256d g8_15 = _mm256_permute2f128_pd(xmm12, xmm14, 1 | (3 << 4));
		__m256d h8_15 = _mm256_permute2f128_pd(xmm13, xmm15, 1 | (3 << 4));

		if constexpr (FieldCount > 0) _mm512_storeu_pd(&ret[0], ymm_x2_to_zmm(a0_7, a8_15));
		if constexpr (FieldCount > 1) _mm512_storeu_pd(&ret[1], ymm_x2_to_zmm(b0_7, b8_15));
		if constexpr (FieldCount > 2) _mm512_storeu_pd(&ret[2], ymm_x2_to_zmm(c0_7, c8_15));
		if constexpr (FieldCount > 3) _mm512_storeu_pd(&ret[3], ymm_x2_to_zmm(d0_7, d8_15));
		if constexpr (FieldCount > 4) _mm512_storeu_pd(&ret[4], ymm_x2_to_zmm(e0_7, e8_15));
		if constexpr (FieldCount > 5) _mm512_storeu_pd(&ret[5], ymm_x2_to_zmm(f0_7, f8_15));
		if constexpr (FieldCount > 6) _mm512_storeu_pd(&ret[6], ymm_x2_to_zmm(g0_7, g8_15));
		if constexpr (FieldCount > 7) _mm512_storeu_pd(&ret[7], ymm_x2_to_zmm(h0_7, h8_15));
		return ret;
	}
#ifdef VS_CLANG //Specialized versions below rely very heavily on Clang optimizations, and if they fail, that will become horrible garbage code, probably slower than gathers anyway.
	if constexpr (FieldCount > 8 && FieldCount <= 16) //TODO: this version is untested
	{
		for (int i = 0; i < 16; i += 4)
		{
			__m512 v0 = _mm512_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[i]));      //|abcd|efgh|ijkl|mnop|_0/4/8/12
			__m512 v1 = _mm512_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[i + 1]));  //|abcd|efgh|ijkl|mnop|_1/5/9/13
			__m512 v2 = _mm512_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[i + 2]));  //|abcd|efgh|ijkl|mnop|_2/6/10/14
			__m512 v3 = _mm512_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[i + 3]));  //|abcd|efgh|ijkl|mnop|_3/7/11/15

			__m512d tmp0 = _mm512_castps_pd(_mm512_unpacklo_ps(v0, v1)); //|a0,a1,b0,b1|e0,e1,f0,f1|i0,i1,j0,j1|m0,m1,n0,n1|
			__m512d tmp1 = _mm512_castps_pd(_mm512_unpacklo_ps(v2, v3)); //|a2,a3,b2,b3|e2,e3,f2,f3|i2,i3,j2,j3|m2,m3,n2,n3|
			__m512d tmp2 = _mm512_castps_pd(_mm512_unpackhi_ps(v0, v1)); //|c0,c1,d0,d1|g0,g1,h0,h1|k0,k1,l0,l1|o0,o1,p0,p1|
			__m512d tmp3 = _mm512_castps_pd(_mm512_unpackhi_ps(v2, v3)); //|c2,c3,d2,d3|g2,g3,h2,h3|k2,k3,l2,l3|o2,o3,p2,p3|
			__m512 aeim = _mm512_castpd_ps(_mm512_unpacklo_pd(tmp0, tmp1)); //|a0,a1,a2,a3|e0,e1,e2,e3|i0,i1,i2,i3|m0,m1,m2,m3|
			__m512 bfjn = _mm512_castpd_ps(_mm512_unpackhi_pd(tmp0, tmp1)); //|b0,b1,b2,b3|f0,f1,f2,f3|j0,j1,j2,j3|n0,n1,n2,n3|
			__m512 cgko = _mm512_castpd_ps(_mm512_unpacklo_pd(tmp2, tmp3)); //|c0,c1,c2,c3|g0,g1,g2,g3|k0,k1,k2,k3|o0,o1,o2,o3|
			__m512 dhlp = _mm512_castpd_ps(_mm512_unpackhi_pd(tmp2, tmp3)); //|d0,d1,d2,d3|h0,h1,h2,h3|l0,l1,l2,l3|p0,p1,p2,p3|

			if constexpr (FieldCount > 0) _mm_storeu_ps((float*)&ret[0] + i, _mm512_extractf32x4_ps(aeim, 0)); //a
			if constexpr (FieldCount > 1) _mm_storeu_ps((float*)&ret[1] + i, _mm512_extractf32x4_ps(bfjn, 0)); //b
			if constexpr (FieldCount > 2) _mm_storeu_ps((float*)&ret[2] + i, _mm512_extractf32x4_ps(cgko, 0)); //c
			if constexpr (FieldCount > 3) _mm_storeu_ps((float*)&ret[3] + i, _mm512_extractf32x4_ps(dhlp, 0)); //d
			if constexpr (FieldCount > 4) _mm_storeu_ps((float*)&ret[4] + i, _mm512_extractf32x4_ps(aeim, 1)); //e
			if constexpr (FieldCount > 5) _mm_storeu_ps((float*)&ret[5] + i, _mm512_extractf32x4_ps(bfjn, 1)); //f
			if constexpr (FieldCount > 6) _mm_storeu_ps((float*)&ret[6] + i, _mm512_extractf32x4_ps(cgko, 1)); //g
			if constexpr (FieldCount > 7) _mm_storeu_ps((float*)&ret[7] + i, _mm512_extractf32x4_ps(dhlp, 1)); //h
			if constexpr (FieldCount > 8) _mm_storeu_ps((float*)&ret[8] + i, _mm512_extractf32x4_ps(aeim, 2)); //i
			if constexpr (FieldCount > 9) _mm_storeu_ps((float*)&ret[9] + i, _mm512_extractf32x4_ps(bfjn, 2)); //j
			if constexpr (FieldCount > 10) _mm_storeu_ps((float*)&ret[10] + i, _mm512_extractf32x4_ps(cgko, 2)); //k
			if constexpr (FieldCount > 11) _mm_storeu_ps((float*)&ret[11] + i, _mm512_extractf32x4_ps(dhlp, 2)); //l
			if constexpr (FieldCount > 12) _mm_storeu_ps((float*)&ret[12] + i, _mm512_extractf32x4_ps(aeim, 3)); //m
			if constexpr (FieldCount > 13) _mm_storeu_ps((float*)&ret[13] + i, _mm512_extractf32x4_ps(bfjn, 3)); //n
			if constexpr (FieldCount > 14) _mm_storeu_ps((float*)&ret[14] + i, _mm512_extractf32x4_ps(cgko, 3)); //o
			if constexpr (FieldCount > 15) _mm_storeu_ps((float*)&ret[15] + i, _mm512_extractf32x4_ps(dhlp, 3)); //p
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
 * @brief Gathers AoS-encoded structures, transposes them into SoA layout, and writes the result to the output buffer.
 *
 * Each source structure consists of `fieldCount` consecutive `DataType` elements.
 * The function gathers `structCount` structures from `input` using indices from `indices`,
 * converts the data from Array-of-Structures (AoS) layout into Structure-of-Arrays (SoA) layout,
 * and stores the transposed result in `output`.
 *
 * If `maskBits` is provided, structures with a cleared mask bit are not gathered.
 * Instead, their output fields are filled from `fallbackData` if supplied, or with
 * value-initialized `DataType{}` otherwise.
 *
 * Output layout:
 *   output[fieldIndex * structCount + structIndex]
 *
 * @tparam DataType  Element type of structure fields.
 * @tparam IndexType Integral type used for gather indices.
 *
 * @param input
 *   Pointer to the source AoS data buffer that indices are defined relative to.
 *   Interpreted as an array of `DataType`.
 *   Must be least `fieldCount * max(indices for structs with mask bits set) + fieldCount` elements large.
 *
 * @param indices
 *   Pointer to an array of structure indices to gather.
 *   Interpreted as an array of `IndexType` with `structCount` `DataType` elements.
 *   Must be at least `structCount` elements large
 *
 * @param output
 *   Pointer to the destination SoA buffer.
 *   Interpreted as an array of `DataType`.
 *   Must have space to hold at least `fieldCount * structCount` `DataType` elements.
 *
 * @param fieldCount
 *   Number of fields in each source structure.
 *
 * @param structCount
 *   Number of structures to gather and transpose.
 *
 * @param maskBits
 *   Optional pointer to a bit mask array containing at least
 *   `ceil(structCount / 8)` bytes.
 *   Bit `i` controls whether structure `i` is gathered:
 *     - set bit   -> gather from input
 *     - cleared bit -> use fallback/default value
 *   If null, all structures are gathered unconditionally (same as having all mask bits set).
 *
 * @param fallbackData
 *   Optional pointer to `fieldCount` fallback field values.
 *   Used for masked-out structures.
 *   If null, masked-out fields are filled with `DataType{}`.
 */
template<typename DataType, typename IndexType>
__forceinline void aos2soa_gather_and_transpose_dwords_generic(const void* input, const void* indices, void* output, size_t fieldCount, size_t structCount, const void* maskBits = nullptr, const void* fallbackData = nullptr)
{
	static_assert(std::is_integral_v<IndexType>, "aos2soa_gather_and_transpose requires integral index type");
	const uint8_t* maskBytes = (const uint8_t*)maskBits;
	const DataType* inp = (const DataType*)input;
	const IndexType* ind = (const IndexType*)indices;
	const DataType* fbck = (const DataType*)fallbackData;
	DataType* out = (DataType*)output;
	for (size_t structIndex = 0; structIndex < structCount; ++structIndex)
	{
		bool structIsMaskedOut = false;
		if (maskBits)
		{
			uint8_t andmsk = maskBytes[structIndex / 8] & (1 << (structIndex % 8));
			if (!andmsk)
			{
				structIsMaskedOut = true;
			}
		}

		for (size_t fieldIndex = 0; fieldIndex < fieldCount; ++fieldIndex)
		{
			out[fieldIndex * structCount + structIndex] = structIsMaskedOut ? (fbck ? fbck[fieldIndex] : DataType()) : inp[ind[structIndex] * fieldCount + fieldIndex];
		}
	}
}