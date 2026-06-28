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
__forceinline std::array<ReturnType, FieldCount> aos2soa_gather_and_transpose(const void* base, i32x16 ind, mask16d mask)
	requires (sizeof(ReturnType) == 64 && FieldCount >= 1)
{
	//Unmasked elements use safe dummy index for load (first valid index is broadcasted to all lanes and replaces unmasked ones).
	//This is done to avoid branching in a loop or mask calculation frenzy for masked loads. 
	//The algorithm will break if all elements are invalid, but then the function can't do anything anyway, 
	// so instead of forcing users to sanitize the mask, we do it ourselves. Function contract already says that values are undefined, so it's OK
	std::array<ReturnType, FieldCount> ret;
	if (!mask) [[unlikely]] return ret;

	i32x16 compressedInd = compress(mask, ind);
	ind = mask_mov(i32x16(compressedInd[0]), mask, ind);

	//ind can overflow if multiplied in place, causing silent corruption of the results. Thus, extend and multiply
	//can use extra *4 for easier addressing modes. It's unlikely to matter much, but since it free, why not. 
	//Multiplication and extension is already required due to indices being struct indices, not element indices
	// and this function promises to load all 32-bit indices properly.
	u64x16 offsets = u64x16(ind) * FieldCount * 4;
	const uint64_t rawBase = (const uint64_t)(base);
	constexpr uint64_t packLoadMask = (1ull << FieldCount) - 1; //avoid touching OOB for tails. Load only FieldCount lower floats for all loads

	if constexpr (FieldCount > 2 && FieldCount <= 4)
	{
		f32x16 r[4];
		for (int i = 0; i < 4; ++i) r[i] = {
			f32x8(load<f32x4>((const void*)(rawBase + offsets[i]), packLoadMask), load<f32x4>((const void*)(rawBase + offsets[i + 4]), packLoadMask)),
			f32x8(load<f32x4>((const void*)(rawBase + offsets[i + 8]), packLoadMask), load<f32x4>((const void*)(rawBase + offsets[i + 12]), packLoadMask))
		};
		
		auto aabb01 = vcast<double>(unpacklo(r[0], r[1]));
		auto aabb23 = vcast<double>(unpacklo(r[2], r[3]));
		auto ccdd01 = vcast<double>(unpackhi(r[0], r[1]));
		auto ccdd23 = vcast<double>(unpackhi(r[2], r[3]));
		
		if constexpr (FieldCount > 0) store(unpacklo(aabb01, aabb23), &ret[0]);
		if constexpr (FieldCount > 1) store(unpackhi(aabb01, aabb23), &ret[1]);
		if constexpr (FieldCount > 2) store(unpacklo(ccdd01, ccdd23), &ret[2]);
		if constexpr (FieldCount > 3) store(unpackhi(ccdd01, ccdd23), &ret[3]);
		return ret;
	}
#if 0
	else if constexpr (FieldCount > 4 && FieldCount <= 8)
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

		if constexpr (FieldCount > 0) _mm512_storeu_ps(&ret[0], ymm_x2_to_zmm(_mm256_castpd_ps(a0_7), _mm256_castpd_ps(a8_15)));
		if constexpr (FieldCount > 1) _mm512_storeu_ps(&ret[1], ymm_x2_to_zmm(_mm256_castpd_ps(b0_7), _mm256_castpd_ps(b8_15)));
		if constexpr (FieldCount > 2) _mm512_storeu_ps(&ret[2], ymm_x2_to_zmm(_mm256_castpd_ps(c0_7), _mm256_castpd_ps(c8_15)));
		if constexpr (FieldCount > 3) _mm512_storeu_ps(&ret[3], ymm_x2_to_zmm(_mm256_castpd_ps(d0_7), _mm256_castpd_ps(d8_15)));
		if constexpr (FieldCount > 4) _mm512_storeu_ps(&ret[4], ymm_x2_to_zmm(_mm256_castpd_ps(e0_7), _mm256_castpd_ps(e8_15)));
		if constexpr (FieldCount > 5) _mm512_storeu_ps(&ret[5], ymm_x2_to_zmm(_mm256_castpd_ps(f0_7), _mm256_castpd_ps(f8_15)));
		if constexpr (FieldCount > 6) _mm512_storeu_ps(&ret[6], ymm_x2_to_zmm(_mm256_castpd_ps(g0_7), _mm256_castpd_ps(g8_15)));
		if constexpr (FieldCount > 7) _mm512_storeu_ps(&ret[7], ymm_x2_to_zmm(_mm256_castpd_ps(h0_7), _mm256_castpd_ps(h8_15)));
		return ret;
	}

	else if constexpr (FieldCount > 8 && FieldCount <= 16)
	{
		__m512 struct0 = _mm512_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[0])); //|a0,b0,c0,d0|e0,f0,g0,h0|i0,j0,k0,l0|m0,n0,o0,p0|
		__m512 struct1 = _mm512_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[1])); //|a1,b1,c1,d1|e1,f1,g1,h1|i1,j1,k1,l1|m1,n1,o1,p1|
		__m512 struct2 = _mm512_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[2])); //|a2,b2,c2,d2|e2,f2,g2,h2|i2,j2,k2,l2|m2,n2,o2,p2|
		__m512 struct3 = _mm512_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[3])); //|a3,b3,c3,d3|e3,f3,g3,h3|i3,j3,k3,l3|m3,n3,o3,p3|
		__m512 struct4 = _mm512_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[4])); //|a4,b4,c4,d4|e4,f4,g4,h4|i4,j4,k4,l4|m4,n4,o4,p4|
		__m512 struct5 = _mm512_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[5])); //|a5,b5,c5,d5|e5,f5,g5,h5|i5,j5,k5,l5|m5,n5,o5,p5|
		__m512 struct6 = _mm512_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[6])); //|a6,b6,c6,d6|e6,f6,g6,h6|i6,j6,k6,l6|m6,n6,o6,p6|
		__m512 struct7 = _mm512_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[7])); //|a7,b7,c7,d7|e7,f7,g7,h7|i7,j7,k7,l7|m7,n7,o7,p7|
		__m512 struct8 = _mm512_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[8])); //|a8,b8,c8,d8|e8,f8,g8,h8|i8,j8,k8,l8|m8,n8,o8,p8|
		__m512 struct9 = _mm512_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[9])); //|a9,b9,c9,d9|e9,f9,g9,h9|i9,j9,k9,l9|m9,n9,o9,p9|
		__m512 struct10 = _mm512_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[10])); //|a10,b10,c10,d10|e10,f10,g10,h10|i10,j10,k10,l10|m10,n10,o10,p10|
		__m512 struct11 = _mm512_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[11])); //|a11,b11,c11,d11|e11,f11,g11,h11|i11,j11,k11,l11|m11,n11,o11,p11|
		__m512 struct12 = _mm512_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[12])); //|a12,b12,c12,d12|e12,f12,g12,h12|i12,j12,k12,l12|m12,n12,o12,p12|
		__m512 struct13 = _mm512_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[13])); //|a13,b13,c13,d13|e13,f13,g13,h13|i13,j13,k13,l13|m13,n13,o13,p13|
		__m512 struct14 = _mm512_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[14])); //|a14,b14,c14,d14|e14,f14,g14,h14|i14,j14,k14,l14|m14,n14,o14,p14|
		__m512 struct15 = _mm512_maskz_loadu_ps(packLoadMask, (const void*)(rawBase + offsets[15])); //|a15,b15,c15,d15|e15,f15,g15,h15|i15,j15,k15,l15|m15,n15,o15,p15|

		//64-bit packs are brought into proper order by these unpacks
		__m512d tmp0 = _mm512_castps_pd(_mm512_unpacklo_ps(struct0, struct1)); //|a0,a1,b0,b1|e0,e1,f0,f1|i0,i1,j0,j1|m0,m1,n0,n1|
		__m512d tmp1 = _mm512_castps_pd(_mm512_unpacklo_ps(struct2, struct3)); //|a2,a3,b2,b3|e2,e3,f2,f3|i2,i3,j2,j3|m2,m3,n2,n3|
		__m512d tmp2 = _mm512_castps_pd(_mm512_unpacklo_ps(struct4, struct5)); //|a4,a5,b4,b5|e4,e5,f4,f5|i4,i5,j4,j5|m4,m5,n4,n5|
		__m512d tmp3 = _mm512_castps_pd(_mm512_unpacklo_ps(struct6, struct7)); //|a6,a7,b6,b7|e6,e7,f6,f7|i6,i7,j6,j7|m6,m7,n6,n7|
		__m512d tmp4 = _mm512_castps_pd(_mm512_unpacklo_ps(struct8, struct9)); //|a8,a9,b8,b9|e8,e9,f8,f9|i8,i9,j8,j9|m8,m9,n8,n9|
		__m512d tmp5 = _mm512_castps_pd(_mm512_unpacklo_ps(struct10, struct11)); //|a10,a11,b10,b11|e10,e11,f10,f11|i10,i11,j10,j11|m10,m11,n10,n11|
		__m512d tmp6 = _mm512_castps_pd(_mm512_unpacklo_ps(struct12, struct13)); //|a12,a13,b12,b13|e12,e13,f12,f13|i12,i13,j12,j13|m12,m13,n12,n13|
		__m512d tmp7 = _mm512_castps_pd(_mm512_unpacklo_ps(struct14, struct15)); //|a14,a15,b14,b15|e14,e15,f14,f15|i14,i15,j14,j15|m14,m15,n14,n15|
		__m512d tmp8 = _mm512_castps_pd(_mm512_unpackhi_ps(struct0, struct1)); //|c0,c1,d0,d1|g0,g1,h0,h1|k0,k1,l0,l1|o0,o1,p0,p1|
		__m512d tmp9 = _mm512_castps_pd(_mm512_unpackhi_ps(struct2, struct3)); //|c2,c3,d2,d3|g2,g3,h2,h3|k2,k3,l2,l3|o2,o3,p2,p3|
		__m512d tmp10 = _mm512_castps_pd(_mm512_unpackhi_ps(struct4, struct5)); //|c4,c5,d4,d5|g4,g5,h4,h5|k4,k5,l4,l5|o4,o5,p4,p5|
		__m512d tmp11 = _mm512_castps_pd(_mm512_unpackhi_ps(struct6, struct7)); //|c6,c7,d6,d7|g6,g7,h6,h7|k6,k7,l6,l7|o6,o7,p6,p7|
		__m512d tmp12 = _mm512_castps_pd(_mm512_unpackhi_ps(struct8, struct9)); //|c8,c9,d8,d9|g8,g9,h8,h9|k8,k9,l8,l9|o8,o9,p8,p9|
		__m512d tmp13 = _mm512_castps_pd(_mm512_unpackhi_ps(struct10, struct11)); //|c10,c11,d10,d11|g10,g11,h10,h11|k10,k11,l10,l11|o10,o11,p10,p11|
		__m512d tmp14 = _mm512_castps_pd(_mm512_unpackhi_ps(struct12, struct13)); //|c12,c13,d12,d13|g12,g13,h12,h13|k12,k13,l12,l13|o12,o13,p12,p13|
		__m512d tmp15 = _mm512_castps_pd(_mm512_unpackhi_ps(struct14, struct15)); //|c14,c15,d14,d15|g14,g15,h14,h15|k14,k15,l14,l15|o14,o15,p14,p15|

		//128-bit packs are brought into proper order by these unpacks
		__m512d xmm0 = _mm512_unpacklo_pd(tmp0, tmp1); //|a0,a1,a2,a3|e0,e1,e2,e3|i0,i1,i2,i3|m0,m1,m2,m3|
		__m512d xmm1 = _mm512_unpackhi_pd(tmp0, tmp1); //|b0,b1,b2,b3|f0,f1,f2,f3|j0,j1,j2,j3|n0,n1,n2,n3|
		__m512d xmm2 = _mm512_unpacklo_pd(tmp8, tmp9); //|c0,c1,c2,c3|g0,g1,g2,g3|k0,k1,k2,k3|o0,o1,o2,o3|
		__m512d xmm3 = _mm512_unpackhi_pd(tmp8, tmp9); //|d0,d1,d2,d3|h0,h1,h2,h3|l0,l1,l2,l3|p0,p1,p2,p3|
		__m512d xmm4 = _mm512_unpacklo_pd(tmp2, tmp3); //|a4,a5,a6,a7|e4,e5,e6,e7|i4,i5,i6,i7|m4,m5,m6,m7|
		__m512d xmm5 = _mm512_unpackhi_pd(tmp2, tmp3); //|b4,b5,b6,b7|f4,f5,f6,f7|j4,j5,j6,j7|n4,n5,n6,n7|
		__m512d xmm6 = _mm512_unpacklo_pd(tmp10, tmp11); //|c4,c5,c6,c7|g4,g5,g6,g7|k4,k5,k6,k7|o4,o5,o6,o7|
		__m512d xmm7 = _mm512_unpackhi_pd(tmp10, tmp11); //|d4,d5,d6,d7|h4,h5,h6,h7|l4,l5,l6,l7|p4,p5,p6,p7|
		__m512d xmm8 = _mm512_unpacklo_pd(tmp4, tmp5); //|a8,a9,a10,a11|e8,e9,e10,e11|i8,i9,i10,i11|m8,m9,m10,m11|
		__m512d xmm9 = _mm512_unpackhi_pd(tmp4, tmp5); //|b8,b9,b10,b11|f8,f9,f10,f11|j8,j9,j10,j11|n8,n9,n10,n11|
		__m512d xmm10 = _mm512_unpacklo_pd(tmp12, tmp13); //|c8,c9,c10,c11|g8,g9,g10,g11|k8,k9,k10,k11|o8,o9,o10,o11|
		__m512d xmm11 = _mm512_unpackhi_pd(tmp12, tmp13); //|d8,d9,d10,d11|h8,h9,h10,h11|l8,l9,l10,l11|p8,p9,p10,p11|
		__m512d xmm12 = _mm512_unpacklo_pd(tmp6, tmp7); //|a12,a13,a14,a15|e12,e13,e14,e15|i12,i13,i14,i15|m12,m13,m14,m15|
		__m512d xmm13 = _mm512_unpackhi_pd(tmp6, tmp7); //|b12,b13,b14,b15|f12,f13,f14,f15|j12,j13,j14,j15|n12,n13,n14,n15|
		__m512d xmm14 = _mm512_unpacklo_pd(tmp14, tmp15); //|c12,c13,c14,c15|g12,g13,g14,g15|k12,k13,k14,k15|o12,o13,o14,o15|
		__m512d xmm15 = _mm512_unpackhi_pd(tmp14, tmp15); //|d12,d13,d14,d15|h12,h13,h14,h15|l12,l13,l14,l15|p12,p13,p14,p15|

		__m512d ymm0 = _mm512_shuffle_f64x2(xmm0, xmm4, _MM_SHUFFLE(2, 0, 2, 0)); //|a0_3|i0_3|a4_7|i4_7|
		__m512d ymm1 = _mm512_shuffle_f64x2(xmm8, xmm12, _MM_SHUFFLE(2, 0, 2, 0)); //|a8_11|i8_11|a12_15|i12_15|
		if constexpr (FieldCount > 0) _mm512_storeu_pd(&ret[0], _mm512_shuffle_f64x2(ymm0, ymm1, _MM_SHUFFLE(2, 0, 2, 0))); //final A
		if constexpr (FieldCount > 8) _mm512_storeu_pd(&ret[8], _mm512_shuffle_f64x2(ymm0, ymm1, _MM_SHUFFLE(3, 1, 3, 1))); //final I

		ymm0 = _mm512_shuffle_f64x2(xmm1, xmm5, _MM_SHUFFLE(2, 0, 2, 0)); //|b0_3|j0_3|b4_7|j4_7|
		ymm1 = _mm512_shuffle_f64x2(xmm9, xmm13, _MM_SHUFFLE(2, 0, 2, 0)); //|b8_11|j8_11|b12_15|j12_15|
		if constexpr (FieldCount > 1) _mm512_storeu_pd(&ret[1], _mm512_shuffle_f64x2(ymm0, ymm1, _MM_SHUFFLE(2, 0, 2, 0))); //final B
		if constexpr (FieldCount > 9) _mm512_storeu_pd(&ret[9], _mm512_shuffle_f64x2(ymm0, ymm1, _MM_SHUFFLE(3, 1, 3, 1))); //final J

		ymm0 = _mm512_shuffle_f64x2(xmm2, xmm6, _MM_SHUFFLE(2, 0, 2, 0)); //|c0_3|k0_3|c4_7|k4_7|
		ymm1 = _mm512_shuffle_f64x2(xmm10, xmm14, _MM_SHUFFLE(2, 0, 2, 0)); //|c8_11|k8_11|c12_15|k12_15|
		if constexpr (FieldCount > 2) _mm512_storeu_pd(&ret[2], _mm512_shuffle_f64x2(ymm0, ymm1, _MM_SHUFFLE(2, 0, 2, 0))); //final C
		if constexpr (FieldCount > 10) _mm512_storeu_pd(&ret[10], _mm512_shuffle_f64x2(ymm0, ymm1, _MM_SHUFFLE(3, 1, 3, 1))); //final K

		ymm0 = _mm512_shuffle_f64x2(xmm3, xmm7, _MM_SHUFFLE(2, 0, 2, 0)); //|d0_3|l0_3|d4_7|l4_7|
		ymm1 = _mm512_shuffle_f64x2(xmm11, xmm15, _MM_SHUFFLE(2, 0, 2, 0)); //|d8_11|l8_11|d12_15|l12_15|
		if constexpr (FieldCount > 3) _mm512_storeu_pd(&ret[3], _mm512_shuffle_f64x2(ymm0, ymm1, _MM_SHUFFLE(2, 0, 2, 0))); //final D
		if constexpr (FieldCount > 11) _mm512_storeu_pd(&ret[11], _mm512_shuffle_f64x2(ymm0, ymm1, _MM_SHUFFLE(3, 1, 3, 1))); //final L


		ymm0 = _mm512_shuffle_f64x2(xmm0, xmm4, _MM_SHUFFLE(3, 1, 3, 1)); //|e0_3|m0_3|e4_7|m4_7|
		ymm1 = _mm512_shuffle_f64x2(xmm8, xmm12, _MM_SHUFFLE(3, 1, 3, 1)); //|e8_11|m8_11|e12_15|m12_15|
		if (FieldCount > 4) _mm512_storeu_pd(&ret[4], _mm512_shuffle_f64x2(ymm0, ymm1, _MM_SHUFFLE(2, 0, 2, 0))); //final E
		if (FieldCount > 12) _mm512_storeu_pd(&ret[12], _mm512_shuffle_f64x2(ymm0, ymm1, _MM_SHUFFLE(3, 1, 3, 1))); //final M

		ymm0 = _mm512_shuffle_f64x2(xmm1, xmm5, _MM_SHUFFLE(3, 1, 3, 1)); //|f0_3|n0_3|f4_7|n4_7|
		ymm1 = _mm512_shuffle_f64x2(xmm9, xmm13, _MM_SHUFFLE(3, 1, 3, 1)); //|f8_11|n8_11|f12_15|n12_15|
		if (FieldCount > 5) _mm512_storeu_pd(&ret[5], _mm512_shuffle_f64x2(ymm0, ymm1, _MM_SHUFFLE(2, 0, 2, 0))); //final F
		if (FieldCount > 13) _mm512_storeu_pd(&ret[13], _mm512_shuffle_f64x2(ymm0, ymm1, _MM_SHUFFLE(3, 1, 3, 1))); //final N

		ymm0 = _mm512_shuffle_f64x2(xmm2, xmm6, _MM_SHUFFLE(3, 1, 3, 1)); //|g0_3|o0_3|g4_7|o4_7|
		ymm1 = _mm512_shuffle_f64x2(xmm10, xmm14, _MM_SHUFFLE(3, 1, 3, 1)); //|g8_11|o8_11|g12_15|o12_15|
		if (FieldCount > 6) _mm512_storeu_pd(&ret[6], _mm512_shuffle_f64x2(ymm0, ymm1, _MM_SHUFFLE(2, 0, 2, 0))); //final G
		if (FieldCount > 14) _mm512_storeu_pd(&ret[14], _mm512_shuffle_f64x2(ymm0, ymm1, _MM_SHUFFLE(3, 1, 3, 1))); //final O

		ymm0 = _mm512_shuffle_f64x2(xmm3, xmm7, _MM_SHUFFLE(3, 1, 3, 1)); //|h0_3|p0_3|h4_7|h4_7|
		ymm1 = _mm512_shuffle_f64x2(xmm11, xmm15, _MM_SHUFFLE(3, 1, 3, 1)); //|h8_11|p8_11|p12_15|p12_15|
		if (FieldCount > 7) _mm512_storeu_pd(&ret[7], _mm512_shuffle_f64x2(ymm0, ymm1, _MM_SHUFFLE(2, 0, 2, 0))); //final H
		if (FieldCount > 15) _mm512_storeu_pd(&ret[15], _mm512_shuffle_f64x2(ymm0, ymm1, _MM_SHUFFLE(3, 1, 3, 1))); //final P
		return ret;
	}
#endif
	//if no specialized version available, then just gather as normal.
	const float* p = (const float*)base;
	for (size_t i = 0; i < FieldCount; ++i)
	{
		auto v = gather<float, 16, FieldCount * 4>(p + i, i32x16(ind), mask);
		store(v, &ret[i]);
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