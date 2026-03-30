#pragma once
#include <cassert>

namespace bob
{
	template <typename ValT, typename MaskT, typename ElemT>
	//do NOT set try to set the members of this struct, it is internal and only used for syntactic sugar via constructor magic.
	//It is very easy to leave it in inconsistent state. Make sure to look at the assembly output if performance is critical,
	//as other classes rely heavily on aggressive optimization to choose proper code paths without building this struct at all
	//(during normal usage this struct should be very short-lived and completely optimized away by the compiler)
	struct AVX512_MaskedOp
	{
		enum class Mode
		{
			UNCONDITIONAL, //unconditional operation: perform operation for all elements
			CONDITIONAL, //pass through original register: if mask bit is 1, perform operation, else passes through old contents
			MERGE_MASKING, //merge with other register: if mask bit is 1, perform operation, else pass through src contents
			ZERO_MASKING, //zero-masking: if mask bit is 1, perform operation, else zero out the element
			BLEND, //blends the two input operands based on mask, then unconditionally performs operation, useful for ternary-like operations
		};

		const MaskT mask = 0xFFFF; //mask for masked operations, unused in unmasked operations
		const ValT src; //operand for merge-masking mode only, else unused
		const ValT a; //mandatory operand, used as right hand side for operations
		const ValT b; //mandatory operand in blend mode, else unused
		const Mode mode = Mode::UNCONDITIONAL; //automatically filled based on which overloaded contructor is used

		AVX512_MaskedOp(const ValT& a) : a(a), mode(Mode::UNCONDITIONAL) { assertCorrectness(); }; //dst = x[i] OP a[i]
		AVX512_MaskedOp(const ValT& a, MaskT mask) : a(a), mask(mask), mode(Mode::CONDITIONAL) { assertCorrectness(); }; //dst = mask[i] ? (x[i] OP a[i]) : x[i]
		AVX512_MaskedOp(const ValT& a, MaskT mask, const ValT& b) : a(a), b(b), mask(mask), mode(Mode::BLEND) { assertCorrectness(); }; //dst = mask[i] ? (x[i] OP a[i]) : (x[i] OP b[i])
		AVX512_MaskedOp(MaskT mask, const ValT& a) : a(a), mask(mask), mode(Mode::ZERO_MASKING) { assertCorrectness(); }; //dst = mask[i] ? (x[i] OP a[i]) : 0
		AVX512_MaskedOp(MaskT mask, const ValT& a, const ValT& src) : a(a), src(src), mask(mask), mode(Mode::MERGE_MASKING) { assertCorrectness(); }; //dst = mask[i] ? (x[i] OP a[i]) : src[i]

		AVX512_MaskedOp(const ElemT& a) : a(ValT(a)), mode(Mode::UNCONDITIONAL) { assertCorrectness(); }; //dst = x[i] OP a[i]
		AVX512_MaskedOp(const ElemT& a, MaskT mask) : a(ValT(a)), mask(mask), mode(Mode::CONDITIONAL) { assertCorrectness(); }; //dst = mask[i] ? (x[i] OP a[i]) : x[i]
		AVX512_MaskedOp(const ElemT& a, MaskT mask, const ElemT& b) : a(ValT(a)), b(ValT(b)), mask(mask), mode(Mode::BLEND) { assertCorrectness(); }; //dst = mask[i] ? (x[i] OP a[i]) : (x[i] OP b[i])
		AVX512_MaskedOp(MaskT mask, const ElemT& a) : a(ValT(a)), mask(mask), mode(Mode::ZERO_MASKING) { assertCorrectness(); }; //dst = mask[i] ? (x[i] OP a[i]) : 0
		AVX512_MaskedOp(MaskT mask, const ElemT& a, const ElemT& src) : a(ValT(a)), src(VelT(src)), mask(mask), mode(Mode::MERGE_MASKING) { assertCorrectness(); }; //dst = mask[i] ? (x[i] OP a[i]) : src[i]

		void assertCorrectness() const
		{
			/*
			switch (mode)
			{
			case Mode::UNCONDITIONAL:
			case Mode::CONDITIONAL:
			case Mode::ZERO_MASKING:
				assert(a && !b && !src);
				break;
			case Mode::BLEND:
				assert(a && b && !src);
				break;
			case Mode::MERGE_MASKING:
				assert(a && src && !b);
				break;
			default:
				break;
			}*/
		}
	};
}