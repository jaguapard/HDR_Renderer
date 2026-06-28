#pragma once
#include "namespace.h"

namespace AVXXY_NAMESPACE
{
	namespace settings
	{
		//If set to true, all uses of scalar fallback will dump info about the operation and it's call site into stdout
		//This is usually not useful for the users, but if you are sure that absoultely no scalar fallbacks must be used, you can enable it as sanity check
		//However, it is mostly meant for library debugging and can cause severe performance penalty if scalar fallbacks are part of usual workflow.
		//Default value: false
		static constexpr bool NOISY_SCALAR = false;

		//TODO: this setting can't be properly implemented with current reinterpreting engine.
		//AVX512 narrowing conversion intrinsics zero-fill the destination register if their output vector is not filled fully by the operation result.
		//For instance, _mm512_cvtepi64_epi8 takes a 512-bit vector and outputs 128-bit vector. However, only lower 64-bits are occupied with the converted bytes, others are filled with zero.
		//If this is set to true, emulations will replicate this behavior with slight performance penalty.
		//However, if such behavior is not required and you are not going to intentionally rely on it by unsafely casting vectors after conversions, you can set it to false
		//Default value: true
		//static constexpr bool NARROWING_CVT_ZERO_FILL = false;


		//When using unsafe operations (vreinterpret_us, SIMD_Vector::from_bits_us) and output type is larger than input, the output's upper bits are undefined.
		//With this set to true, they will be forced to zero with some performance penalty. Usually, this does not matter if you're not relying on this exact behavior
		//Default value: false
		static constexpr bool ZERO_FILL_UNSAFES = false;

		//If set to false, unsafe operations will not be available (vreinterpret_us, SIMD_Vector::from_bits_us)
		//Operations relying on it will use zero-filling if they have size mismatch
		static constexpr bool ALLOW_UNSAFE_OPERATIONS = true;

		//If this value is true, vpopcnt function will allow passing floating point types to it.
		//The returned vectors will still be integral.
		//Default value: true
		static constexpr bool ALLOW_VPOPCNT_FOR_NON_INTS = true;

		//The conflict detection operation requires output elements to be large enough to hold at least N bits,
		//where N is the number of lanes in input vector.
		//If this setting is true, conflict detection is allowed to return vectors larger than inputs
		//static constexpr bool ALLOW_LARGER_CONFLICTD_RETURN = false;

		//The floating point comparisons do not adhere to bitwise equal == compare returns equal
		//Conflict detection operation compares bitwise representations of input elements
		//Thus, a conflict detection operation may return unexpected results for floating point results
		//However, this can only happen for values of NaN, infinities and denormals.
		//Enabling this setting will allow conflict detection operation to be used on floating point types with bitwise comparisons
		//Default value: false
		//static constexpr bool ALLOW_FLOATING_POINT_CONFLICTD = false;

		//If set to true, allows SIMD_Vector<type, 1> to exist. 
		//Default value: true
		//static constexpr bool ALLOW_SINGLE_ELEMENT_VECTORS = true;

		//If set to true, forces all SIMD_Vector to have sizes of 16, 32, 64 or integer multiples of 64 bytes
		//Default value: false
		//static constexpr bool FORCE_INTRINSIC_SIZE_MATCH_FOR_VECTORS = false;

		//FP16 operations without native support (AVX512-FP16) are emulated via convert to FP32 + do operation + convert back to FP16
		//Repeated conversions introduce performance penalty and less precision in calculations
		//This setting governs wheter or not to skip back-conversion when returning the results of the mathematical operations
		//However, if FP16 conversion is required (i.e. assigning result to FP16 variable), the conversion will still be performed
		//This does not affect data-type-agnostic operations (data movement)
		//Default value: true		
		//TODO: this may cause major headaches on false, make wrapper class for FP32-in-FP16 clothing?
		static constexpr bool FP16_EMULATIONS_RETURN_FP16_ONLY = true;


		//static_assert(!NARROWING_CVT_ZERO_FILL, "Narrowing cvt zero-fill emulation is not currently supported.");
		static_assert(FP16_EMULATIONS_RETURN_FP16_ONLY, "FP16-in-FP32 proxy not yet supported");
		static_assert(!NOISY_SCALAR, "Noisy scalar flag is currently not supported");
		static_assert(!ZERO_FILL_UNSAFES, "Zero fill unsafes flag is currently not supported");
		static_assert(ALLOW_UNSAFE_OPERATIONS, "Disallowing unsafe operations is currently not supported");
	}
}