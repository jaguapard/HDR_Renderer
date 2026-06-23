#pragma once
#include "FeatureSet.h"
#include <tuple>
#include "ISAs/Scalar.h"
#include "ISAs/AVX512_F.h"
#include "ISAs/AVX512_BW.h"
#include "ISAs/AVX512_DQ.h"
#include "ISAs/AVX512_CD.h"
#include "ISAs/AVX512_VBMI.h"
#include "ISAs/AVX512_VBMI2.h"
#include "ISAs/AVX2.h"
#include "ISAs/F16C.h"
#include "meta/meta.h"
namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		class Dispatcher
		{
		private:
			struct Dummy {};
		public:
			//static inline constexpr FeatureSet FS = FS_current;
			//struct fail_ack_t {};

			using order = std::tuple<
				std::conditional_t<FS.has(Feature::AVX512_VBMI2), ISA_AVX512_VBMI2, Dummy>,
				std::conditional_t<FS.has(Feature::AVX512_VBMI), ISA_AVX512_VBMI, Dummy>,
				std::conditional_t<FS.has(Feature::AVX512_DQ), ISA_AVX512_DQ, Dummy>,
				std::conditional_t<FS.has(Feature::AVX512_BW), ISA_AVX512_BW, Dummy>,
				std::conditional_t<FS.has(Feature::AVX512_CD), ISA_AVX512_CD, Dummy>,
				std::conditional_t<FS.has(Feature::AVX512_F), ISA_AVX512_F, Dummy>,
				std::conditional_t<FS.has(Feature::AVX2), ISA_AVX2, Dummy>,
				std::conditional_t<FS.has(Feature::F16C), ISA_F16C, Dummy>,
				ISA_Scalar>;

			//Dispatches operation through this dispatcher. Attempts to pick best available implementation for target operation respecting template argument feature set limitations
			template<typename Op, typename... Args>
			static auto run(Args&&... args)
			{
				//don't allow outsiders to poison I, that's why this run_private exists
				return run_private<Op>(std::forward<Args>(args)...);
			}

		private:
			template<typename Op, size_t I = 0, typename... Args>
			static auto run_private(Args&&... args)
			{
				if constexpr (I < std::tuple_size_v<order>)
				{
					using instr_set_t = std::tuple_element_t<I, order>;
					//Search tuple for fitting implementation, and return the value returned by first valid implementation.
					//If no implementations exist, static_assert triggers
					if constexpr (requires {instr_set_t::template eval<Op>(std::forward<Args>(args)...); })
					{
						auto ret = instr_set_t::template eval<Op>(std::forward<Args>(args)...);
						//if fail_ack_t is returned, it means that implementation exists, but it all fell through to the fail_ack_t return,
						//This is considered invalid, so continue searching
						if constexpr (std::same_as<decltype(ret), fail_ack_t>) return run_private<Op, I + 1>(std::forward<Args>(args)...);
						else return ret;
					}
					else return run_private<Op, I + 1>(std::forward<Args>(args)...);
				}
				else static_assert(meta::always_false_v<Op, Args...>, "AVXxy routing: no implementation exists for operation");
			}

		};
	}
}