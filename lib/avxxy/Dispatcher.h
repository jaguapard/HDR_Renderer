#pragma once
#include "FeatureSet.h"
#include <tuple>
#include "ISAs/scalar.h"
#include "ISAs/AVX512F.h"
#include "ISAs/AVX512BW.h"
#include "ISAs/AVX512VL.h"

namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		template<typename... Args> inline constexpr bool always_false_v = false;

		template<FeatureSet FS>
		class Dispatcher
		{
		private:
			struct Dummy {};
		public:
			static inline constexpr FeatureSet FeatureSet = FS;
			using order = std::tuple<
				std::conditional_t<FS.AVX512.VL, ISA::AVX512VL<FS>, Dummy>,
				std::conditional_t<FS.AVX512.BW, ISA::AVX512BW<FS>, Dummy>,
				std::conditional_t<FS.AVX512.F, ISA::AVX512F<FS>, Dummy>, 
				ISA::Scalar<FS>>;

			//Dispatches operation through this dispatcher. Attempts to pick best available implementation for target operation respecting template argument feature set limitations
			template<typename Op, typename... Args>
			static auto run(Op op, Args&&... args)
			{
				//don't allow outsiders to poison I, that's why this run_private exists
				return run_private(op, std::forward<Args>(args)...);
			}
		private:
			template<size_t I = 0, typename Op, typename... Args>
			static auto run_private(Op op, Args&&... args)
			{
				if constexpr (I < std::tuple_size_v<order>)
				{
					using Impl = std::tuple_element_t<I, order>;
					//Search tuple for fitting implementation, and return the value returned by first valid implementation.
					//If no implementations exist, static_assert triggers
					if constexpr (requires { Impl::eval(op, std::forward<Args>(args)...); })
						return Impl::eval(op, std::forward<Args>(args)...);
					else return run_private<I + 1>(op, std::forward<Args>(args)...);
				}
				else static_assert(always_false_v<Op, Args...>, "AVXxy dispatcher: no implementation exists for operation");
			}
		};

		//Dispatcher that uses current feature set.
		using DefaultDispatcher = Dispatcher<FS_current>;
	}
}