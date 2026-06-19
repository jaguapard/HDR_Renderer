#pragma once
#include "FeatureSet.h"
#include <tuple>
#include "ISAs/scalar.h"

namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		template<typename Op, typename... Args> inline constexpr bool dependent_false_v = false;

		template<FeatureSet FS>
		class Dispatcher
		{
		private:
			struct Dummy {};
		public:
			static inline constexpr FeatureSet FeatureSet = FS;
			using order = std::tuple<ISA::Scalar>;

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
				else static_assert(dependent_false_v<Op, Args...>, "AVXxy dispatcher: no implementation exists for operation");
			}
		};

		//Dispatcher that uses current feature set.
		using DefaultDispatcher = Dispatcher<FS_current>;
	}
}