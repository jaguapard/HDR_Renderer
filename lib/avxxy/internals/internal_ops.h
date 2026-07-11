#pragma once
#include "../settings.h"
#include <type_traits>
#include <bit>

namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		template<typename To, typename From>
		requires (std::is_trivially_copyable_v<From> && std::is_trivially_copyable_v<To>)
		To avxxy_bit_cast(const From& a)
		{
			if constexpr (sizeof(From) == sizeof(To)) return std::bit_cast<To>(a);
			else
			{
				To ret;
				if constexpr (settings::ZERO_FILL_UNSAFES) memset(&ret, 0, sizeof(ret));
				memcpy(&ret, &a, std::min(sizeof(ret), sizeof(a)));
				return ret;
			}
		}
	}
}
