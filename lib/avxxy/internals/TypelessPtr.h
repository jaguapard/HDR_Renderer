#pragma once
#include "../namespace.h"
#include <stdexcept>
#include <span>
namespace AVXXY_NAMESPACE
{
	namespace internals
	{
		template<bool konst = false>
		//Represents an address in memory. Convertible to and constructible from any pointer type
		class __TypelessPtrBase
		{
		private:
			size_t addr;
		public:
			__TypelessPtrBase() {};

			__TypelessPtrBase(void* p)
				requires (!konst)
			{
				this->addr = size_t(p);
			}

			__TypelessPtrBase(const void* p)
				requires (konst)
			{
				this->addr = size_t(p);
			}

			template<typename P>
			requires (!konst)
			operator P* () const
			{
				return (P*)(this->addr);
			}
			template<typename P>
			requires (konst)
			operator const P* () const
			{
				return (const P*)(this->addr);
			}

			//Adds in bytes!
			__TypelessPtrBase<konst> operator+(size_t n) const
			{
				__TypelessPtrBase ret;
				ret.addr = addr + n;
				return ret;
			}

			//Adds in bytes!
			__TypelessPtrBase<konst>& operator+=(size_t n)
			{
				*this = *this + n;
				return *this;
			}
			//Subracts in bytes!
			__TypelessPtrBase<konst> operator-(size_t n) const
			{
				__TypelessPtrBase ret;
				ret.addr = addr - n;
				return ret;
			}
			//Subracts in bytes!
			__TypelessPtrBase<konst>& operator-=(size_t n)
			{
				*this = *this - n;
				return *this;
			}

			//Returns this pointer aligned to boundary. The result's address is not greater than this pointer's one.
			//Does not change address if pointer is already aligned
			__TypelessPtrBase alignDec(size_t alignmentInBytes) const
			{
				if (addr % alignmentInBytes == 0) return *this;
				__TypelessPtrBase ret;
				ret.addr = this->addr - this->addr % alignmentInBytes;
				return ret;
			}
			//Returns this pointer aligned to boundary. The result's address is not less than this pointer's one.
			//Does not change address if pointer is already aligned
			__TypelessPtrBase alignInc(size_t alignmentInBytes) const
			{
				if (addr % alignmentInBytes == 0) return *this;
				__TypelessPtrBase ret = alignDec(alignmentInBytes);
				ret.addr += alignmentInBytes;
				return ret;
			}

			//Returns pointer aligned to alignmentRequirementInBytes boundary.
			//If resultant memory address is less than the beginning of the span, advances it by one alignment requirement forward
			//If after that the pointer is out of span's bounds, throws an exception. This can only happen if span content size is smaller than alignment requirement
			template<typename T>
			__TypelessPtrBase alignBounded(size_t alignmentInBytes, const std::span<T>& bounds) const
			{
				size_t boundLo = size_t(bounds.data());
				size_t boundHi = size_t(&bounds.back()) + sizeof(T);

				__TypelessPtrBase ret = this->alignDec(alignmentInBytes);
				if (ret.addr < boundLo) ret.addr += alignmentInBytes;
				if (ret.addr >= boundHi) throw std::runtime_error("Can't align pointer to bounds: too small bounds!");
				return ret;
			}
		};

		//Represents non-const typeless pointer, can be constructed from and casted to any non-const pointer
		//Making this const will not prevent you from casting to non-const raw pointers, use ConstTypessPtr for that
		typedef __TypelessPtrBase<false> TypelessPtr;
		//Represents const typeless pointer, can be constructed from and casted to any const pointer.
		typedef __TypelessPtrBase<true> ConstTypelessPtr;
	}
}