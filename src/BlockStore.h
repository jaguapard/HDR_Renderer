#pragma once
#include <vector>
#include <memory>

// Stores elements in fixed-size blocks, allocated as needed.
// For each added element, pointers are valid until the store is cleared or reset.
// Logical size is tracked separately from allocated storage.
// Items can only be removed from the logical store via reset() or clear().
// Underlying memory will persist until clear() or trim() is called.
template <typename ElementType, size_t N>
class BlockStore
{
public:
	static inline constexpr size_t ELEMENTS_PER_BLOCK = N;
	//Appends element to the end of the store and increases store's logical size by 1.
	ElementType& append(const ElementType& el)
	{
		auto& ret = this->reserve_slot() = el;
		return ret;
	}

	//Accessing indices greater than or equal to store's logical size is undefined behavior.
	ElementType& operator[](size_t i)
	{
		assert(i < this->size());
		return this->blocks[i / ELEMENTS_PER_BLOCK][i % ELEMENTS_PER_BLOCK];
	}

	//Returns a reference to memory location that an element can safely be written into and increases store's logical size by 1. 
	//The returned reference may contain stale data.
	//The caller is responsible for initialization.
	ElementType& reserve_slot()
	{
		size_t blockIndex = this->elementCount / ELEMENTS_PER_BLOCK;
		size_t elementIndex = this->elementCount % ELEMENTS_PER_BLOCK;
		while (blockIndex >= this->blocks.size()) this->blocks.emplace_back(std::make_unique<ElementType[]>(ELEMENTS_PER_BLOCK));
		++this->elementCount;
		return this->blocks[blockIndex][elementIndex];
	}

	//Sets store's logical size to zero without freeing any allocated blocks.
	void reset()
	{
		this->elementCount = 0;
	}

	//Sets store's logical size to zero and frees all allocated blocks
	void clear()
	{
		this->reset();
		this->blocks.clear();
	}

	//Returns logical size of the store
	size_t size() const 
	{
		return this->elementCount;
	}

	//Returns true if store's logical size is 0
	bool empty() const
	{
		return this->size() == 0;
	}

	//Frees empty blocks
	void trim()
	{
		size_t sz = this->size();
		if (sz == 0) return this->clear();
		size_t occupiedBlockCount = sz / ELEMENTS_PER_BLOCK;
		if (sz % ELEMENTS_PER_BLOCK != 0) ++occupiedBlockCount;
		this->blocks.resize(occupiedBlockCount);
	}
private:
	size_t elementCount = 0;
	std::vector<std::unique_ptr<ElementType[]>> blocks;
};

//unlike BlockStore, the BoundedBlockStore has a maximum size cap at MAX_BLOCKS * ELEMENTS_PER_BLOCK elements. In return, it is guaranteed to never reallocate any items after they are added
/*
template<typename ElementType, size_t ELEMENTS_PER_BLOCK, size_t MAX_BLOCKS>
class BoundedBlockStore
{
public:
	
private:
	std::array<std::unique_ptr<ElementType[]>, MAX_BLOCKS> blocks = { nullptr };
	size_t elementCount = 0;
};*/