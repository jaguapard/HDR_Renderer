#pragma once
#include <functional>
#include <unordered_map>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <unordered_set>
#include <atomic>
#include <optional>
#include <set>
#include <variant>

typedef std::function<void()> taskfunc_t;

class Threadpool;

class Threadpool
{
private:
	class GenericHandle
	{
	public:
		GenericHandle() = default;
		uint64_t id = 0;
		Threadpool* pool = nullptr;
	};
public:
	class TaskHandle
	{
	public:
		friend class Threadpool;
		friend class WaitableHandle;

		//TaskHandle() = delete;
		bool hasStarted() const;
		bool hasFinished() const;
		bool isInProgress() const;
	private:
		static inline constexpr bool isThreadpoolHandleType = true;
		GenericHandle handle;
	};

	class GateHandle
	{
	public:
		friend class Threadpool;
		friend class WaitableHandle;
		//GateHandle() = delete;
		void open();
		bool isOpen() const;
	private:
		static inline constexpr bool isThreadpoolHandleType = true;
		GenericHandle handle;
	};

	class WaitableHandle
	{
	public:
		friend class Threadpool;

		//WaitableHandle() = delete;
		WaitableHandle() = default;
		WaitableHandle(const TaskHandle& task);
		WaitableHandle(const GateHandle& gate);
		void blockUntilComplete();
		bool isComplete() const;
	private:
		static inline constexpr bool isThreadpoolHandleType = true;
		enum class Type
		{
			TASK,
			GATE
		};
		GenericHandle handle;
		Type type;
	};

	struct WaitableCollection
	{
		std::vector<WaitableHandle> store;
		WaitableCollection() {};

		template<typename T> WaitableCollection(const std::vector<T>& v)
		{
			static_assert(T::isThreadpoolHandleType, "Waitable collection can only be constructed from Threadpool handle types");
			for (auto& it : v) store.emplace_back(it);
		}
		template<typename T> WaitableCollection& operator=(const std::vector<T>& v)
		{
			static_assert(T::isThreadpoolHandleType, "Waitable collection can only be assigned to Threadpool handle types");
			store.clear();
			for (auto& it : v) store.emplace_back(it);
			return *this;
		}

		auto begin() { return store.begin(); }
		auto end() { return store.end(); }
		const auto begin() const { return store.begin(); }
		const auto end() const { return store.end(); }

		size_t size() const { return store.size(); }
		void clear() { store.clear(); }
		template<typename T> WaitableHandle& emplace_back(const T& t) { 
			static_assert(T::isThreadpoolHandleType, "Waitable collection can only be empaced to by Threadpool handle types"); 
			return store.emplace_back(t); }
		template<typename T> void push_back(const T& t) { return store.push_back(WaitableHandle(t)); }
	};

	struct Task
	{
		Task() {};
		Task(const taskfunc_t& func, const WaitableCollection& dependencies = {}) : func(func), dependencies(dependencies) {};
		taskfunc_t func;
		WaitableCollection dependencies; //Forces the task to be considered runnable only if all the input handles complete.
	};
public:
#ifdef NDEBUG
	static constexpr bool SINGLE_THREAD_MODE = false;
#else
	static constexpr bool SINGLE_THREAD_MODE = true;
#endif
	Threadpool(size_t numThreads = 0);

	//Adds task batch to the thread pool. Returns handles to tasks in the same order as in tasks vector.
	std::vector<TaskHandle> addTaskBatch(const std::vector<Task>& tasks);

	//Adds a single task to the threadpool and returns a handle for it
	TaskHandle addTask(const Threadpool::Task& task);

	//Creates a new gate and returns a handle for it. The gates can be used as dependencies for tasks to prevent then from being ran before the gate opens
	GateHandle createGate();

	//Blocks the calling thread until waitable handle signals completion.  Blocking is unsupported on worker threads and will throw an exception if it gets called by one. Use helpWhileWaiting instead
	void blockUntilComplete(const WaitableHandle& handle);

	//Blocks the calling thread until all waitable handles have signaled completion. Blocking is unsupported on worker threads and will throw an exception if it gets called by one. Use helpWhileWaiting instead
	void blockUntilComplete(const WaitableCollection& collection);

	//Makes the calling thread execute other tasks until input task is completed. Care should be taken, since it may hang the caller for indeterminate amount of time
	void helpWhileWaiting(const WaitableHandle& handle);

	//Makes the calling thread execute other tasks until input tasks are completed. Care should be taken, since it may hang the caller for indeterminate amount of time.
	void helpWhileWaiting(const WaitableCollection& collection);

	bool hasTaskFinished(const TaskHandle& task);
	bool isGateOpen(const GateHandle& gate);

	//Opens the gate and signals all waiting threads.
	void openGate(const GateHandle& gate);

	//Returns true if this function is called by threadpool's worker thread
	bool isWorkerThread() const;

	//Returns the worker thread count
	size_t getWorkerCount() const;

	//Returns the worker thread count
	size_t getThreadCount() const;

	//Returns evenly-split limits of quantity for given thread count. If count is 0, the threadpool's worker count is used instead.
	std::pair<double, double> getLimitsForThread(size_t threadIndex, double min, double max, size_t threadCount = 0) const;

	//Returns count ranges as partitions of a given integer range min (inclusive) to max (excusive). If count is 0, the threadpool's worker count is used instead.
	//Partitions cover entire given range with no gaps or overlaps. All subranges returned differ at most by 1 in their size.
	//First element of each returned pair signifies range's start (inclusive). Second element of each returned pair signifies range's end (exclusive)
	//The results are valid only if max >= min. Some returned intervals may be empty (pair.first == pair.second) if (max - min) < count
	std::vector<std::pair<uint64_t, uint64_t>> getNonOverlappingPartitions(uint64_t min, uint64_t max, uint64_t count = 0) const;
	static inline Threadpool* instance = nullptr;
private:
	std::unique_ptr<std::jthread[]> workers;
	size_t workerCount = 0;
	std::atomic<uint64_t> lastFreeHandle = 1;

	mutable std::condition_variable cv;
	mutable std::mutex cv_mtx;

	mutable std::recursive_mutex structs_mtx;
	std::unordered_map<uint64_t, Task> unassignedTasks;
	std::unordered_set<uint64_t> inProgressTaskIds;
	std::unordered_set<uint64_t> closedGates;

	void workerRoutine(size_t myNumber);
	void spawnThreads(size_t threadCount);
	std::optional<std::pair<uint64_t, Task>> tryPopTask(); //if there are runnable tasks, then pops one and returns it. Else returns empty optional
	void markTaskFinished(uint64_t id);
};