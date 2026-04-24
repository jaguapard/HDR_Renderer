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
class TaskHandle;
class GateHandle;
class WaitableHandle;
class GenericHandle
{
public:
	friend class Threadpool;
	friend class TaskHandle;
	friend class GateHandle;
	friend class WaitableHandle;
protected:
	uint64_t id = 0;
	Threadpool* pool = nullptr;
};
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
	GenericHandle handle;
};

class WaitableHandle
{
public:
	friend class Threadpool;

	//WaitableHandle() = delete;
	WaitableHandle(const TaskHandle& task);
	WaitableHandle(const GateHandle& gate);
	void blockUntilComplete();
	bool isComplete() const;
private:
	enum class Type
	{
		TASK,
		GATE
	};
	GenericHandle handle;
	Type type;
};

struct DependencyStore
{
	std::vector<WaitableHandle> store;
	DependencyStore() {};
	
	template<typename T> DependencyStore(const std::vector<T>& v) 
	{ 
		for (auto& it : v) store.emplace_back(it); 
	}
	template<typename T> DependencyStore& operator=(const std::vector<T>& v) 
	{ 
		store.clear(); 
		for (auto& it : v) store.emplace_back(it); 
		return *this; 
	}

	size_t size() const { return store.size(); }
	void clear() { store.clear(); }
	template<typename T> WaitableHandle& emplace_back(const T& t) { return store.emplace_back(t); }
};

struct ThreadpoolTask
{
	ThreadpoolTask() {};
	ThreadpoolTask(const taskfunc_t& func, const DependencyStore& dependencies = {}) : func(func), dependencies(dependencies) {};
	taskfunc_t func;
	DependencyStore dependencies; //Forces the task to be considered runnable only if all the input handles complete.
};
class Threadpool
{
public:
	/*
	friend class GenericHandle;
	friend class TaskHandle;
	friend class GateHandle;
	friend class WaitableHandle;*/
	Threadpool(size_t numThreads = 0);

	//Adds task batch to the thread pool. Returns handles to tasks in the same order as in tasks vector.
	std::vector<TaskHandle> addTaskBatch(const std::vector<ThreadpoolTask>& tasks);

	//Adds a single task to the threadpool and returns a handle for it
	TaskHandle addTask(const ThreadpoolTask& task);

	//Creates a new gate and returns a handle for it. The gates can be used as dependencies for tasks to prevent then from being ran before the gate opens
	GateHandle createGate();

	//Blocks the calling thread until all passed tasks are complete. Blocking is unsupported on worker threads and will throw an exception if it gets called by one.
	void blockUntilTasksComplete(const std::vector<TaskHandle>& tasks);

	void blockUntilWaitablesComplete(const std::vector<WaitableHandle>& waitables);

	void blockUntilWaitableComplete(const WaitableHandle& waitable);
	//Makes the calling thread execute other tasks until input tasks are completed. Care should be taken, since it may hang the caller for indeterminate amount of time
	//void helpWhileWaiting(const std::vector<TaskHandle>& tasks);

	bool hasTaskFinished(const TaskHandle& task);
	bool isGateOpen(const GateHandle& gate);

	//Opens the gate and signals all waiting threads.
	void openGate(const GateHandle& gate);

	//Returns true if this function is called by threadpool's worker thread
	bool isWorkerThread() const;

	size_t getWorkerCount() const;

	std::pair<double, double> getLimitsForThread(size_t threadIndex, double min, double max, std::optional<size_t> threadCount) const;

	static inline Threadpool* instance = nullptr;
private:
	std::unique_ptr<std::jthread[]> workers;
	size_t workerCount = 0;
	std::atomic<uint64_t> lastFreeHandle = 1;

	mutable std::condition_variable cv;
	mutable std::mutex cv_mtx;

	mutable std::recursive_mutex structs_mtx;
	std::unordered_map<uint64_t, ThreadpoolTask> unassignedTasks;
	std::unordered_set<uint64_t> inProgressTaskIds;
	std::unordered_set<uint64_t> closedGates;

	void workerRoutine(size_t myNumber);
	void spawnThreads(size_t threadCount);
	std::optional<std::pair<uint64_t, ThreadpoolTask>> tryPopTask(); //if there are runnable tasks, then pops one and returns it. Else returns empty optional
	void markTaskFinished(uint64_t id);
};