#include "Threadpool.h"
#include <string>
#include <assert.h>
#include "helpers.h"

static thread_local bool bIsWorkerThread = false;
Threadpool::Threadpool(size_t numThreads)
{
	if (instance) throw std::runtime_error("Second threadpool attempted to be constructed while only one is allowed.");
	instance = this;
	size_t threadCount = numThreads;
	if (!numThreads)
	{
		threadCount = std::max<int64_t>(1, int64_t(std::thread::hardware_concurrency()) - 1); //avoid crowding out the main thread unless that's impossible. Also, hardware_concurrency() can return 0
	}
	if (SINGLE_THREAD_MODE) threadCount = 1;
	//threadCount = 24;
	this->spawnThreads(threadCount);
}

void Threadpool::spawnThreads(size_t threadCount)
{
	this->workerCount = threadCount;
	this->workers = std::make_unique<std::jthread[]>(workerCount);
	for (size_t i = 0; i < workerCount; ++i)
		this->workers[i] = std::jthread(&Threadpool::workerRoutine, this, i);
}

std::optional<std::pair<uint64_t, Threadpool::Task>> Threadpool::tryPopTask()
{
	std::lock_guard lck(this->structs_mtx);
	for (auto& it : this->unassignedTasks)
	{
		size_t dependenciesSatisfied = 0;
		for (auto& dep : it.second.dependencies.store)
		{
			if (dep.isComplete()) ++dependenciesSatisfied;
			else break;
		}
		if (dependenciesSatisfied == it.second.dependencies.store.size())
		{
			auto ret = it;
			this->unassignedTasks.erase(it.first);
			this->inProgressTaskIds.insert(ret.first);
			return ret;
		}
	}
	return std::nullopt;
}

void Threadpool::markTaskFinished(uint64_t id)
{
	{
		std::lock_guard lck(this->structs_mtx);
		assert(this->inProgressTaskIds.find(id) != this->inProgressTaskIds.end());
		this->inProgressTaskIds.erase(id);
	}
	std::lock_guard lck(cv_mtx);
	cv.notify_all();
}

Threadpool::TaskHandle Threadpool::addTask(const Threadpool::Task& task)
{
	uint64_t id = this->lastFreeHandle++;
	{
		std::lock_guard lck(this->structs_mtx);
		this->unassignedTasks[id] = task;
	}
	TaskHandle h;
	h.handle.pool = this;
	h.handle.id = id;

	std::lock_guard lck(this->cv_mtx);
	cv.notify_one();
	return h;
}

Threadpool::GateHandle Threadpool::createGate()
{
	GateHandle ret;
	ret.handle.pool = this;
	ret.handle.id = this->lastFreeHandle++;

	std::lock_guard lck(this->structs_mtx);
	this->closedGates.insert(ret.handle.id);
	return ret;
}

void Threadpool::blockUntilComplete(const WaitableHandle& handle)
{
	if (bIsWorkerThread) throw std::runtime_error("Thread pool worker attempted to block on waitable " + std::to_string(handle.handle.id) + ". Workers blocking on thread pool's waitables is prone to deadlocks and is not supported. Define dependecies while adding a task if you want the task to start after prerequisites are complete.");
	if (handle.handle.id >= this->lastFreeHandle) throw std::runtime_error("Attempting to wait on non-existant handle.");
	if (handle.isComplete()) return;

	std::unique_lock cv_lck(cv_mtx);
	this->cv.wait(cv_lck, [&]() {
		return handle.isComplete();
		});
}

void Threadpool::blockUntilComplete(const WaitableCollection& collection)
{
	for (auto& it : collection) this->blockUntilComplete(it);
}

void Threadpool::helpWhileWaiting(const WaitableCollection& collection)
{
	for (auto& it : collection) this->helpWhileWaiting(it);
}

void Threadpool::helpWhileWaiting(const WaitableHandle& handle)
{
	std::optional<std::pair<uint64_t, Threadpool::Task>> backloggedTask;
	while (true)
	{
		if (backloggedTask)
		{
			backloggedTask->second.func();
			this->markTaskFinished(backloggedTask->first);
			backloggedTask = std::nullopt;
		}
		if (handle.isComplete()) break;
		else
		{
			auto popped = this->tryPopTask();
			if (popped) //if there's a job - take it
			{
				backloggedTask = popped;
			}
			else //if not - wait until either one is available or task we're waiting for is finished
			{
				std::unique_lock cv_lck(this->cv_mtx);
				cv.wait(cv_lck, [&]() {
					backloggedTask = this->tryPopTask();
					return backloggedTask.has_value() || handle.isComplete();
					});
			}
		}
	}
}

bool Threadpool::hasTaskFinished(const TaskHandle& task)
{
	std::lock_guard lck(this->structs_mtx);
	assert(task.handle.id < this->lastFreeHandle);
	return unassignedTasks.find(task.handle.id) == unassignedTasks.end()
		&& inProgressTaskIds.find(task.handle.id) == inProgressTaskIds.end();
}

bool Threadpool::isGateOpen(const GateHandle& gate)
{
	std::lock_guard lck(this->structs_mtx);
	assert(gate.handle.id < lastFreeHandle);
	return this->closedGates.find(gate.handle.id) == this->closedGates.end();
}

void Threadpool::openGate(const GateHandle& gate)
{
	{
		std::lock_guard lck(this->structs_mtx);
		assert(gate.handle.id < lastFreeHandle);
		assert(this->closedGates.find(gate.handle.id) != this->closedGates.end());
		this->closedGates.erase(gate.handle.id);
	}
	std::lock_guard lck(this->cv_mtx);
	this->cv.notify_all();
}

bool Threadpool::isWorkerThread() const
{
	return bIsWorkerThread;
}

size_t Threadpool::getWorkerCount() const
{
	return this->workerCount;
}

size_t Threadpool::getThreadCount() const
{
	return this->getWorkerCount();
}

std::vector<Threadpool::TaskHandle> Threadpool::addTaskBatch(const std::vector<Threadpool::Task>& tasks)
{
	size_t sz = tasks.size();
	uint64_t id = this->lastFreeHandle.fetch_add(sz);
	std::vector<TaskHandle> ret(sz);

	{
		std::lock_guard lck(this->structs_mtx);
		for (size_t i = 0; i < sz; ++i)
		{
			size_t idd = id + i;
			this->unassignedTasks[idd] = tasks[i];
			ret[i].handle.id = idd;
			ret[i].handle.pool = this;
		}
	}

	std::lock_guard lck(this->cv_mtx);
	if (tasks.size() == 1) cv.notify_one();
	else cv.notify_all();
	return ret;
}

void Threadpool::workerRoutine(size_t myNumber)
{
	bIsWorkerThread = true;
	while (true)
	{
		std::optional<std::pair<uint64_t, Threadpool::Task>> taskForMe;
		{
			std::unique_lock cv_lck(cv_mtx);
			cv.wait(cv_lck, [&]() {
				taskForMe = this->tryPopTask();
				return taskForMe.has_value();
				});
		}

		taskForMe->second.func();
		this->markTaskFinished(taskForMe->first);
	}
}

std::pair<double, double> Threadpool::getLimitsForThread(size_t threadIndex, double min, double max, size_t threadCount) const
{
	assert(min <= max);
	size_t nThreads = threadCount ? threadCount : this->getWorkerCount();
	double minLimit = lerp(min, max, double(threadIndex) / nThreads);
	double maxLimit = lerp(min, max, double(threadIndex + 1) / nThreads);
	return std::make_pair(std::clamp(minLimit, min, max), std::clamp(maxLimit, min, max));
}
std::vector<std::pair<uint64_t, uint64_t>> Threadpool::getNonOverlappingPartitions(uint64_t min, uint64_t max, uint64_t count) const
{
	count = count ? count : this->getWorkerCount();
	assert(max >= min);
	assert(count > 0);
	std::vector<std::pair<uint64_t, uint64_t>> ret;
	uint64_t remaining = max - min, partStart = min;
	ret.reserve(count);
	for (uint64_t i = 0; i < count; ++i)
	{
		uint64_t part = remaining / (count - i);
		ret.push_back(std::make_pair(partStart, partStart + part));
		partStart += part;
		remaining -= part;
	}
	return ret;
}
Threadpool::WaitableHandle::WaitableHandle(const TaskHandle& task)
{
	this->handle = task.handle;
	this->type = Type::TASK;
}

Threadpool::WaitableHandle::WaitableHandle(const GateHandle& gate)
{
	this->handle = gate.handle;
	this->type = Type::GATE;
}

void Threadpool::WaitableHandle::blockUntilComplete()
{
	handle.pool->blockUntilComplete(*this);
}

bool Threadpool::WaitableHandle::isComplete() const
{
	switch (this->type)
	{
	case Type::TASK: { TaskHandle h; h.handle = handle; return h.hasFinished(); }
	case Type::GATE: { GateHandle h; h.handle = handle; return h.isOpen(); }
	}
}

bool Threadpool::TaskHandle::hasFinished() const
{
	return handle.pool->hasTaskFinished(*this);
}

void Threadpool::GateHandle::open()
{
	handle.pool->openGate(*this);
}

bool Threadpool::GateHandle::isOpen() const
{
	return handle.pool->isGateOpen(*this);
}
