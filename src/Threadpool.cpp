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
		threadCount = std::max(1u, std::thread::hardware_concurrency() - 1);
	}
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

std::optional<std::pair<uint64_t, ThreadpoolTask>> Threadpool::tryPopTask()
{
	std::lock_guard lck(this->structs_mtx);
	for (auto& it : this->unassignedTasks)
	{
		size_t dependenciesSatisfied = 0;
		for (auto& dep : it.second.dependencies)
		{
			if (dep.isComplete()) ++dependenciesSatisfied;
			else break;
		}
		if (dependenciesSatisfied == it.second.dependencies.size())
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
	std::lock_guard lck(this->structs_mtx);
	assert(this->inProgressTaskIds.find(id) != this->inProgressTaskIds.end());
	this->inProgressTaskIds.erase(id);
}

TaskHandle Threadpool::addTask(const ThreadpoolTask& task)
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

GateHandle Threadpool::createGate()
{
	GateHandle ret;
	ret.handle.pool = this;
	ret.handle.id = this->lastFreeHandle++;

	std::lock_guard lck(this->structs_mtx);
	this->closedGates.insert(ret.handle.id);
	return ret;
}

void Threadpool::blockUntilTasksComplete(const std::vector<TaskHandle>& tasks)
{
	for (auto& it : tasks)
	{
		this->blockUntilWaitableComplete(WaitableHandle(it));
	}
}

void Threadpool::blockUntilWaitablesComplete(const std::vector<WaitableHandle>& waitables)
{
	for (auto& it : waitables) this->blockUntilWaitableComplete(it);
}

void Threadpool::blockUntilWaitableComplete(const WaitableHandle& waitable)
{
	if (bIsWorkerThread) throw std::runtime_error("Thread pool worker attempted to block on waitable " + std::to_string(waitable.handle.id) + ". Workers blocking on thread pool's waitables is prone to deadlocks and is not supported. Define dependecies while adding a task if you want the task to start after prerequisites are complete.");
	if (waitable.handle.id >= this->lastFreeHandle) throw std::runtime_error("Attempting to wait on non-existant handle.");
	if (waitable.isComplete()) return;

	std::unique_lock cv_lck(cv_mtx);
	this->cv.wait(cv_lck, [&]() {
		return waitable.isComplete();
		});
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

std::vector<TaskHandle> Threadpool::addTaskBatch(const std::vector<ThreadpoolTask>& tasks)
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
		std::optional<std::pair<uint64_t, ThreadpoolTask>> taskForMe;
		{
			std::unique_lock cv_lck(cv_mtx);
			cv.wait(cv_lck, [&]() {
				taskForMe = this->tryPopTask();
				return taskForMe.has_value();
				});
		}

		taskForMe->second.func();
		this->markTaskFinished(taskForMe->first);

		std::unique_lock cv_lck(cv_mtx);
		cv.notify_all();
	}
}

std::pair<double, double> Threadpool::getLimitsForThread(size_t threadIndex, double min, double max, std::optional<size_t> threadCount) const
{
	size_t nThreads = threadCount ? threadCount.value() : this->getWorkerCount();
	double minLimit = lerp(min, max, double(threadIndex) / nThreads);
	double maxLimit = lerp(min, max, double(threadIndex + 1) / nThreads);
	return std::make_pair(std::clamp(minLimit, min, max), std::clamp(maxLimit, min, max));
}
WaitableHandle::WaitableHandle(const TaskHandle& task)
{
	this->handle = task.handle;
	this->type = Type::TASK;
}

WaitableHandle::WaitableHandle(const GateHandle& gate)
{
	this->handle = gate.handle;
	this->type = Type::GATE;
}

void WaitableHandle::blockUntilComplete()
{
	handle.pool->blockUntilWaitableComplete(*this);
}

bool WaitableHandle::isComplete() const
{
	switch (this->type)
	{
	case Type::TASK: { TaskHandle h; h.handle = handle; return h.hasFinished(); }
	case Type::GATE: { GateHandle h; h.handle = handle; return h.isOpen(); }
	}
}

bool TaskHandle::hasFinished() const
{
	return handle.pool->hasTaskFinished(*this);
}

void GateHandle::open()
{
	handle.pool->openGate(*this);
}

bool GateHandle::isOpen() const
{
	return handle.pool->isGateOpen(*this);
}
