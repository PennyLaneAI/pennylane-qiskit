#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>

/** @brief This class implements a simplified thread-safe queue allowing
 * concurrent access from multiple threads. This follows the implementation of
 * `threadsafe_queue` from C++: Concurrency in Action Ch 6 by A. Williams.
 */
template <typename T> class TSQueue {
  private:
    mutable std::mutex m;
    std::queue<T> q;
    std::condition_variable cond;

  public:
    TSQueue() = default;

    /**
     * @brief Push data to the queue. Thread-safe.
     *
     * @param data
     */
    void push(T data) {
        std::lock_guard<std::mutex> lk(m);
        q.push(std::move(data));
        cond.notify_one();
    }

    /**
     * @brief Pop element from queue. Thread-safe (blocking).
     *
     * @param data Data reference to overwrite with value.
     */
    void wait_and_pop(T &data) {
        std::unique_lock<std::mutex> lk(m);
        cond.wait(lk, [this] { return !q.empty(); });
        data = std::move(q.front());
        q.pop();
    }
    /**
     * @brief Pop element from queue. Thread-safe (blocking).
     *
     * @return std::shared_ptr<T> Shared-pointer to popped data.
     */
    std::shared_ptr<T> wait_and_pop() {
        std::unique_lock<std::mutex> lk(m);
        cond.wait(lk, [this] { return !q.empty(); });
        std::shared_ptr<T> res(std::make_shared<T>(std::move(q.front())));
        q.pop();
        return res;
    }
    /**
     * @brief Pop element from queue if available. Thread-safe.
     *
     * @param data Data reference to overwrite.
     * @return true Element successfully popped.
     * @return false Element failed to pop.
     */
    bool try_pop(T &data) {
        std::lock_guard<std::mutex> lk(m);
        if (q.empty()) {
            return false;
        }
        data = std::move(q.front());
        q.pop();
        return true;
    }
    /**
     * @brief Pop element from queue. Thread safe.
     *
     * @return std::shared_ptr<T>
     */
    std::shared_ptr<T> try_pop() {
        std::lock_guard<std::mutex> lk(m);
        if (q.empty()) {
            return std::shared_ptr<T>();
        }
        std::shared_ptr<T> res(std::make_shared<T>(std::move(q.front())));
        q.pop();
        return res;
    }
    /**
     * @brief Check if queue is empty.
     *
     * @return true
     * @return false
     */
    bool empty() const {
        std::lock_guard<std::mutex> lk(m);
        return q.empty();
    }
};