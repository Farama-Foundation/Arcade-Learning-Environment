#ifndef ALE_VECTOR_ACTION_QUEUE_HPP_
#define ALE_VECTOR_ACTION_QUEUE_HPP_

#include <atomic>
#include <vector>

#ifndef MOODYCAMEL_DELETE_FUNCTION
    #define MOODYCAMEL_DELETE_FUNCTION = delete
#endif

#include "ale/external/lightweightsemaphore.h"
#include "types.hpp"

namespace ale::vector {

/// Lock-free queue for actions to be processed by worker threads.
/// Supports bulk enqueue and single dequeue.
class ActionQueue {
public:
    explicit ActionQueue(std::size_t capacity)
        : capacity_(capacity),
          queue_(capacity),
          alloc_idx_(0),
          dequeue_idx_(0),
          items_available_(0) {}

    /// Enqueue multiple actions at once. Thread-safe.
    void enqueue_bulk(const std::vector<Action>& actions) {
        std::size_t pos = alloc_idx_.fetch_add(actions.size());
        for (std::size_t i = 0; i < actions.size(); ++i) {
            queue_[(pos + i) % capacity_] = actions[i];
        }
        items_available_.signal(static_cast<int>(actions.size()));
    }

    /// Dequeue a single action. Blocks if queue is empty. Thread-safe.
    Action dequeue() {
        while (!items_available_.wait()) {}
        std::size_t idx = dequeue_idx_.fetch_add(1);
        return queue_[idx % capacity_];
    }

    /// Approximate number of items in queue
    std::size_t size_approx() const {
        return alloc_idx_.load() - dequeue_idx_.load();
    }

private:
    std::size_t capacity_;
    std::vector<Action> queue_;
    std::atomic<std::size_t> alloc_idx_;
    std::atomic<std::size_t> dequeue_idx_;
    moodycamel::LightweightSemaphore items_available_;
};

}  // namespace ale::vector

#endif  // ALE_VECTOR_ACTION_QUEUE_HPP_
