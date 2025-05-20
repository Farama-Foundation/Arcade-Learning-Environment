#ifndef ALE_VECTOR_UTILS_HPP_
#define ALE_VECTOR_UTILS_HPP_

#include <vector>
#include <atomic>
#include <mutex>
#include <memory>

#ifndef MOODYCAMEL_DELETE_FUNCTION
    #define MOODYCAMEL_DELETE_FUNCTION = delete
#endif

#include "ale/external/lightweightsemaphore.h"

namespace ale::vector {

    /**
     * ActionSlice represents a single action or command to be processed by a worker thread
     */
    struct ActionSlice {
        int env_id;        // ID of the environment to apply the action to
        int order;         // Order in the batch for synchronous operation (-1 for async)
        bool force_reset;  // Whether to force a reset of the environment
    };

    /**
     * EnvironmentAction represents an action to be taken in an environment
     */
    struct EnvironmentAction {
        int env_id;            // ID of the environment to apply the action to
        int action_id;         // ID of the action to take
        float paddle_strength; // Strength for paddle-based games (default: 1.0)
    };

    /**
     * Timestep represents the output from an environment step
     */
    struct Timestep {
        int env_id;                       // ID of the environment this observation is from
        std::vector<uint8_t> observation; // Screen pixel data
        float reward;                     // Reward received in this step
        bool terminated;                  // Whether the game ended
        bool truncated;                   // Whether the episode was truncated due to a time limit
        int lives;                        // Remaining lives in the game
        int frame_number;                 // Frame number since the beginning of the game
        int episode_frame_number;         // Frame number since the beginning of the episode
    };

    /**
     * Observation format enumeration
     */
    enum class ObsFormat {
        Grayscale,  // Single channel grayscale observations
        RGB         // Three channel RGB observations
    };

    /**
     * Lock-free queue for actions to be processed by worker threads
     */
    class ActionBufferQueue {
    public:
        explicit ActionBufferQueue(const std::size_t num_envs)
            : alloc_ptr_(0),
              done_ptr_(0),
              queue_size_(num_envs * 2),
              queue_(queue_size_),
              sem_(0),
              sem_enqueue_(1),
              sem_dequeue_(1) {}

        /**
         * Enqueue multiple actions at once
         */
        void enqueue_bulk(const std::vector<ActionSlice>& actions) {
            while (!sem_enqueue_.wait()) {}

            const uint64_t pos = alloc_ptr_.fetch_add(actions.size());
            for (std::size_t i = 0; i < actions.size(); ++i) {
                queue_[(pos + i) % queue_size_] = actions[i];
            }

            sem_.signal(actions.size());
            sem_enqueue_.signal(1);
        }

        /**
         * Dequeue a single action
         */
        ActionSlice dequeue() {
            while (!sem_.wait()) {}
            while (!sem_dequeue_.wait()) {}

            const auto ptr = done_ptr_.fetch_add(1);
            const auto ret = queue_[ptr % queue_size_];

            sem_dequeue_.signal(1);
            return ret;
        }

        /**
         * Get the approximate size of the queue
         */
        std::size_t size_approx() const {
            return alloc_ptr_ - done_ptr_;
        }

    private:
        std::atomic<uint64_t> alloc_ptr_;  // Pointer to next allocation position
        std::atomic<uint64_t> done_ptr_;   // Pointer to next dequeue position
        std::size_t queue_size_;           // Size of the queue
        std::vector<ActionSlice> queue_;   // The actual queue data
        moodycamel::LightweightSemaphore sem_;           // Semaphore for queue access
        moodycamel::LightweightSemaphore sem_enqueue_;   // Semaphore for enqueue operations
        moodycamel::LightweightSemaphore sem_dequeue_;   // Semaphore for dequeue operations
    };

    /**
     * StateBufferQueue handles the collection of timesteps from environments
     */
    class StateBufferQueue {
    public:
        StateBufferQueue(const std::size_t batch_size, const std::size_t num_envs)
            : batch_size_(batch_size),
              num_buffers_((num_envs / batch_size + 2) * 2),
              current_buffer_(0),
              timesteps_(num_buffers_),
              buffer_count_(num_buffers_, 0),
              buffer_filled_(num_buffers_, false),
              ready_sem_(0) {

            // Initialize the timesteps vectors
            for (auto& ts : timesteps_) {
                ts.reserve(batch_size_);
            }
        }

        /**
     * Write a timestep to the buffer
     */
        void write(const Timestep& timestep, const int order = -1) {
            std::unique_lock lock(mutex_);

            // Determine which buffer to write to
            const size_t buffer_idx = current_buffer_;

            // If using ordered timesteps (order >= 0), place in the correct position
            if (order >= 0) {
                // Ensure we have enough space
                if (timesteps_[buffer_idx].size() <= static_cast<size_t>(order)) {
                    timesteps_[buffer_idx].resize(order + 1);
                }
                // Place at the specific order position
                timesteps_[buffer_idx][order] = timestep;
            } else {
                // For unordered timesteps
                timesteps_[buffer_idx].push_back(timestep);
            }

            buffer_count_[buffer_idx]++;

            // Check if the buffer is full
            if (buffer_count_[buffer_idx] >= batch_size_) {
                buffer_filled_[buffer_idx] = true;
                ready_sem_.signal();
            }

            lock.unlock();
        }

        /**
         * Wait for Timestep to be ready and return them
         *
         * @param additional_wait Number of additional timesteps to wait for
         * @return Vector of timesteps
         */
        std::vector<Timestep> wait(const int additional_wait = 0) {
            while (!ready_sem_.wait()) {}

            std::unique_lock lock(mutex_);

            // Find a filled buffer
            const size_t buffer_idx = current_buffer_;

            // If we're waiting for additional timesteps
            if (additional_wait > 0) {
                // This would handle the case where we're synchronously waiting
                // for the batch to complete
                buffer_count_[buffer_idx] += additional_wait;
                if (buffer_count_[buffer_idx] >= batch_size_) {
                    buffer_filled_[buffer_idx] = true;
                }
            }

            std::vector<Timestep> result;
            if (buffer_filled_[buffer_idx]) {
                // Get the timesteps
                result = std::move(timesteps_[buffer_idx]);

                // Reset the buffer
                timesteps_[buffer_idx].clear();
                timesteps_[buffer_idx].reserve(batch_size_);
                buffer_count_[buffer_idx] = 0;
                buffer_filled_[buffer_idx] = false;

                // Move to the next buffer
                current_buffer_ = (current_buffer_ + 1) % num_buffers_;
            }

            return result;
        }

    private:
        std::size_t batch_size_;                     // Size of each batch
        std::size_t num_buffers_;                    // Number of circular buffers
        std::size_t current_buffer_;                 // Current buffer index
        std::vector<std::vector<Timestep>> timesteps_;  // Timesteps storage
        std::vector<std::size_t> buffer_count_;      // Count of timesteps in each buffer
        std::vector<bool> buffer_filled_;            // Whether each buffer is filled
        std::mutex mutex_;                           // Mutex for thread safety
        moodycamel::LightweightSemaphore ready_sem_; // Semaphore for ready buffer
    };
}

#endif // ALE_VECTOR_UTILS_HPP_
