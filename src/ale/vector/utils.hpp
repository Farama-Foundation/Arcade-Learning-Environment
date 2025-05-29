#ifndef ALE_VECTOR_UTILS_HPP_
#define ALE_VECTOR_UTILS_HPP_

#include <vector>
#include <atomic>
#include <mutex>
#include <memory>

#ifndef MOODYCAMEL_DELETE_FUNCTION
    #define MOODYCAMEL_DELETE_FUNCTION = delete
#endif

#include "ale/common/Constants.h"
#include "ale/external/lightweightsemaphore.h"

namespace ale::vector {

    /**
     * ActionSlice represents a single action or command to be processed by a worker thread
     */
    struct ActionSlice {
        int env_id;        // ID of the environment to apply the action to
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
        reward_t reward;                  // Reward received in this step
        bool terminated;                  // Whether the game ended
        bool truncated;                   // Whether the episode was truncated due to a time limit
        int lives;                        // Remaining lives in the game
        int frame_number;                 // Frame number since the beginning of the game
        int episode_frame_number;         // Frame number since the beginning of the episode

        std::vector<uint8_t>* final_observation; // Screen pixel data for previous episode last observation with Autoresetmode == SameStep
    };

    /**
     * Observation format enumeration
     */
    enum class ObsFormat {
        Grayscale,  // Single channel grayscale observations
        RGB         // Three channel RGB observations
    };

    enum class AutoresetMode {
        NextStep,  // Will reset the sub-environment in the next step if the episode ended in the previous timestep
        SameStep   // Will reset the sub-environment in the same timestep if the episode ended
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
     *
     * Two modes of operation:
     * 1. Ordered mode (batch_size == num_envs): Waits for all env_ids to be filled
     * 2. Unordered mode (batch_size != num_envs): Uses circular buffer for continuous operation
     */
    class StateBufferQueue {
    public:
        StateBufferQueue(const std::size_t batch_size, const std::size_t num_envs)
            : batch_size_(batch_size),
              num_envs_(num_envs),
              ordered_mode_(batch_size == num_envs),
              timesteps_(num_envs_),
              count_(0),
              write_idx_(0),
              read_idx_(0),
              ready_cv_() {
        }

        /**
         * Write a timestep to the buffer
         */
        void write(const Timestep& timestep) {
            std::unique_lock lock(mutex_);

            if (ordered_mode_) {
                // In ordered mode, place timestep at env_id position
                const int env_id = timestep.env_id;
                timesteps_[env_id] = timestep;
                count_++;
            } else {
                // In unordered mode, use circular buffer
                timesteps_[write_idx_] = timestep;
                write_idx_ = (write_idx_ + 1) % num_envs_;
                count_++;
            }

            // Signal if we have enough for a batch
            if (count_ >= batch_size_) {
                ready_cv_.notify_one();
            }
        }

        /**
         * Collect timesteps when ready and return them
         *
         * @return Vector of timesteps
         */
        std::vector<Timestep> collect() {
            std::unique_lock lock(mutex_);

            // Wait until we have enough timesteps
            ready_cv_.wait(lock, [this] { return count_ >= batch_size_; });

            // Collect the results
            std::vector<Timestep> result;
            result.reserve(batch_size_);

            if (ordered_mode_) {
                // In ordered mode, read in env_id order
                for (size_t i = 0; i < batch_size_; ++i) {
                    result.push_back(std::move(timesteps_[i]));
                }

                // Reset for ordered mode
                count_ = 0;
            } else {
                // In unordered mode, read from circular buffer
                for (size_t i = 0; i < batch_size_; ++i) {
                    result.push_back(std::move(timesteps_[read_idx_]));
                    read_idx_ = (read_idx_ + 1) % num_envs_;
                }

                // Update count
                count_ -= batch_size_;
            }

            return result;
        }

        /**
         * Get the number of timesteps currently buffered
         */
        size_t filled_timesteps() const {
            std::unique_lock lock(mutex_);
            return count_;
        }

    private:
        const std::size_t batch_size_;                  // Size of each batch
        const std::size_t num_envs_;                    // Number of environments
        const bool ordered_mode_;                       // Whether we're in ordered mode
        std::vector<Timestep> timesteps_;               // Buffer for timesteps
        std::size_t count_;                             // Current count of available timesteps
        std::size_t write_idx_;                         // Write position (for unordered mode)
        std::size_t read_idx_;                          // Read position (for unordered mode)
        mutable std::mutex mutex_;                      // Mutex for thread safety
        std::condition_variable ready_cv_;              // Condition variable for signaling
    };
}

#endif // ALE_VECTOR_UTILS_HPP_
