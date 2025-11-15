#ifndef ALE_VECTOR_UTILS_HPP_
#define ALE_VECTOR_UTILS_HPP_

#include <vector>
#include <atomic>
#include <mutex>
#include <memory>
#include <thread>

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
    class ActionQueue {
    public:
        explicit ActionQueue(const std::size_t num_envs)
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
     * StateBuffer handles the collection of timesteps from environments
     *
     * Two modes of operation:
     * 1. Ordered mode (batch_size == num_envs): Waits for all env_ids to be filled
     * 2. Unordered mode (batch_size != num_envs): Uses circular buffer for continuous operation
     */
    class StateBuffer {
    public:
        StateBuffer(const std::size_t batch_size, const std::size_t num_envs)
            : batch_size_(batch_size),
              num_envs_(num_envs),
              ordered_mode_(batch_size == num_envs),
              timesteps_(num_envs_),
              slot_ready_(num_envs_),
              count_(0),
              write_idx_(0),
              read_idx_(0),
              sem_ready_(0),      // Initially no batches ready
              sem_read_(1) {      // Allow one reader at a time
            // Initialize all slot ready flags to false
            for (auto& flag : slot_ready_) {
                flag.store(false, std::memory_order_relaxed);
            }
        }

        /**
         * Write a timestep to the buffer
         * Multiple threads can write simultaneously
         */
        void write(const Timestep& timestep) {
            if (ordered_mode_) {
                // In ordered mode, place timestep at env_id position
                const int env_id = timestep.env_id;
                timesteps_[env_id] = timestep;

                // Memory barrier: mark this slot as ready with release semantics
                // This ensures all writes to timesteps_[env_id] are visible before
                // the slot is marked ready
                slot_ready_[env_id].store(true, std::memory_order_release);

                // Atomically increment count and check if batch is ready
                const auto old_count = count_.fetch_add(1);
                if (old_count + 1 == batch_size_) {
                    // Exactly one thread will see count == batch_size_ in ordered mode
                    sem_ready_.signal(1);
                }
            } else {
                // In unordered mode, use circular buffer
                // Each thread gets a unique index atomically
                const auto idx = write_idx_.fetch_add(1) % num_envs_;

                // Wait for the slot to be available (consumed from previous circular buffer cycle)
                // This prevents overwriting data that hasn't been read yet
                while (slot_ready_[idx].load(std::memory_order_acquire)) {
                    // Spin until the reader marks this slot as consumed
                    std::this_thread::yield();
                }

                // Write the timestep data
                timesteps_[idx] = timestep;

                // Memory barrier: ensure all writes to timesteps_[idx] are visible
                // before we mark the slot as ready. Release semantics guarantee that
                // any thread that sees slot_ready_[idx] == true will also see the
                // complete timestep data.
                slot_ready_[idx].store(true, std::memory_order_release);

                // Atomically increment count and check if batch is ready
                const auto old_count = count_.fetch_add(1);
                // Signal if we just crossed a batch boundary
                if ((old_count + 1) / batch_size_ > old_count / batch_size_) {
                    sem_ready_.signal(1);
                }
            }
        }

        /**
         * Collect timesteps when ready and return them
         *
         * @return Vector of timesteps
         */
        std::vector<Timestep> collect() {
            // Wait until a batch is ready
            while (!sem_ready_.wait()) {}

            // Acquire read semaphore
            while (!sem_read_.wait()) {}

            // Collect the results
            std::vector<Timestep> result;
            result.reserve(batch_size_);

            if (ordered_mode_) {
                // In ordered mode, read in env_id order
                for (size_t i = 0; i < batch_size_; ++i) {
                    // Memory barrier: ensure we see the complete timestep data
                    // by acquiring the slot_ready flag that was set with release semantics
                    while (!slot_ready_[i].load(std::memory_order_acquire)) {
                        // Spin until this slot's write is complete
                        std::this_thread::yield();
                    }

                    result.push_back(std::move(timesteps_[i]));

                    // Mark slot as consumed
                    slot_ready_[i].store(false, std::memory_order_release);
                }

                // Reset count for ordered mode (all items consumed)
                count_.store(0);
            } else {
                // In unordered mode, read from circular buffer
                for (size_t i = 0; i < batch_size_; ++i) {
                    const auto idx = read_idx_.fetch_add(1) % num_envs_;

                    // Memory barrier: wait until the slot is ready and acquire semantics
                    // ensure we see all writes to timesteps_[idx] that happened before
                    // slot_ready_[idx] was set to true
                    while (!slot_ready_[idx].load(std::memory_order_acquire)) {
                        // Spin until the writer completes writing to this slot
                        std::this_thread::yield();
                    }

                    result.push_back(std::move(timesteps_[idx]));

                    // Mark slot as available for the next write (after data is moved)
                    slot_ready_[idx].store(false, std::memory_order_release);
                }

                // Atomically decrease count by batch_size_
                count_.fetch_sub(batch_size_);
            }

            // Release read semaphore
            sem_read_.signal(1);

            return result;
        }

        /**
         * Get the number of timesteps currently buffered
         */
        size_t filled_timesteps() const {
            return count_.load();
        }

    private:
        const std::size_t batch_size_;                    // Size of each batch
        const std::size_t num_envs_;                      // Number of environments
        const bool ordered_mode_;                         // Whether we're in ordered mode
        std::vector<Timestep> timesteps_;                 // Buffer for timesteps
        std::vector<std::atomic<bool>> slot_ready_;       // Per-slot ready flags for synchronization

        // Atomic counters for lock-free operations
        std::atomic<std::size_t> count_;                  // Current count of available timesteps
        std::atomic<std::size_t> write_idx_;              // Write position (for unordered mode)
        std::atomic<std::size_t> read_idx_;               // Read position (for unordered mode)

        // Semaphores for coordination
        moodycamel::LightweightSemaphore sem_ready_;      // Signals when a batch is ready for collection
        moodycamel::LightweightSemaphore sem_read_;       // Controls access to read operations
    };
}

#endif // ALE_VECTOR_UTILS_HPP_
