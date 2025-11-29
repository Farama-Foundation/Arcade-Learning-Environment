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
     * WriteSlot provides destinations for workers to write data directly.
     * All pointers point into externally allocated BatchData arrays.
     */
    struct WriteSlot {
        int slot_index;                      // Index in the batch
        uint8_t* obs_dest;                   // Pointer to write observation data
        int* env_id_dest;                    // Pointer to write env_id
        int* reward_dest;                    // Pointer to write reward
        bool* terminated_dest;               // Pointer to write terminated flag
        bool* truncated_dest;                // Pointer to write truncated flag
        int* lives_dest;                     // Pointer to write lives
        int* frame_number_dest;              // Pointer to write frame_number
        int* episode_frame_number_dest;      // Pointer to write episode_frame_number
        uint8_t* final_obs_dest;             // Pointer for final_obs (SameStep mode)
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
     * StateBuffer manages output buffers for vectorized environment results.
     *
     * The buffer is set externally before workers begin writing.
     * Workers write directly to allocated slots, avoiding intermediate copies.
     *
     * Two modes of operation:
     * 1. Ordered mode (batch_size == num_envs): Slot index equals env_id
     * 2. Unordered mode (batch_size != num_envs): Atomic slot allocation
     */
    class StateBuffer {
    public:
        StateBuffer(const std::size_t batch_size, const std::size_t num_envs, const std::size_t obs_size)
            : batch_size_(batch_size),
              num_envs_(num_envs),
              obs_size_(obs_size),
              ordered_mode_(batch_size == num_envs),
              output_obs_buffer_(nullptr),
              final_obs_buffer_(nullptr),
              env_ids_buffer_(nullptr),
              rewards_buffer_(nullptr),
              terminations_buffer_(nullptr),
              truncations_buffer_(nullptr),
              lives_buffer_(nullptr),
              frame_numbers_buffer_(nullptr),
              episode_frame_numbers_buffer_(nullptr),
              count_(0),
              write_idx_(0),
              sem_ready_(0),
              sem_read_(1),
              sem_slots_(batch_size) {}  // Initialize with batch_size permits

        /**
         * Set the output buffer that workers will write observations into.
         * MUST be called before enqueueing any actions that will use this buffer.
         *
         * @param obs_buffer Pointer to allocated buffer of size batch_size * obs_size
         */
        void set_output_buffer(uint8_t* obs_buffer) {
            output_obs_buffer_ = obs_buffer;
        }

        /**
         * Set the final_obs output buffer for SameStep autoreset mode.
         *
         * @param final_obs_buffer Pointer to allocated buffer of size batch_size * obs_size
         */
        void set_final_obs_buffer(uint8_t* final_obs_buffer) {
            final_obs_buffer_ = final_obs_buffer;
        }

        /**
         * Set the metadata output buffers that workers will write into.
         * MUST be called before enqueueing any actions that will use these buffers.
         *
         * @param env_ids Pointer to allocated array of size batch_size
         * @param rewards Pointer to allocated array of size batch_size
         * @param terminations Pointer to allocated array of size batch_size
         * @param truncations Pointer to allocated array of size batch_size
         * @param lives Pointer to allocated array of size batch_size
         * @param frame_numbers Pointer to allocated array of size batch_size
         * @param episode_frame_numbers Pointer to allocated array of size batch_size
         */
        void set_metadata_buffers(
            int* env_ids,
            int* rewards,
            bool* terminations,
            bool* truncations,
            int* lives,
            int* frame_numbers,
            int* episode_frame_numbers
        ) {
            env_ids_buffer_ = env_ids;
            rewards_buffer_ = rewards;
            terminations_buffer_ = terminations;
            truncations_buffer_ = truncations;
            lives_buffer_ = lives;
            frame_numbers_buffer_ = frame_numbers;
            episode_frame_numbers_buffer_ = episode_frame_numbers;
        }

        /**
         * Allocate a write slot for a worker thread.
         * Returns pointers for direct writing into the output buffer.
         *
         * Thread-safe: multiple workers can call simultaneously.
         * In unordered mode, blocks if all slots are occupied.
         *
         * @param env_id The environment ID requesting a slot
         * @return WriteSlot with pointers into output buffers
         */
        WriteSlot allocate_write_slot(int env_id) {
            // In unordered mode, block if all slots are occupied
            if (!ordered_mode_) {
                while (!sem_slots_.wait()) {}  // Acquire permit, blocks if none available
            }

            WriteSlot slot;

            if (ordered_mode_) {
                // In ordered mode, slot index equals env_id
                slot.slot_index = env_id;
            } else {
                // In unordered mode, atomically allocate next available slot
                slot.slot_index = static_cast<int>(write_idx_.fetch_add(1) % batch_size_);
            }

            const int idx = slot.slot_index;

            // Set observation pointers
            slot.obs_dest = output_obs_buffer_ + idx * obs_size_;

            // Set final_obs pointer (only used in SameStep mode, nullptr in NextStep mode)
            slot.final_obs_dest = final_obs_buffer_ != nullptr
                ? final_obs_buffer_ + idx * obs_size_
                : nullptr;

            // Set metadata pointers (directly into BatchData arrays)
            slot.env_id_dest = &env_ids_buffer_[idx];
            slot.reward_dest = &rewards_buffer_[idx];
            slot.terminated_dest = &terminations_buffer_[idx];
            slot.truncated_dest = &truncations_buffer_[idx];
            slot.lives_dest = &lives_buffer_[idx];
            slot.frame_number_dest = &frame_numbers_buffer_[idx];
            slot.episode_frame_number_dest = &episode_frame_numbers_buffer_[idx];

            return slot;
        }

        /**
         * Mark a slot as complete. Called by worker after writing all data.
         * When all slots are complete, signals that batch is ready.
         */
        void mark_complete() {
            const auto old_count = count_.fetch_add(1);
            if (old_count + 1 == batch_size_) {
                sem_ready_.signal(1);
            }
        }

        /**
         * Wait for batch to complete. Blocks until all slots are filled.
         */
        void wait_for_batch() {
            while (!sem_ready_.wait()) {}
        }

        /**
         * Reset state for next batch. Must be called after collecting results.
         */
        void reset() {
            count_.store(0);
            write_idx_.store(0);
            output_obs_buffer_ = nullptr;
            final_obs_buffer_ = nullptr;
            env_ids_buffer_ = nullptr;
            rewards_buffer_ = nullptr;
            terminations_buffer_ = nullptr;
            truncations_buffer_ = nullptr;
            lives_buffer_ = nullptr;
            frame_numbers_buffer_ = nullptr;
            episode_frame_numbers_buffer_ = nullptr;
        }

        /**
         * Release all slots for the next batch.
         * Called by recv() after transferring buffer ownership to Python.
         * This allows waiting workers to proceed and allocate slots.
         */
        void release_slots() {
            if (!ordered_mode_) {
                sem_slots_.signal(batch_size_);  // Release batch_size permits
            }
        }

        // Accessors
        std::size_t get_batch_size() const { return batch_size_; }
        std::size_t get_obs_size() const { return obs_size_; }

    private:
        const std::size_t batch_size_;
        const std::size_t num_envs_;
        const std::size_t obs_size_;
        const bool ordered_mode_;

        // External output buffers (set via set_output_buffer / set_final_obs_buffer / set_metadata_buffers)
        uint8_t* output_obs_buffer_;
        uint8_t* final_obs_buffer_;
        int* env_ids_buffer_;
        int* rewards_buffer_;
        bool* terminations_buffer_;
        bool* truncations_buffer_;
        int* lives_buffer_;
        int* frame_numbers_buffer_;
        int* episode_frame_numbers_buffer_;

        // Synchronization
        std::atomic<std::size_t> count_;
        std::atomic<std::size_t> write_idx_;
        moodycamel::LightweightSemaphore sem_ready_;
        moodycamel::LightweightSemaphore sem_read_;
        moodycamel::LightweightSemaphore sem_slots_;  // Controls slot availability
    };
}

#endif // ALE_VECTOR_UTILS_HPP_
