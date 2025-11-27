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
     * Lightweight metadata without observation data.
     * Used when observations are written directly to output buffer.
     */
    struct TimestepMetadata {
        int env_id;                       // ID of the environment
        reward_t reward;                  // Reward received
        bool terminated;                  // Whether the game ended
        bool truncated;                   // Whether episode was truncated
        int lives;                        // Remaining lives
        int frame_number;                 // Frame number since game start
        int episode_frame_number;         // Frame number since episode start
    };

    /**
     * WriteSlot provides destinations for workers to write data directly.
     */
    struct WriteSlot {
        int slot_index;           // Index in the batch
        uint8_t* obs_dest;        // Pointer to write observation data
        TimestepMetadata* meta;   // Pointer to write metadata
        uint8_t* final_obs_dest;  // Pointer for final_obs (SameStep mode)
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
              metadata_(batch_size),
              final_obs_buffer_(batch_size * obs_size),
              has_final_obs_(batch_size, false),
              output_obs_buffer_(nullptr),
              count_(0),
              write_idx_(0),
              sem_ready_(0),
              sem_read_(1) {}

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
         * Allocate a write slot for a worker thread.
         * Returns pointers for direct writing into the output buffer.
         *
         * Thread-safe: multiple workers can call simultaneously.
         *
         * @param env_id The environment ID requesting a slot
         * @return WriteSlot with pointers into output buffers
         */
        WriteSlot allocate_write_slot(int env_id) {
            WriteSlot slot;

            if (ordered_mode_) {
                // In ordered mode, slot index equals env_id
                slot.slot_index = env_id;
            } else {
                // In unordered mode, atomically allocate next available slot
                slot.slot_index = static_cast<int>(write_idx_.fetch_add(1) % batch_size_);
            }

            slot.obs_dest = output_obs_buffer_ + slot.slot_index * obs_size_;
            slot.meta = &metadata_[slot.slot_index];
            slot.final_obs_dest = final_obs_buffer_.data() + slot.slot_index * obs_size_;

            return slot;
        }

        /**
         * Mark that a slot has final observation data (for SameStep autoreset).
         *
         * @param slot_index The slot index to mark
         */
        void mark_slot_has_final_obs(int slot_index) {
            has_final_obs_[slot_index] = true;
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
            std::fill(has_final_obs_.begin(), has_final_obs_.end(), false);
            output_obs_buffer_ = nullptr;
        }

        // Accessors
        TimestepMetadata* get_metadata() { return metadata_.data(); }
        const TimestepMetadata* get_metadata() const { return metadata_.data(); }
        uint8_t* get_final_obs_buffer() { return final_obs_buffer_.data(); }
        const uint8_t* get_final_obs_buffer() const { return final_obs_buffer_.data(); }
        uint8_t* get_has_final_obs() { return has_final_obs_.data(); }
        const uint8_t* get_has_final_obs() const { return has_final_obs_.data(); }
        std::size_t get_batch_size() const { return batch_size_; }
        std::size_t get_obs_size() const { return obs_size_; }

    private:
        const std::size_t batch_size_;
        const std::size_t num_envs_;
        const std::size_t obs_size_;
        const bool ordered_mode_;

        // Internal storage for metadata and final observations
        std::vector<TimestepMetadata> metadata_;
        std::vector<uint8_t> final_obs_buffer_;
        std::vector<uint8_t> has_final_obs_;  // uint8_t instead of bool for .data() access

        // External output buffer (set via set_output_buffer)
        uint8_t* output_obs_buffer_;

        // Synchronization
        std::atomic<std::size_t> count_;
        std::atomic<std::size_t> write_idx_;
        moodycamel::LightweightSemaphore sem_ready_;
        moodycamel::LightweightSemaphore sem_read_;
    };
}

#endif // ALE_VECTOR_UTILS_HPP_
