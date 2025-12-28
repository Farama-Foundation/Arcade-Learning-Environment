#ifndef ALE_VECTOR_RESULT_STAGING_HPP_
#define ALE_VECTOR_RESULT_STAGING_HPP_

#include <atomic>
#include <memory>
#include <functional>

#ifndef MOODYCAMEL_DELETE_FUNCTION
    #define MOODYCAMEL_DELETE_FUNCTION = delete
#endif

#include "ale/external/lightweightsemaphore.h"
#include "types.hpp"

namespace ale::vector {

/// Manages result collection with backpressure for async (unordered) mode.
///
/// Workers call stage_result() after completing work. In unordered mode,
/// if batch_size slots are already filled, the worker blocks until recv()
/// releases the current batch.
class ResultStaging {
public:
    ResultStaging(std::size_t batch_size, std::size_t num_envs, std::size_t obs_size, bool same_step_mode)
        : batch_size_(batch_size),
          num_envs_(num_envs),
          obs_size_(obs_size),
          ordered_mode_(batch_size == num_envs),
          same_step_mode_(same_step_mode),
          current_batch_(std::make_unique<BatchResult>(batch_size, obs_size, same_step_mode)),
          staged_count_(0),
          next_slot_(0),
          slots_available_(static_cast<int>(batch_size)),
          batch_ready_(0) {}

    /// Stage a result from a worker thread.
    /// In ordered mode: writes to slot[env_id]
    /// In unordered mode: atomically allocates next slot, may block if batch is full
    ///
    /// @param env_id The environment that produced this result
    /// @param write_fn Callback that writes data to the provided OutputSlot
    void stage_result(int env_id, const std::function<void(OutputSlot&)>& write_fn) {
        std::size_t slot;

        if (ordered_mode_) {
            slot = static_cast<std::size_t>(env_id);
        } else {
            // Acquire a slot permit (blocks if batch is full)
            while (!slots_available_.wait()) {}
            slot = next_slot_.fetch_add(1) % batch_size_;
        }

        // Build output slot pointing into current batch
        OutputSlot output;
        output.obs = current_batch_->obs_data() + slot * obs_size_;
        output.env_id = &current_batch_->env_ids_data()[slot];
        output.reward = &current_batch_->rewards_data()[slot];
        output.terminated = &current_batch_->terminations_data()[slot];
        output.truncated = &current_batch_->truncations_data()[slot];
        output.lives = &current_batch_->lives_data()[slot];
        output.frame_number = &current_batch_->frame_numbers_data()[slot];
        output.episode_frame_number = &current_batch_->episode_frame_numbers_data()[slot];
        output.final_obs = same_step_mode_
            ? current_batch_->final_obs_data() + slot * obs_size_
            : nullptr;

        // Let worker write its data
        write_fn(output);

        // Signal completion
        std::size_t completed = staged_count_.fetch_add(1) + 1;
        if (completed == batch_size_) {
            batch_ready_.signal(1);
        }
    }

    /// Wait for batch_size results to be staged. Called by recv().
    void wait_for_batch() {
        while (!batch_ready_.wait()) {}
    }

    /// Release current batch and prepare for next.
    /// Returns the completed batch (transfers ownership).
    /// Releases slot permits for blocked workers.
    BatchResult release_batch() {
        // Take ownership of completed batch
        auto result = std::move(*current_batch_);

        // Allocate fresh batch for next round
        current_batch_ = std::make_unique<BatchResult>(batch_size_, obs_size_, same_step_mode_);

        // Reset counters
        staged_count_.store(0);
        next_slot_.store(0);

        // Release permits for blocked workers (they'll write to new batch)
        if (!ordered_mode_) {
            slots_available_.signal(static_cast<int>(batch_size_));
        }

        return result;
    }

    std::size_t batch_size() const { return batch_size_; }
    std::size_t obs_size() const { return obs_size_; }
    bool is_ordered() const { return ordered_mode_; }

private:
    const std::size_t batch_size_;
    const std::size_t num_envs_;
    const std::size_t obs_size_;
    const bool ordered_mode_;
    const bool same_step_mode_;

    std::unique_ptr<BatchResult> current_batch_;

    std::atomic<std::size_t> staged_count_;
    std::atomic<std::size_t> next_slot_;

    moodycamel::LightweightSemaphore slots_available_;  // Permits for staging (unordered only)
    moodycamel::LightweightSemaphore batch_ready_;      // Signaled when batch is full
};

}  // namespace ale::vector

#endif  // ALE_VECTOR_RESULT_STAGING_HPP_
