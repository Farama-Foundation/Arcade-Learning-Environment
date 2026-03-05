#ifndef ALE_VECTOR_TYPES_HPP_
#define ALE_VECTOR_TYPES_HPP_

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>
#include <stdexcept>

namespace ale::vector {

/// Autoreset behavior when episode ends
enum class AutoresetMode {
    NextStep,  // Reset on next step() call (observation is first frame of new episode)
    SameStep   // Reset immediately, return final_obs separately
};

/// Observation format
enum class ObsFormat {
    Grayscale,
    RGB
};

/// Action to execute in an environment
struct Action {
    int env_id;
    int action_id;
    float paddle_strength;
    bool force_reset;
};

/// Pointers for worker to write environment output directly into batch buffers
struct OutputSlot {
    uint8_t* obs;
    int* env_id;
    int* reward;
    bool* terminated;
    bool* truncated;
    int* lives;
    int* frame_number;
    int* episode_frame_number;
    uint8_t* final_obs;  // nullptr if not SameStep mode or not needed
};

/// Batch of results with ownership semantics
/// Owns all buffers. Supports releasing ownership for Python handoff.
class BatchResult {
public:
    BatchResult(std::size_t batch_size, std::size_t obs_size, bool include_final_obs)
        : batch_size_(batch_size),
          obs_size_(obs_size),
          observations_(new uint8_t[batch_size * obs_size]),
          env_ids_(new int[batch_size]),
          rewards_(new int[batch_size]),
          terminations_(new bool[batch_size]),
          truncations_(new bool[batch_size]),
          lives_(new int[batch_size]),
          frame_numbers_(new int[batch_size]),
          episode_frame_numbers_(new int[batch_size]),
          final_observations_(include_final_obs ? new uint8_t[batch_size * obs_size] : nullptr) {}

    ~BatchResult() {
        delete[] observations_;
        delete[] env_ids_;
        delete[] rewards_;
        delete[] terminations_;
        delete[] truncations_;
        delete[] lives_;
        delete[] frame_numbers_;
        delete[] episode_frame_numbers_;
        delete[] final_observations_;
    }

    // Move only
    BatchResult(BatchResult&& other) noexcept
        : batch_size_(other.batch_size_),
          obs_size_(other.obs_size_),
          observations_(other.observations_),
          env_ids_(other.env_ids_),
          rewards_(other.rewards_),
          terminations_(other.terminations_),
          truncations_(other.truncations_),
          lives_(other.lives_),
          frame_numbers_(other.frame_numbers_),
          episode_frame_numbers_(other.episode_frame_numbers_),
          final_observations_(other.final_observations_) {
        other.observations_ = nullptr;
        other.env_ids_ = nullptr;
        other.rewards_ = nullptr;
        other.terminations_ = nullptr;
        other.truncations_ = nullptr;
        other.lives_ = nullptr;
        other.frame_numbers_ = nullptr;
        other.episode_frame_numbers_ = nullptr;
        other.final_observations_ = nullptr;
    }

    BatchResult& operator=(BatchResult&& other) noexcept {
        if (this != &other) {
            delete[] observations_;
            delete[] env_ids_;
            delete[] rewards_;
            delete[] terminations_;
            delete[] truncations_;
            delete[] lives_;
            delete[] frame_numbers_;
            delete[] episode_frame_numbers_;
            delete[] final_observations_;

            batch_size_ = other.batch_size_;
            obs_size_ = other.obs_size_;
            observations_ = other.observations_;
            env_ids_ = other.env_ids_;
            rewards_ = other.rewards_;
            terminations_ = other.terminations_;
            truncations_ = other.truncations_;
            lives_ = other.lives_;
            frame_numbers_ = other.frame_numbers_;
            episode_frame_numbers_ = other.episode_frame_numbers_;
            final_observations_ = other.final_observations_;

            other.observations_ = nullptr;
            other.env_ids_ = nullptr;
            other.rewards_ = nullptr;
            other.terminations_ = nullptr;
            other.truncations_ = nullptr;
            other.lives_ = nullptr;
            other.frame_numbers_ = nullptr;
            other.episode_frame_numbers_ = nullptr;
            other.final_observations_ = nullptr;
        }
        return *this;
    }

    BatchResult(const BatchResult&) = delete;
    BatchResult& operator=(const BatchResult&) = delete;

    // Data access for workers to write into
    uint8_t* obs_data() { return observations_; }
    uint8_t* final_obs_data() { return final_observations_; }
    int* env_ids_data() { return env_ids_; }
    int* rewards_data() { return rewards_; }
    bool* terminations_data() { return terminations_; }
    bool* truncations_data() { return truncations_; }
    int* lives_data() { return lives_; }
    int* frame_numbers_data() { return frame_numbers_; }
    int* episode_frame_numbers_data() { return episode_frame_numbers_; }

    // Release ownership - returns pointer and nulls internal pointer
    // Caller takes ownership and must delete[]
    uint8_t* release_observations() { auto p = observations_; observations_ = nullptr; return p; }
    uint8_t* release_final_observations() { auto p = final_observations_; final_observations_ = nullptr; return p; }
    int* release_env_ids() { auto p = env_ids_; env_ids_ = nullptr; return p; }
    int* release_rewards() { auto p = rewards_; rewards_ = nullptr; return p; }
    bool* release_terminations() { auto p = terminations_; terminations_ = nullptr; return p; }
    bool* release_truncations() { auto p = truncations_; truncations_ = nullptr; return p; }
    int* release_lives() { auto p = lives_; lives_ = nullptr; return p; }
    int* release_frame_numbers() { auto p = frame_numbers_; frame_numbers_ = nullptr; return p; }
    int* release_episode_frame_numbers() { auto p = episode_frame_numbers_; episode_frame_numbers_ = nullptr; return p; }

    std::size_t batch_size() const { return batch_size_; }
    std::size_t obs_size() const { return obs_size_; }
    bool has_final_obs() const { return final_observations_ != nullptr; }

private:
    std::size_t batch_size_;
    std::size_t obs_size_;
    uint8_t* observations_;
    int* env_ids_;
    int* rewards_;
    bool* terminations_;
    bool* truncations_;
    int* lives_;
    int* frame_numbers_;
    int* episode_frame_numbers_;
    uint8_t* final_observations_;
};

}  // namespace ale::vector

#endif  // ALE_VECTOR_TYPES_HPP_
