#ifndef ALE_VECTOR_PREPROCESSED_ENV_HPP_
#define ALE_VECTOR_PREPROCESSED_ENV_HPP_

#include <memory>
#include <vector>
#include <random>
#include <algorithm>
#include <filesystem>

// SIMD intrinsics for maxpooling optimization
#if defined(__AVX2__)
    #include <immintrin.h>
#elif defined(__SSE2__)
    #include <emmintrin.h>
#elif defined(__ARM_NEON)
    #include <arm_neon.h>
#endif

#include <opencv2/opencv.hpp>

#include "ale/common/Constants.h"
#include "ale/ale_interface.hpp"
#include "reset_pool.hpp"
#include "types.hpp"

namespace fs = std::filesystem;

namespace ale::vector {

/// Single ALE environment with standard preprocessing:
/// - Frame skipping with max-pooling
/// - Grayscale or RGB observations
/// - Resize to specified dimensions
/// - Frame stacking
/// - Noop-max on reset
/// - Fire on reset (for games that require it)
/// - Episodic life / life loss info
/// - Reward clipping
class PreprocessedEnv {
public:
    /// Start states captured before the pool starts being drawn from; 0 disables
    /// the pool. See reset_pool.hpp for why more than one is needed.
    static constexpr int kDefaultResetPoolDraws = 8;

    PreprocessedEnv(
        int env_id,
        const fs::path& rom_path,
        int img_height = 84,
        int img_width = 84,
        int frame_skip = 4,
        bool maxpool = true,
        bool grayscale = true,
        int stack_num = 4,
        int noop_max = 30,
        bool use_fire_reset = true,
        bool episodic_life = false,
        bool life_loss_info = false,
        bool reward_clipping = true,
        int max_episode_steps = 108000,
        float repeat_action_probability = 0.0f,
        bool full_action_space = false,
        int seed = -1,
        int reset_pool_size = kDefaultResetPoolDraws,
        bool reset_pool_prefill = false
    );

    /// Set seed for next reset
    void set_seed(int seed);

    /// Set action for next step
    void set_action(int action_id, float paddle_strength);

    /// Reset environment
    void reset();

    /// Step environment using previously set action
    void step();

    /// Write current state to output slot
    void write_to(const OutputSlot& slot) const;

    /// Write only observation to destination (for final_obs before reset)
    void write_obs_to(uint8_t* dest) const;

    /// Check if episode is over
    bool is_episode_over() const;

    /// Get available actions
    const ActionVect& action_set() const { return action_set_; }

    /// Get stacked observation size in bytes
    std::size_t stacked_obs_size() const { return obs_size_ * stack_num_; }

    /// Get channels per frame (1 for grayscale, 3 for RGB)
    int channels_per_frame() const { return channels_per_frame_; }

private:
    /// Bring the console up to a start state, either by emulating reset_game() or
    /// by restoring a snapshot of an earlier one.
    void reset_console();

    /// Emulate one real reset_game() and keep the resulting start state.
    void capture_reset_state();

    void get_screen_grayscale(uint8_t* buffer) const;
    void get_screen_rgb(uint8_t* buffer) const;
    void process_screen();

    int env_id_;
    fs::path rom_path_;
    std::unique_ptr<ALEInterface> ale_;

    ActionVect action_set_;

    // Observation settings
    ObsFormat obs_format_;
    int channels_per_frame_;
    int raw_frame_height_;
    int raw_frame_width_;
    int raw_frame_size_;
    int raw_size_;
    int obs_frame_height_;
    int obs_frame_width_;
    int obs_size_;
    int stack_num_;

    // Preprocessing settings
    int frame_skip_;
    bool maxpool_;
    int noop_max_;
    bool use_fire_reset_;
    bool has_fire_action_;
    bool episodic_life_;
    bool life_loss_info_;
    bool reward_clipping_;
    int max_episode_steps_;

    // RNG
    std::mt19937 rng_;
    std::uniform_int_distribution<> noop_dist_;

    // State
    int elapsed_steps_;
    bool game_over_;
    int lives_;
    bool was_life_lost_;
    int reward_;
    int current_action_id_;
    float current_paddle_strength_;
    int pending_seed_;

    // Snapshot-restore reset. reset_game() emulates ~68 frames
    // of console bring-up; restoring a snapshot of one costs ~10-20 us instead. The
    // first reset_pool_size resets are emulated for real and kept, and later ones
    // draw from the pool - see reset_pool.hpp for why a single snapshot is not
    // enough. Invalidated by a seeded reset, which reloads the ROM.
    ResetSnapshotPool reset_pool_;
    // When set, the pool is filled in one go on the first seeded reset instead of
    // over the first episodes. Costs reset_pool_size real resets up front and
    // makes every later episode boundary cheap, which callers who care about
    // consistent step latency - benchmarks, evaluation - usually prefer.
    bool reset_pool_prefill_;
    // Kept separate from rng_ so that pool draws cannot shift the noop-start
    // sequence, which must stay exactly as it was.
    std::mt19937 pool_rng_;
    // ALEState::load copies m_frame_number from the snapshot, rewinding the
    // ROM-lifetime frame counter. Report it with the elided frames added back -
    // including those the replaced reset_game() would have counted - so
    // info["frame_number"] matches what real resets report.
    long snapshot_frame_offset_;

    // Frame buffers
    std::vector<std::vector<uint8_t>> raw_frames_;
    std::vector<uint8_t> frame_stack_;
    int frame_stack_idx_;
};

}  // namespace ale::vector

#endif  // ALE_VECTOR_PREPROCESSED_ENV_HPP_
