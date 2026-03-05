#include "preprocessed_env.hpp"

namespace ale::vector {

/**
 * Maxpool two uint8_t frames using SIMD when available
 * @param dst Destination buffer (will be modified in-place with max values)
 * @param src Source buffer to compare against
 * @param size Number of bytes to process
 */
inline void maxpool_frames(uint8_t* dst, const uint8_t* src, int size) {
    int i = 0;

#if defined(__AVX2__)
    // Process 32 bytes at a time with AVX2
    for (; i + 32 <= size; i += 32) {
        __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + i));
        __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        __m256i max_val = _mm256_max_epu8(a, b);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), max_val);
    }
#elif defined(__SSE2__)
    // Process 16 bytes at a time with SSE2
    for (; i + 16 <= size; i += 16) {
        __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dst + i));
        __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        __m128i max_val = _mm_max_epu8(a, b);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), max_val);
    }
#elif defined(__ARM_NEON)
    // Process 16 bytes at a time with NEON
    for (; i + 16 <= size; i += 16) {
        uint8x16_t a = vld1q_u8(dst + i);
        uint8x16_t b = vld1q_u8(src + i);
        uint8x16_t max_val = vmaxq_u8(a, b);
        vst1q_u8(dst + i, max_val);
    }
#endif
    // Handle remainder with scalar code
    for (; i < size; ++i) {
        dst[i] = std::max(dst[i], src[i]);
    }
}

PreprocessedEnv::PreprocessedEnv(
    int env_id,
    const fs::path& rom_path,
    int img_height,
    int img_width,
    int frame_skip,
    bool maxpool,
    bool grayscale,
    int stack_num,
    int noop_max,
    bool use_fire_reset,
    bool episodic_life,
    bool life_loss_info,
    bool reward_clipping,
    int max_episode_steps,
    float repeat_action_probability,
    bool full_action_space,
    int seed
) : env_id_(env_id),
    rom_path_(rom_path),
    obs_format_(grayscale ? ObsFormat::Grayscale : ObsFormat::RGB),
    channels_per_frame_(grayscale ? 1 : 3),
    obs_frame_height_(img_height),
    obs_frame_width_(img_width),
    stack_num_(stack_num),
    frame_skip_(frame_skip),
    maxpool_(maxpool),
    noop_max_(noop_max),
    use_fire_reset_(use_fire_reset),
    has_fire_action_(false),
    episodic_life_(episodic_life),
    life_loss_info_(life_loss_info),
    reward_clipping_(reward_clipping),
    max_episode_steps_(max_episode_steps),
    rng_(seed == -1 ? std::random_device{}() : static_cast<unsigned>(seed)),
    noop_dist_(0, noop_max > 0 ? noop_max - 1 : 0),
    elapsed_steps_(max_episode_steps + 1),
    game_over_(false),
    lives_(0),
    was_life_lost_(false),
    reward_(0),
    current_action_id_(PLAYER_A_NOOP),
    current_paddle_strength_(1.0f),
    pending_seed_(-1),
    frame_stack_idx_(0)
{
    // Turn off verbosity
    Logger::setMode(Logger::Error);

    // Initialize ALE
    ale_ = std::make_unique<ALEInterface>();
    ale_->setFloat("repeat_action_probability", repeat_action_probability);
    ale_->setInt("random_seed", seed);
    ale_->loadROM(rom_path_);

    // Get action set
    if (full_action_space) {
        action_set_ = ale_->getLegalActionSet();
    } else {
        action_set_ = ale_->getMinimalActionSet();
    }

    // Check if fire action is available (needed for fire_reset)
    if (use_fire_reset_) {
        has_fire_action_ = false;
        for (const auto& a : action_set_) {
            if (a == PLAYER_A_FIRE) {
                has_fire_action_ = true;
                break;
            }
        }
    }

    const ALEScreen& screen = ale_->getScreen();
    raw_frame_height_ = screen.height();
    raw_frame_width_ = screen.width();
    raw_frame_size_ = raw_frame_height_ * raw_frame_width_;
    raw_size_ = raw_frame_height_ * raw_frame_width_ * channels_per_frame_;
    obs_size_ = obs_frame_height_ * obs_frame_width_ * channels_per_frame_;

    // Initialize the buffers
    for (int i = 0; i < 2; ++i) {
        raw_frames_.emplace_back(raw_size_);
    }
    frame_stack_ = std::vector<uint8_t>(stack_num_ * obs_size_, 0);
    frame_stack_idx_ = 0;
}

void PreprocessedEnv::set_seed(int seed) {
    pending_seed_ = seed;
}

void PreprocessedEnv::set_action(int action_id, float paddle_strength) {
    current_action_id_ = action_id;
    current_paddle_strength_ = paddle_strength;
}

void PreprocessedEnv::reset() {
    if (pending_seed_ >= 0) {
        ale_->setInt("random_seed", pending_seed_);
        rng_.seed(pending_seed_);
        ale_->loadROM(rom_path_);
        pending_seed_ = -1;
    }
    ale_->reset_game();

    // Press FIRE if required by the environment
    if (use_fire_reset_ && has_fire_action_) {
        ale_->act(PLAYER_A_FIRE);
    }

    // Perform no-op steps
    int noop_steps = noop_dist_(rng_) - static_cast<int>(use_fire_reset_ && has_fire_action_);
    while (noop_steps > 0) {
        ale_->act(PLAYER_A_NOOP);
        if (ale_->game_over()) {
            ale_->reset_game();
        }
        noop_steps--;
    }

    // Clear the frame stack
    std::fill(frame_stack_.begin(), frame_stack_.end(), 0);
    frame_stack_idx_ = 0;

    // Get the screen data and process it
    if (obs_format_ == ObsFormat::Grayscale) {
        get_screen_grayscale(raw_frames_[0].data());
    } else {
        get_screen_rgb(raw_frames_[0].data());
    }
    std::fill(raw_frames_[1].begin(), raw_frames_[1].end(), 0);

    // Process the screen
    process_screen();

    // Update state
    elapsed_steps_ = 0;
    reward_ = 0;
    game_over_ = false;
    lives_ = ale_->lives();
    was_life_lost_ = false;
    current_action_id_ = PLAYER_A_NOOP;
}

void PreprocessedEnv::step() {
    // Validate action
    if (current_action_id_ < 0 || current_action_id_ >= static_cast<int>(action_set_.size())) {
        throw std::out_of_range("Invalid action_id: " + std::to_string(current_action_id_) +
                                ", available actions: " + std::to_string(action_set_.size()));
    }
    const ale::Action action = action_set_[current_action_id_];
    const float strength = current_paddle_strength_;

    // Execute action for frame_skip frames
    reward_t reward = 0;
    for (int skip_id = frame_skip_; skip_id > 0; --skip_id) {
        reward += ale_->act(action, strength);

        game_over_ = ale_->game_over();
        elapsed_steps_++;
        was_life_lost_ = ale_->lives() < lives_ && ale_->lives() > 0;

        if (game_over_ || elapsed_steps_ >= max_episode_steps_ || (episodic_life_ && was_life_lost_)) {
            break;
        }

        // Captures last two frames for maxpooling
        if (skip_id <= 2) {
            if (obs_format_ == ObsFormat::Grayscale) {
                get_screen_grayscale(raw_frames_[skip_id - 1].data());
            } else {
                get_screen_rgb(raw_frames_[skip_id - 1].data());
            }
        }
    }

    // Update state
    process_screen();
    lives_ = ale_->lives();
    reward_ = reward_clipping_ ? std::clamp<int>(reward, -1, 1) : reward;
}

void PreprocessedEnv::write_to(const OutputSlot& slot) const {
    *slot.env_id = env_id_;
    *slot.reward = reward_;
    *slot.terminated = game_over_ || ((life_loss_info_ || episodic_life_) && was_life_lost_);
    *slot.truncated = elapsed_steps_ >= max_episode_steps_ && !(*slot.terminated);
    *slot.lives = lives_;
    *slot.frame_number = ale_->getFrameNumber();
    *slot.episode_frame_number = ale_->getEpisodeFrameNumber();

    // Linearize circular frame_stack to destination
    for (int i = 0; i < stack_num_; ++i) {
        int src_idx = (frame_stack_idx_ + i) % stack_num_;
        std::memcpy(
            slot.obs + i * obs_size_,
            frame_stack_.data() + src_idx * obs_size_,
            obs_size_
        );
    }
}

void PreprocessedEnv::write_obs_to(uint8_t* dest) const {
    for (int i = 0; i < stack_num_; ++i) {
        int src_idx = (frame_stack_idx_ + i) % stack_num_;
        std::memcpy(
            dest + i * obs_size_,
            frame_stack_.data() + src_idx * obs_size_,
            obs_size_
        );
    }
}

bool PreprocessedEnv::is_episode_over() const {
    return game_over_ || elapsed_steps_ >= max_episode_steps_ || (episodic_life_ && was_life_lost_);
}

void PreprocessedEnv::get_screen_grayscale(uint8_t* buffer) const {
    const ALEScreen& screen = ale_->getScreen();
    uint8_t* ale_screen_data = screen.getArray();

    ale_->theOSystem->colourPalette().applyPaletteGrayscale(
        buffer, ale_screen_data, raw_frame_size_
    );
}

void PreprocessedEnv::get_screen_rgb(uint8_t* buffer) const {
    const ALEScreen& screen = ale_->getScreen();
    uint8_t* ale_screen_data = screen.getArray();

    ale_->theOSystem->colourPalette().applyPaletteRGB(
        buffer, ale_screen_data, raw_frame_size_
    );
}

void PreprocessedEnv::process_screen() {
    // Maxpool raw frames if required
    if (maxpool_) {
        maxpool_frames(raw_frames_[0].data(), raw_frames_[1].data(), raw_size_);
    }

    // Get pointer to current position in circular buffer
    uint8_t* dest_ptr = frame_stack_.data() + (frame_stack_idx_ * obs_size_);

    // Resize directly into the circular buffer or copy if no resize needed
    if (obs_frame_height_ != raw_frame_height_ || obs_frame_width_ != raw_frame_width_) {
        auto cv2_format = (obs_format_ == ObsFormat::Grayscale) ? CV_8UC1 : CV_8UC3;
        cv::Mat src_img(raw_frame_height_, raw_frame_width_, cv2_format, raw_frames_[0].data());
        cv::Mat dst_img(obs_frame_height_, obs_frame_width_, cv2_format, dest_ptr);
        cv::resize(src_img, dst_img, dst_img.size(), 0, 0, cv::INTER_AREA);
    } else {
        // No resize needed, copy directly to circular buffer
        std::memcpy(dest_ptr, raw_frames_[0].data(), raw_size_);
    }

    // Move to next position in circular buffer
    frame_stack_idx_ = (frame_stack_idx_ + 1) % stack_num_;
}

}  // namespace ale::vector
