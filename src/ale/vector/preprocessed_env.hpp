#ifndef ALE_VECTOR_ATARI_ENV_HPP_
#define ALE_VECTOR_ATARI_ENV_HPP_

#include <memory>
#include <vector>
#include <deque>
#include <random>
#include <string>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include "ale/ale_interface.hpp"
#include "utils.hpp"

namespace ale {
namespace vector {

/**
 * PreprocessedAtariEnv encapsulates a single Atari environment using the ALE Interface with standard preprocessing and stacking.
 */
class PreprocessedAtariEnv {
public:
    /**
     * Constructor
     *
     * @param env_id Unique ID for this environment instance
     * @param rom_path Path to the ROM file
     * @param obs_height Height to resize frames to for observations
     * @param obs_width Width to resize frames to for observations
     * @param frame_skip Number of frames for which to repeat the action
     * @param gray_scale Whether to convert frames to grayscale observation or keep RGB
     * @param maxpool Whether to maxpool observations
     * @param stack_num Number of frames to stack for observations
     * @param noop_max Maximum number of no-ops to perform on resets
     * @param fire_reset Whether to press FIRE during reset
     * @param episodic_life Whether to end episodes when a life is lost
     * @param max_episode_steps Maximum number of steps per episode before truncating
     * @param repeat_action_probability Probability of repeating the last action
     * @param full_action_space Whether to use the full action space
     * @param seed Random seed
     */
    PreprocessedAtariEnv(
        int env_id,
        const fs::path rom_path,
        int obs_height = 84,
        int obs_width = 84,
        int frame_skip = 4,
        bool gray_scale = true,
        bool maxpool = true,
        int stack_num = 4,
        int noop_max = 30,
        bool fire_reset = true,
        bool episodic_life = false,
        int max_episode_steps = 108000,
        float repeat_action_probability = 0.0f,
        bool full_action_space = false,
        int seed = 0
    ) : env_id_(env_id),
        rom_path_(rom_path),
        obs_height_(obs_height),
        obs_width_(obs_width),
        frame_skip_(frame_skip),
        gray_scale_(gray_scale),
        maxpool_(maxpool),
        stack_num_(stack_num),
        noop_max_(noop_max),
        fire_reset_(fire_reset),
        episodic_life_(episodic_life),
        max_episode_steps_(max_episode_steps),
        elapsed_step_(max_episode_steps + 1),
        seed_(seed + env_id),
        rng_gen_(seed_) {

        // Turn off verbosity
        Logger::setMode(Logger::Error);

        // Initialize ALE
        env_ = std::make_unique<ALEInterface>();
        env_->setFloat("repeat_action_probability", repeat_action_probability);
        env_->setInt("random_seed", seed_);
        env_->loadROM(rom_path_);

        // Get action set
        if (full_action_space) {
            action_set_ = env_->getLegalActionSet();
        } else {
            action_set_ = env_->getMinimalActionSet();
        }

        // Check if fire action is available (needed for fire_reset)
        if (fire_reset_) {
            has_fire_action_ = false;
            for (auto a : action_set_) {
                if (a == PLAYER_A_FIRE) {
                    has_fire_action_ = true;
                    break;
                }
            }
            fire_reset_ = has_fire_action_;
        }

        // Initialize random distribution for no-ops
        noop_generator_ = std::uniform_int_distribution<>(0, noop_max_ - 1);

        // Initialize the buffers
        const int channels = gray_scale_ ? 1 : 3;
        for (int i = 0; i < 2; ++i) {
            raw_frames_.emplace_back(210 * 160 * channels);
        }
        resized_frame_.resize(obs_height_ * obs_width_ * channels);
        for (int i = 0; i < stack_num_; ++i) {
            std::vector<uint8_t> frame(obs_height_ * obs_width_ * channels, 0);
            frame_stack_.push_back(std::move(frame));
        }
    }

    /**
     * Reset the environment and return the initial observation
     */
    void reset() {
        env_->reset_game();

        // Perform no-op steps
        int noop_steps = noop_generator_(rng_gen_) + 1 - static_cast<int>(fire_reset_ && has_fire_action_);
        while (noop_steps > 0) {
            env_->act(PLAYER_A_NOOP);
            if (env_->game_over()) {
                env_->reset_game();
            }
            noop_steps--;
        }

        // Press FIRE if required by the environment
        if (fire_reset_ && has_fire_action_) {
            env_->act(PLAYER_A_FIRE);
        }

        // Get the screen data and process it
        std::fill(raw_frames_[0].begin(), raw_frames_[0].end(), 0);
        get_screen_data(raw_frames_[1].data());
        process_screen();
        for (int stack_id = 0; stack_id < stack_num_ - 1; ++stack_id) {
            frame_stack_[stack_id] = resized_frame_;
        }

        // Update state
        elapsed_step_ = 0;
        game_over_ = false;
        lives_ = env_->lives();
        current_action_.action_id = PLAYER_A_NOOP;
    }

    /**
     * Set the action to be taken in the next step
     */
    void set_action(const Action& action) {
        current_action_ = action;
    }

    /**
     * Step the environment using the current action
     */
    void step() {
        float reward = 0.0;
        game_over_ = false;

        int action_id = current_action_.action_id;
        ale::Action action = (action_id < 0 || action_id >= static_cast<int>(action_set_.size())) ? PLAYER_A_NOOP : action_set_[action_id];
        float strength = current_action_.paddle_strength;

        // Execute action for frame_skip frames
        int skip_id = frame_skip_;
        for (; skip_id > 0 && !game_over_; --skip_id) {
            reward += env_->act(action, strength);

            game_over_ = env_->game_over();
            // Handle episodic life
            if (episodic_life_ && env_->lives() < lives_ && env_->lives() > 0) {
                game_over_ = true;
            }

            // Capture last two frames for maxpooling
            if (skip_id <= 2) {
                get_screen_data(raw_frames_[2 - skip_id].data());
            }
        }

        // Process the screen
        process_screen();

        // Update state
        elapsed_step_++;
        lives_ = env_->lives();
        last_reward_ = reward;
    }

    /**
     * Get the current observation
     */
    Timestep get_timestep() const {
        Timestep timestep;
        timestep.env_id = env_id_;

        timestep.reward = last_reward_;
        timestep.terminated = game_over_;
        timestep.truncated = elapsed_step_ >= max_episode_steps_;

        timestep.lives = lives_;
        timestep.frame_number = env_->getFrameNumber();
        timestep.episode_frame_number = env_->getEpisodeFrameNumber();

        // Combine stacked frames into a single observation
        const int channels = gray_scale_ ? 1 : 3;
        const size_t frame_size = obs_height_ * obs_width_ * channels;
        timestep.observation.resize(frame_size * stack_num_);

        for (int i = 0; i < stack_num_; ++i) {
            std::memcpy(
                timestep.observation.data() + i * frame_size,
                frame_stack_[i].data(),
                frame_size
            );
        }

        return timestep;
    }

    /**
     * Check if the episode is over (terminated or truncated)
     */
    bool is_episode_over() const {
        return game_over_ || elapsed_step_ >= max_episode_steps_;
    }

    /**
     * Get the list of available actions
     */
    const ActionVect& get_action_set() const {
        return action_set_;
    }

private:
    /**
     * Get the current screen data from ALE
     */
    void get_screen_data(uint8_t* buffer) {
        const ALEScreen& screen = env_->getScreen();
        uint8_t* ale_screen_data = screen.getArray();

        if (gray_scale_) {
            // Get grayscale screen
            env_->theOSystem->colourPalette().applyPaletteGrayscale(
                buffer, ale_screen_data, screen.width() * screen.height()
            );
        } else {
            // Get RGB screen
            env_->theOSystem->colourPalette().applyPaletteRGB(
                buffer, ale_screen_data, screen.width() * screen.height()
            );
        }
    }

    /**
     * Process the screen and update the frame stack
     */
    void process_screen() {
        const int channels = gray_scale_ ? 1 : 3;
        const int raw_height = 210;
        const int raw_width = 160;

        if (maxpool_) {
            // Maxpool over the last two frames
            const int raw_size = raw_height * raw_width * channels;
            for (int i = 0; i < raw_size; ++i) {
                raw_frames_[0][i] = std::max(raw_frames_[0][i], raw_frames_[1][i]);
            }
        }

        // Resize the raw frame to target dimensions
        cv::Mat src_img(raw_height, raw_width, channels == 1 ? CV_8UC1 : CV_8UC3,
                        const_cast<uint8_t*>(raw_frames_[0].data()));
        cv::Mat dst_img(obs_height_, obs_width_, channels == 1 ? CV_8UC1 : CV_8UC3,
                        resized_frame_.data());
        // Use INTER_AREA for downsampling to avoid moire patterns
        cv::resize(src_img, dst_img, dst_img.size(), 0, 0, cv::INTER_AREA);

        // Push the new frame into the stack
        frame_stack_.pop_front();
        frame_stack_.push_back(resized_frame_);
    }

    int env_id_;                                  // Unique ID for this environment
    fs::path rom_path_;                           // Path to the ROM file
    std::unique_ptr<ALEInterface> env_;           // ALE interface

    ActionVect action_set_;                       // Available actions
    int obs_height_;                              // Height to resize frames to for observations
    int obs_width_;                               // Width to resize frames to for observations
    int frame_skip_;                              // Number of frames for which to repeat the action
    bool gray_scale_;                             // Whether to convert frames to grayscale observation or keep RGB
    bool maxpool_;                                // Whether to maxpool observations
    int stack_num_;                               // Number of frames to stack for observations
    int noop_max_;                                // Maximum number of no-ops at reset
    bool fire_reset_;                             // Whether to press FIRE during reset
    bool has_fire_action_;                        // Whether FIRE action is available for reset
    bool episodic_life_;                          // Whether to end episodes when a life is lost
    int max_episode_steps_;                       // Maximum number of steps per episode before truncating

    int elapsed_step_;                            // Current step in the episode
    bool game_over_;                              // Whether the game is over
    int lives_;                                   // Current number of lives
    float last_reward_;                           // Last reward received
    int seed_;                                    // Random seed
    std::mt19937 rng_gen_;                             // Random number generator
    std::uniform_int_distribution<> noop_generator_;   // Distribution for no-op steps

    Action current_action_;                       // Current action to take

    // Frame buffers
    std::vector<std::vector<uint8_t>> raw_frames_;  // Raw frame buffers for maxpooling
    std::vector<uint8_t> resized_frame_;            // Resized frame buffer
    std::deque<std::vector<uint8_t>> frame_stack_;  // Stack of recent frames
};

} // namespace vector
} // namespace ale

#endif // ALE_VECTOR_ATARI_ENV_HPP_
