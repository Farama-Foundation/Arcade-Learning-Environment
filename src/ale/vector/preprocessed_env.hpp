#ifndef ALE_VECTOR_ATARI_ENV_HPP_
#define ALE_VECTOR_ATARI_ENV_HPP_

#include <memory>
#include <vector>
#include <deque>
#include <random>
#include <string>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include "ale/common/Constants.h"
#include "ale/ale_interface.hpp"
#include "utils.hpp"

namespace ale::vector {

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
         * @param maxpool Whether to maxpool observations
         * @param obs_format Format of observations (grayscale or RGB)
         * @param stack_num Number of frames to stack for observations
         * @param noop_max Maximum number of no-ops to perform on resets
         * @param use_fire_reset Whether to press FIRE during reset
         * @param episodic_life Whether to end episodes when a life is lost
         * @param life_loss_info Whether to return `terminated=True` on a life loss but not reset until `lives==0`
         * @param reward_clipping Whether to clip the environment rewards between -1 and 1
         * @param max_episode_steps Maximum number of steps per episode before truncating
         * @param repeat_action_probability Probability of repeating the last action
         * @param full_action_space Whether to use the full action space
         * @param seed Random seed
         */
        PreprocessedAtariEnv(
            const int env_id,
            const fs::path &rom_path,
            const int obs_height = 84,
            const int obs_width = 84,
            const int frame_skip = 4,
            const bool maxpool = true,
            const ObsFormat obs_format = ObsFormat::Grayscale,
            const int stack_num = 4,
            const int noop_max = 30,
            const bool use_fire_reset = true,
            const bool episodic_life = false,
            const bool life_loss_info = false,
            const bool reward_clipping = true,
            const int max_episode_steps = 108000,
            const float repeat_action_probability = 0.0f,
            const bool full_action_space = false,
            const int seed = -1
        ) : env_id_(env_id),
            rom_path_(rom_path),
            obs_frame_height_(obs_height),
            obs_frame_width_(obs_width),
            frame_skip_(frame_skip),
            maxpool_(maxpool),
            obs_format_(obs_format),
            channels_per_frame_(obs_format == ObsFormat::Grayscale ? 1 : 3),
            stack_num_(stack_num),
            noop_max_(noop_max),
            use_fire_reset_(use_fire_reset),
            episodic_life_(episodic_life),
            life_loss_info_(life_loss_info),
            reward_clipping_(reward_clipping),
            max_episode_steps_(max_episode_steps),
            rng_gen_(seed == -1 ? std::random_device{}() : seed),
            elapsed_step_(max_episode_steps + 1),
            // Uninitialised variables
            game_over_(false), lives_(0), was_life_lost_(false), reward_(0),
            current_action_(EnvironmentAction()), current_seed_(0)
        {
            // Turn off verbosity
            Logger::setMode(Logger::Error);

            // Initialize ALE
            env_ = std::make_unique<ALEInterface>();
            env_->setFloat("repeat_action_probability", repeat_action_probability);
            env_->setInt("random_seed", seed);
            env_->loadROM(rom_path_);

            // Get action set
            if (full_action_space) {
                action_set_ = env_->getLegalActionSet();
            } else {
                action_set_ = env_->getMinimalActionSet();
            }

            // Check if fire action is available (needed for fire_reset)
            if (use_fire_reset_) {
                has_fire_action_ = false;
                for (const auto a: action_set_) {
                    if (a == PLAYER_A_FIRE) {
                        has_fire_action_ = true;
                        break;
                    }
                }
            }

            // Initialize random distribution for no-ops
            if (noop_max_ > 0) {
                noop_generator_ = std::uniform_int_distribution<>(0, noop_max_ - 1);
            } else {
                // If noop_max is 0, create a distribution that always returns 0
                noop_generator_ = std::uniform_int_distribution<>(0, 0);
            }

            const ALEScreen& screen = env_->getScreen();
            raw_frame_height_ = screen.height();
            raw_frame_width_ = screen.width();
            raw_frame_size_ = raw_frame_height_ * raw_frame_width_;
            raw_size_ = raw_frame_height_ * raw_frame_width_ * channels_per_frame_;
            obs_size_ = obs_frame_height_ * obs_frame_width_ * channels_per_frame_;

            // Initialize the buffers
            for (int i = 0; i < 2; ++i) {
                raw_frames_.emplace_back(raw_size_);
            }
            resized_frame_.resize(obs_size_, 0);
            for (int i = 0; i < stack_num_; ++i) {
                frame_stack_.push_back(std::vector<uint8_t>(obs_size_, 0));
            }
        }

        void set_seed(const int seed) {
            current_seed_ = seed;
        }

        /**
         * Reset the environment and return the initial observation
         */
        void reset() {
            if (current_seed_ >= 0) {
                env_->setInt("random_seed", current_seed_);
                rng_gen_.seed(current_seed_);

                env_->loadROM(rom_path_);
                current_seed_ = -1;
            }
            env_->reset_game();

            // Press FIRE if required by the environment
            if (use_fire_reset_ && has_fire_action_) {
                env_->act(PLAYER_A_FIRE);
            }

            // Perform no-op steps
            int noop_steps = noop_generator_(rng_gen_) - static_cast<int>(use_fire_reset_ && has_fire_action_);
            while (noop_steps > 0) {
                env_->act(PLAYER_A_NOOP);
                if (env_->game_over()) {
                    env_->reset_game();
                }
                noop_steps--;
            }

            // Get the screen data and process it
            if (obs_format_ == ObsFormat::Grayscale) {
                get_screen_data_grayscale(raw_frames_[0].data());
            } else {
                get_screen_data_rgb(raw_frames_[0].data());
            }
            std::fill(raw_frames_[1].begin(), raw_frames_[1].end(), 0);

            // Clear the frame stack
            for (int stack_id = 0; stack_id < stack_num_; ++stack_id) {
                std::fill(frame_stack_[stack_id].begin(), frame_stack_[stack_id].end(), 0);
            }
            process_screen();

            // Update state
            elapsed_step_ = 0;
            reward_ = 0;
            game_over_ = false;
            lives_ = env_->lives();
            was_life_lost_ = false;
            current_action_.action_id = PLAYER_A_NOOP;
        }

        /**
         * Set the action to be taken in the next step
         */
        void set_action(const EnvironmentAction& action) {
            current_action_ = action;
        }

        /**
         * Steps the environment using the current action
         */
        void step() {
            // Convert the current action to Action and Paddle Strength
            const int action_id = current_action_.action_id;
            if (action_id < 0 || action_id >= action_set_.size()) {
                throw std::out_of_range("Stepping sub-environment with action_id: " + std::to_string(action_id) + ", however, this is either less than zero or greater than available actions (" + std::to_string(action_set_.size()) + ")");
            }
            const Action action = action_set_[action_id];
            const float strength = current_action_.paddle_strength;

            // Execute action for frame_skip frames
            reward_t reward = 0;
            for (int skip_id = frame_skip_; skip_id > 0; --skip_id) {
                reward += env_->act(action, strength);

                game_over_ = env_->game_over();
                elapsed_step_++;
                was_life_lost_ = env_->lives() < lives_ && env_->lives() > 0;

                if (game_over_ || elapsed_step_ >= max_episode_steps_ || (episodic_life_ && was_life_lost_)) {
                    break;
                }

                // Captures last two frames for maxpooling
                if (skip_id <= 2) {
                    if (obs_format_ == ObsFormat::Grayscale) {
                        get_screen_data_grayscale(raw_frames_[skip_id - 1].data());
                    } else {
                        get_screen_data_rgb(raw_frames_[skip_id - 1].data());
                    }
                }
            }

            // Update state
            process_screen();
            lives_ = env_->lives();
            reward_ = reward_clipping_ ? std::clamp<int>(reward, -1, 1) : reward;
        }

        /**
         * Get the current observation
         */
        Timestep get_timestep() const {
            Timestep timestep;
            timestep.env_id = env_id_;

            timestep.reward = reward_;
            timestep.terminated = game_over_ || ((life_loss_info_ || episodic_life_) && was_life_lost_);
            timestep.truncated = elapsed_step_ >= max_episode_steps_ && !timestep.terminated;

            timestep.lives = lives_;
            timestep.frame_number = env_->getFrameNumber();
            timestep.episode_frame_number = env_->getEpisodeFrameNumber();

            // Combine stacked frames into a single observation
            timestep.observation.resize(obs_size_ * stack_num_);
            for (int i = 0; i < stack_num_; ++i) {
                std::memcpy(
                    timestep.observation.data() + i * obs_size_,
                    frame_stack_[i].data(),
                    obs_size_
                );
            }

            // Initialize as nullptr and set in AsyncVectorizer if needed
            timestep.final_observation = nullptr;

            return timestep;
        }

        /**
         * Check if the episode is over (terminated or truncated)
         */
        const bool is_episode_over() const {
            return game_over_ || elapsed_step_ >= max_episode_steps_ || (episodic_life_ && was_life_lost_);
        }

        /**
         * Get the list of available actions
         */
        const ActionVect& get_action_set() const {
            return action_set_;
        }

        /**
         * Get observation size
         */
        const int get_stacked_obs_size() const {
            return obs_size_ * stack_num_;
        }

        /**
         * Get channels per frame
         */
        const int get_channels_per_frame() const {
            return channels_per_frame_;
        }

    private:
        /**
         * Get the current screen data from ALE in grayscale format
         */
        void get_screen_data_grayscale(uint8_t* buffer) const {
            const ALEScreen& screen = env_->getScreen();
            uint8_t* ale_screen_data = screen.getArray();

            env_->theOSystem->colourPalette().applyPaletteGrayscale(
                buffer, ale_screen_data, raw_frame_size_
            );
        }

        /**
         * Get the current screen data from ALE in RGB format
         */
        void get_screen_data_rgb(uint8_t* buffer) const {
            const ALEScreen& screen = env_->getScreen();
            uint8_t* ale_screen_data = screen.getArray();

            env_->theOSystem->colourPalette().applyPaletteRGB(
                buffer, ale_screen_data, raw_frame_size_
            );
        }

        /**
         * Process the screen and update the frame stack
         */
        void process_screen() {
            // Maxpool raw frames if required (different for grayscale and RGB)
            if (maxpool_) {
                for (int i = 0; i < raw_size_; ++i) {
                    raw_frames_[0][i] = std::max(raw_frames_[0][i], raw_frames_[1][i]);
                }
            }

            // Resize the raw frame based on format
            if (obs_frame_height_ != raw_frame_height_ || obs_frame_width_ != raw_frame_width_) {
                auto cv2_format = (obs_format_ == ObsFormat::Grayscale) ? CV_8UC1 : CV_8UC3;
                cv::Mat src_img(raw_frame_height_, raw_frame_width_, cv2_format, raw_frames_[0].data());
                cv::Mat dst_img(obs_frame_height_, obs_frame_width_, cv2_format, resized_frame_.data());

                // Use INTER_AREA for downsampling to avoid moiré patterns
                cv::resize(src_img, dst_img, dst_img.size(), 0, 0, cv::INTER_AREA);
            } else {
                // No resize needed, just copy
                std::memcpy(resized_frame_.data(), raw_frames_[0].data(), raw_size_);
            }

            // Update frame stack - remove oldest frame and add new one
            frame_stack_.pop_front();
            frame_stack_.push_back(resized_frame_);
        }

        int env_id_;                         // Unique ID for this environment
        fs::path rom_path_;                  // Path to the ROM file
        std::unique_ptr<ALEInterface> env_;  // ALE interface

        ActionVect action_set_;              // Available actions

        ObsFormat obs_format_;               // Format of observations (grayscale or RGB)
        int channels_per_frame_;             // The number of channels for each frame based on obs_format
        int raw_frame_height_;               // The raw frame height
        int raw_frame_width_;                // The raw frame width
        int raw_frame_size_;                 // The raw frame size (height * width)
        int raw_size_;
        int obs_frame_height_;               // Height to resize frames to for observations
        int obs_frame_width_;                // Width to resize frames to for observations
        int obs_size_;                       // Observation size (height * width * channels)
        int stack_num_;                      // Number of frames to stack for observations

        int frame_skip_;                     // Number of frames for which to repeat the action
        bool maxpool_;                       // Whether to maxpool observations
        int noop_max_;                       // Maximum number of no-ops at reset
        bool use_fire_reset_;                // Whether to press FIRE during reset
        bool has_fire_action_;               // Whether FIRE action is available for reset
        bool episodic_life_;                 // Whether to end episodes when a life is lost
        bool life_loss_info_;                // If to provide termination signal (but not reset) on life loss
        bool reward_clipping_;               // If to clip rewards between -1 and 1
        int max_episode_steps_;              // Maximum number of steps per episode before truncating

        std::mt19937 rng_gen_;                             // Random number generator
        std::uniform_int_distribution<> noop_generator_;   // Distribution for no-op steps

        int elapsed_step_;                   // Current step in the episode
        bool game_over_;                     // Whether the game is over
        int lives_;                          // Current number of lives
        bool was_life_lost_;                 // If a life is loss from a step
        reward_t reward_;                    // Last reward received

        EnvironmentAction current_action_;   // Current action to take
        int current_seed_;                   // Current seed to update

        // Frame buffers
        std::vector<std::vector<uint8_t>> raw_frames_;  // Raw frame buffers for maxpooling
        std::vector<uint8_t> resized_frame_;            // Resized frame buffer
        std::deque<std::vector<uint8_t>> frame_stack_;  // Stack of recent frames
    };
}

#endif // ALE_VECTOR_ATARI_ENV_HPP_
