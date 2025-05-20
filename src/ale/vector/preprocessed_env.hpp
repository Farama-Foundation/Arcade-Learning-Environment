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
            obs_height_(obs_height),
            obs_width_(obs_width),
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
            game_over_(false), lives_(0), was_life_loss_(false), reward_(0),
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

            // Initialize the buffers
            for (int i = 0; i < 2; ++i) {
                raw_frames_.emplace_back(210 * 160 * channels_per_frame_);
            }
            const int frame_size = obs_height_ * obs_width_ * channels_per_frame_;
            resized_frame_.resize(frame_size, 0);
            for (int i = 0; i < stack_num_; ++i) {
                frame_stack_.push_back(std::vector<uint8_t>(frame_size, 0));
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

            // Perform no-op steps
            int noop_steps = noop_generator_(rng_gen_) - static_cast<int>(use_fire_reset_ && has_fire_action_);
            while (noop_steps > 0) {
                env_->act(PLAYER_A_NOOP);
                if (env_->game_over()) {
                    env_->reset_game();
                }
                noop_steps--;
            }

            // Press FIRE if required by the environment
            if (use_fire_reset_ && has_fire_action_) {
                env_->act(PLAYER_A_FIRE);
            }

            // Get the screen data and process it
            if (obs_format_ == ObsFormat::Grayscale) {
                get_screen_data_grayscale(raw_frames_[0].data());
            } else {
                get_screen_data_rgb(raw_frames_[0].data());
            }
            std::fill(raw_frames_[1].begin(), raw_frames_[1].end(), 0);

            // Clear the frame stack
            for (int stack_id = 0; stack_id < stack_num_ - 1; ++stack_id) {
                std::fill(frame_stack_[stack_id].begin(), frame_stack_[stack_id].end(), 0);
            }
            process_screen();

            // Update state
            elapsed_step_ = 0;
            game_over_ = false;
            lives_ = env_->lives();
            was_life_loss_ = false;
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
            float reward = 0.0;
            game_over_ = false;

            const int action_id = current_action_.action_id;
            const Action action = (action_id < 0 || action_id >= static_cast<int>(action_set_.size())) ? PLAYER_A_NOOP : action_set_[action_id];
            const float strength = current_action_.paddle_strength;

            // Execute action for frame_skip frames
            for (int skip_id = frame_skip_; skip_id > 0 && !game_over_; --skip_id) {
                reward += env_->act(action, strength);

                game_over_ = env_->game_over();
                // Handle episodic life
                if (episodic_life_ && env_->lives() < lives_ && env_->lives() > 0) {
                    game_over_ = true;
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

            // Process the screen
            process_screen();

            // Update state
            elapsed_step_++;
            was_life_loss_ = env_->lives() < lives_;
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
            timestep.terminated = game_over_ || (life_loss_info_ && was_life_loss_);
            timestep.truncated = elapsed_step_ >= max_episode_steps_;

            timestep.lives = lives_;
            timestep.frame_number = env_->getFrameNumber();
            timestep.episode_frame_number = env_->getEpisodeFrameNumber();

            // Combine stacked frames into a single observation
            const size_t frame_size = obs_height_ * obs_width_ * channels_per_frame_;
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

        /**
         * Get observation size
         */
        int get_obs_size() const {
            return obs_height_ * obs_width_ * channels_per_frame_ * stack_num_;
        }

        /**
         * Get channels per frame
         */
        int get_channels_per_frame() const {
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
                buffer, ale_screen_data, screen.width() * screen.height()
            );
        }

        /**
         * Get the current screen data from ALE in RGB format
         */
        void get_screen_data_rgb(uint8_t* buffer) const {
            const ALEScreen& screen = env_->getScreen();
            uint8_t* ale_screen_data = screen.getArray();

            env_->theOSystem->colourPalette().applyPaletteRGB(
                buffer, ale_screen_data, screen.width() * screen.height()
            );
        }

        /**
         * Process the screen and update the frame stack
         */
        void process_screen() {
            int raw_height = 210;
            int raw_width = 160;

            // Maxpool raw frames if required (different for grayscale and RGB)
            if (maxpool_) {
                int raw_size = raw_height * raw_width * channels_per_frame_;
                for (int i = 0; i < raw_size; ++i) {
                    raw_frames_[0][i] = std::max(raw_frames_[0][i], raw_frames_[1][i]);
                }
            }

            // Resize the raw frame based on format
            if (obs_height_ != raw_height || obs_width_ != raw_width) {
                auto cv2_format = (obs_format_ == ObsFormat::Grayscale) ? CV_8UC1 : CV_8UC3;
                cv::Mat src_img(raw_height, raw_width, cv2_format, raw_frames_[0].data());
                cv::Mat dst_img(obs_height_, obs_width_, cv2_format, resized_frame_.data());

                // Use INTER_AREA for downsampling to avoid moiré patterns
                cv::resize(src_img, dst_img, dst_img.size(), 0, 0, cv::INTER_AREA);
            } else {
                // No resize needed, just copy
                int original_shape = raw_height * raw_width * channels_per_frame_;
                std::memcpy(resized_frame_.data(), raw_frames_[0].data(), original_shape);
            }

            // Update frame stack - remove oldest frame and add new one
            frame_stack_.pop_front();
            frame_stack_.push_back(resized_frame_);
        }

        int env_id_;                         // Unique ID for this environment
        fs::path rom_path_;                  // Path to the ROM file
        std::unique_ptr<ALEInterface> env_;  // ALE interface

        ActionVect action_set_;              // Available actions
        int obs_height_;                     // Height to resize frames to for observations
        int obs_width_;                      // Width to resize frames to for observations
        int frame_skip_;                     // Number of frames for which to repeat the action
        bool maxpool_;                       // Whether to maxpool observations
        ObsFormat obs_format_;               // Format of observations (grayscale or RGB)
        int channels_per_frame_;             // The number of channels for each frame based on obs_format
        int stack_num_;                      // Number of frames to stack for observations
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
        bool was_life_loss_;                 // If a life is loss from a step
        float reward_;                       // Last reward received

        EnvironmentAction current_action_;   // Current action to take
        int current_seed_;                   // Current seed to update

        // Frame buffers
        std::vector<std::vector<uint8_t>> raw_frames_;  // Raw frame buffers for maxpooling
        std::vector<uint8_t> resized_frame_;            // Resized frame buffer
        std::deque<std::vector<uint8_t>> frame_stack_;  // Stack of recent frames
    };
}

#endif // ALE_VECTOR_ATARI_ENV_HPP_
