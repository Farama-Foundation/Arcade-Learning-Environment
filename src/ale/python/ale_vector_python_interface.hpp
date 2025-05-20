#ifndef ALE_VECTOR_INTERFACE_HPP_
#define ALE_VECTOR_INTERFACE_HPP_

#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include <stdexcept>

#include "ale/vector/async_vectorizer.hpp"
#include "ale/vector/preprocessed_env.hpp"
#include "ale/vector/utils.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
namespace fs = std::filesystem;

namespace ale::vector {

    /**
     * ALEVectorInterface provides a vectorized interface to the Arcade Learning Environment.
     * It manages multiple Atari environments running in parallel and allows sending actions
     * and receiving observations in batches.
     */
    class ALEVectorInterface {
    public:
        /**
         * Constructor
         *
         * @param rom_path Path to the ROM file
         * @param num_envs Number of parallel environments
         * @param frame_skip Number of frames to skip between agent decisions (default: 4)
         * @param stack_num Number of frames to stack for observations (default: 4)
         * @param img_height Height to resize frames to (default: 84)
         * @param img_width Width to resize frames to (default: 84)
         * @param grayscale Whether to use grayscale observations (default: true)
         * @param maxpool If to maxpool over frames (default: true)
         * @param noop_max Maximum number of no-ops to perform at reset (default: 30)
         * @param use_fire_reset Whether to press FIRE during reset (default: true)
         * @param episodic_life Whether to end episodes when a life is lost (default: false)
         * @param life_loss_info Whether to return `terminated=True` on a life loss but not reset until `lives==0`
         * @param reward_clipping Whether to clip the environment rewards between -1 and 1
         * @param max_episode_steps Maximum number of steps per episode (default: 108000)
         * @param repeat_action_probability Probability of repeating the last action (default: 0.0f)
         * @param full_action_space Whether to use the full action space (default: false)
         * @param batch_size The number of environments to process in a batch (0 means use num_envs, default: 0)
         * @param num_threads The number of worker threads to use (0 means use hardware concurrency, default: 0)
         * @param thread_affinity_offset The CPU core offset for thread affinity (-1 means no affinity, default: -1)
         */
        ALEVectorInterface(
            const fs::path &rom_path,
            int num_envs,
            const int frame_skip = 4,
            const int stack_num = 4,
            const int img_height = 84,
            const int img_width = 84,
            const bool grayscale = true,
            const bool maxpool = true,
            const int noop_max = 30,
            const bool use_fire_reset = true,
            const bool episodic_life = false,
            const bool life_loss_info = false,
            const bool reward_clipping = true,
            const int max_episode_steps = 108000,
            const float repeat_action_probability = 0.0f,
            const bool full_action_space = false,
            int batch_size = 0,
            int num_threads = 0,
            int thread_affinity_offset = -1
        ) : rom_path_(rom_path),
            num_envs_(num_envs),
            frame_skip_(frame_skip),
            stack_num_(stack_num),
            img_height_(img_height),
            img_width_(img_width),
            grayscale_(grayscale),
            obs_format_(grayscale_ ? ObsFormat::Grayscale : ObsFormat::RGB),
            maxpool_(maxpool),
            noop_max_(noop_max),
            use_fire_reset_(use_fire_reset),
            episodic_life_(episodic_life),
            life_loss_info_(life_loss_info),
            reward_clipping_(reward_clipping),
            max_episode_steps_(max_episode_steps),
            repeat_action_probability_(repeat_action_probability),
            full_action_space_(full_action_space),
            received_env_ids_(batch_size > 0 ? batch_size : num_envs) {

            // Create environment factory
            auto env_factory = [this](int env_id) {
                return std::make_unique<PreprocessedAtariEnv>(
                    env_id,
                    rom_path_,
                    img_height_,
                    img_width_,
                    frame_skip_,
                    maxpool_,
                    obs_format_,
                    stack_num_,
                    noop_max_,
                    use_fire_reset_,
                    episodic_life_,
                    life_loss_info_,
                    reward_clipping_,
                    max_episode_steps_,
                    repeat_action_probability_,
                    full_action_space_,
                    -1
                );
            };

            // Create vectorizer
            vectorizer_ = std::make_unique<AsyncVectorizer>(
                num_envs,
                batch_size,
                num_threads,
                thread_affinity_offset,
                env_factory
            );

            // Initialize the action set (assuming all environments have the same action set)
            const auto temp_env = env_factory(0);
            action_set_ = temp_env->get_action_set();
        }

        /**
         * Reset all environments
         *
         * @param reset_indices Vector of environment indices to be reset
         * @param reset_seeds Vector of environment seeds to use
         * @return Timesteps from all environments after reset
         */
        std::vector<Timestep> reset(const std::vector<int> &reset_indices, const std::vector<int> &reset_seeds) {
            vectorizer_->reset(reset_indices, reset_seeds);
            return recv();
        }

        /**
         * Step environments with actions
         *
         * @param action_ids Vector of actions ids to take
         * @param paddle_strengths Vector of paddle strengths to take
         */
        void send(const std::vector<int>& action_ids, const std::vector<float>& paddle_strengths) const {
            if (action_ids.size() != paddle_strengths.size()) {
                throw std::invalid_argument(
                    "The size of the action_ids is different from the paddle_strengths, action_ids length=" + std::to_string(action_ids.size())
                    + ", paddle_strengths length=" + std::to_string(paddle_strengths.size()));
            }
            std::vector<EnvironmentAction> environment_actions;
            environment_actions.resize(action_ids.size());

            for (int i = 0; i < action_ids.size(); i++) {
                EnvironmentAction env_action;
                env_action.env_id = received_env_ids_[i];
                env_action.action_id = action_ids[i];
                env_action.paddle_strength = paddle_strengths[i];

                environment_actions[i] = env_action;
            }

            vectorizer_->send(environment_actions);
        }

        /**
        * Returns the environment's data for the environments
        */
        std::vector<Timestep> recv() {
            std::vector<Timestep> timesteps = vectorizer_->recv();
            for (int i = 0; i < timesteps.size(); i++) {
                received_env_ids_[i] = timesteps[i].env_id;
            }
            return timesteps;
        }

        /**
         * Get the available actions for the environments
         *
         * @return Vector of available actions
         */
        const ActionVect& get_action_set() const {
            return action_set_;
        }

        /**
         * Get the number of environments
         *
         * @return Number of environments
         */
        int get_num_envs() const {
            return num_envs_;
        }

        /**
         * Get the dimensions of the observation space
         *
         * @return Tuple of (stack_num, height, width, 0) if grayscale or (stack_num, height, width, 3) if RGB
         */
        std::tuple<int, int, int, int> get_observation_shape() const {
            if (grayscale_) {
                return std::make_tuple(stack_num_, img_height_, img_width_, 0);
            } else {
                return std::make_tuple(stack_num_, img_height_, img_width_, 3);
            }
        }

        /**
         * Check if observations are grayscale
         *
         * @return true if observations are grayscale, false if RGB
         */
        bool is_grayscale() const {
            return grayscale_;
        }

        /**
         * Get the underlying vectorizer
         *
         * @return pointer for the underlying vectorizer
         */
        AsyncVectorizer* get_vectorizer() const {
            return vectorizer_.get();
        }

    private:
        fs::path rom_path_;                       // Path to the ROM file
        int num_envs_;                            // Number of parallel environments
        int frame_skip_;                          // Number of frames to skip
        int stack_num_;                           // Number of frames to stack
        int img_height_;                          // Height of resized frames
        int img_width_;                           // Width of resized frames
        bool grayscale_;                          // Whether to use grayscale observations
        ObsFormat obs_format_;                    // Observation format based on grayscale
        bool maxpool_;                            // If to maxpool over frames
        int noop_max_;                            // Max no-ops on reset
        bool use_fire_reset_;                     // Whether to fire on reset
        bool episodic_life_;                      // End episode on life loss
        bool life_loss_info_;                     // If to provide a termination signal (but not reset) on life loss
        bool reward_clipping_;                    // If to clip rewards between -1 and 1
        int max_episode_steps_;                   // Max steps per episode
        float repeat_action_probability_;         // Repeat actions probability for sticky actions
        bool full_action_space_;                  // Use full action space

        std::vector<int> received_env_ids_;        // Vector of environment ids for the most recently received data

        std::unique_ptr<AsyncVectorizer> vectorizer_;  // Vectorizer
        ActionVect action_set_;                    // Set of available actions
    };
}

#endif // ALE_VECTOR_INTERFACE_HPP_
