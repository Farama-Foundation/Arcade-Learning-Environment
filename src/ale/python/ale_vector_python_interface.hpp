#ifndef ALE_VECTOR_INTERFACE_HPP_
#define ALE_VECTOR_INTERFACE_HPP_

#include <string>
#include <vector>
#include <memory>
#include <random>
#include <filesystem>

#include "ale/vector/async_vectorizer.hpp"
#include "ale/vector/preprocessed_env.hpp"
#include "ale/vector/utils.hpp"

namespace fs = std::filesystem;

namespace ale {
namespace vector {

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
     * @param num_envs Number of parallel environments
     * @param rom_path Path to the ROM file
     * @param frame_skip Number of frames to skip between agent decisions (default: 4)
     * @param gray_scale Whether to convert frames to grayscale (default: true)
     * @param stack_num Number of frames to stack for observations (default: 4)
     * @param img_height Height to resize frames to (default: 84)
     * @param img_width Width to resize frames to (default: 84)
     * @param noop_max Maximum number of no-ops to perform at reset (default: 30)
     * @param fire_reset Whether to press FIRE during reset (default: true)
     * @param episodic_life Whether to end episodes when a life is lost (default: false)
     * @param max_episode_steps Maximum number of steps per episode (default: 108000)
     * @param repeat_action_probability Probability of repeating the last action (default: 0.0f)
     * @param full_action_space Whether to use the full action space (default: false)
     * @param batch_size The number of environments to process in a batch (0 means use num_envs, default: 0)
     * @param num_threads The number of worker threads to use (0 means use hardware concurrency, default: 0)
     * @param seed Random seed (default: 0)
     * @param thread_affinity_offset The CPU core offset for thread affinity (-1 means no affinity, default: -1)
     */
    ALEVectorInterface(
        int num_envs,
        const std::string& rom_path,
        int frame_skip = 4,
        bool gray_scale = true,
        int stack_num = 4,
        int img_height = 84,
        int img_width = 84,
        int noop_max = 30,
        bool fire_reset = true,
        bool episodic_life = false,
        int max_episode_steps = 108000,
        float repeat_action_probability = 0.0f,
        bool full_action_space = false,
        int batch_size = 0,
        int num_threads = 0,
        int seed = 0,
        int thread_affinity_offset = -1
    ) : num_envs_(num_envs),
        rom_path_(rom_path),
        frame_skip_(frame_skip),
        gray_scale_(gray_scale),
        stack_num_(stack_num),
        img_height_(img_height),
        img_width_(img_width),
        noop_max_(noop_max),
        fire_reset_(fire_reset),
        episodic_life_(episodic_life),
        max_episode_steps_(max_episode_steps),
        repeat_action_probability_(repeat_action_probability),
        full_action_space_(full_action_space),
        seed_(seed),
        gen_(seed) {

        // Create environment factory
        auto env_factory = [this](int env_id) {
            return std::make_unique<PreprocessedAtariEnv>(
                env_id,
                rom_path_,
                frame_skip_,
                gray_scale_,
                stack_num_,
                img_height_,
                img_width_,
                noop_max_,
                fire_reset_,
                episodic_life_,
                max_episode_steps_,
                repeat_action_probability_,
                full_action_space_,
                seed_ + env_id
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

        // Initialize action set (assuming all environments have the same action set)
        auto temp_env = env_factory(0);
        action_set_ = temp_env->get_action_set();
    }

    /**
     * Reset all environments
     *
     * @return Observations from all environments after reset
     */
    std::vector<Observation> reset() {
        std::vector<int> env_ids(num_envs_);
        for (int i = 0; i < num_envs_; ++i) {
            env_ids[i] = i;
        }
        vectorizer_->reset(env_ids);
        return vectorizer_->recv();
    }

    /**
     * Step environments with actions
     *
     * @param actions Vector of actions to take in environments
     * @return Observations from environments after stepping
     */
    std::vector<Observation> step(const std::vector<Action>& actions) {
        vectorizer_->send(actions);
        return vectorizer_->recv();
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
     * @return Tuple of (stack_num, channels, height, width)
     */
    std::tuple<int, int, int, int> get_observation_shape() const {
        const int channels = gray_scale_ ? 1 : 3;
        return std::make_tuple(stack_num_, channels, img_height_, img_width_);
    }

private:
    int num_envs_;                            // Number of parallel environments
    std::string rom_path_;                    // Path to the ROM file
    int frame_skip_;                          // Number of frames to skip
    bool gray_scale_;                         // Whether to use grayscale
    int stack_num_;                           // Number of frames to stack
    int img_height_;                          // Height of resized frames
    int img_width_;                           // Width of resized frames
    int noop_max_;                            // Max no-ops on reset
    bool fire_reset_;                         // Whether to fire on reset
    bool episodic_life_;                      // End episode on life loss
    int max_episode_steps_;                   // Max steps per episode
    float repeat_action_probability_;         // Sticky actions probability
    bool full_action_space_;                  // Use full action space
    int seed_;                                // Random seed
    std::mt19937 gen_;                        // Random number generator

    std::unique_ptr<AsyncVectorizer> vectorizer_;  // Vectorizer
    ActionVect action_set_;                  // Set of available actions
};

} // namespace vector
} // namespace ale

#endif // ALE_VECTOR_INTERFACE_HPP_
