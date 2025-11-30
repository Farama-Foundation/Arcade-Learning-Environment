#ifndef ALE_VECTOR_ENV_VECTORIZER_HPP_
#define ALE_VECTOR_ENV_VECTORIZER_HPP_

#include <vector>
#include <thread>
#include <memory>
#include <atomic>
#include <mutex>
#include <exception>
#include <filesystem>

#include "ale/common/Constants.h"
#include "types.hpp"
#include "action_queue.hpp"
#include "result_staging.hpp"
#include "preprocessed_env.hpp"

namespace fs = std::filesystem;

namespace ale::vector {

class EnvVectorizer {
public:
    EnvVectorizer(
        const fs::path& rom_path,
        int num_envs,
        int batch_size = 0,
        int num_threads = 0,
        int thread_affinity_offset = -1,
        AutoresetMode autoreset_mode = AutoresetMode::NextStep,
        int img_height = 84,
        int img_width = 84,
        int stack_num = 4,
        bool grayscale = true,
        int frame_skip = 4,
        bool maxpool = true,
        int noop_max = 30,
        bool use_fire_reset = true,
        bool episodic_life = false,
        bool life_loss_info = false,
        bool reward_clipping = true,
        int max_episode_steps = 108000,
        float repeat_action_probability = 0.0f,
        bool full_action_space = false
    );

    ~EnvVectorizer();

    // Non-copyable, non-movable
    EnvVectorizer(const EnvVectorizer&) = delete;
    EnvVectorizer& operator=(const EnvVectorizer&) = delete;

    /// Reset specified environments with given seeds.
    /// @param env_ids Environment indices to reset
    /// @param seeds Seeds for each environment (-1 to keep current seed)
    /// @return Batch of results from batch_size environments
    BatchResult reset(const std::vector<int>& env_ids, const std::vector<int>& seeds);

    /// Send actions to environments.
    /// actions[i] applies to the environment that was at position i in the last recv() result.
    /// @param actions Actions with env_id, action_id, paddle_strength
    void send(const std::vector<Action>& actions);

    /// Receive results from batch_size environments.
    /// In ordered mode: returns results for all envs in order
    /// In unordered mode: returns results from first batch_size envs to complete
    BatchResult recv();

    // Accessors
    int num_envs() const { return num_envs_; }
    int batch_size() const { return batch_size_; }
    std::size_t stacked_obs_size() const { return stacked_obs_size_; }
    const ActionVect& action_set() const { return action_set_; }
    AutoresetMode autoreset_mode() const { return autoreset_mode_; }
    bool is_grayscale() const { return grayscale_; }

    /// Get observation shape as tuple (stack_num, height, width) or (stack_num, height, width, 3)
    std::tuple<int, int, int, int> observation_shape() const {
        return grayscale_
            ? std::make_tuple(stack_num_, img_height_, img_width_, 0)
            : std::make_tuple(stack_num_, img_height_, img_width_, 3);
    }

    /// Get raw pointer for JAX FFI handle
    const void* handle() const { return this; }

private:
    // Configuration
    int num_envs_;
    int batch_size_;
    int num_threads_;
    int img_height_;
    int img_width_;
    int stack_num_;
    bool grayscale_;
    std::size_t stacked_obs_size_;
    AutoresetMode autoreset_mode_;

    // Environments
    std::vector<std::unique_ptr<PreprocessedEnv>> envs_;
    ActionVect action_set_;

    // Worker threads
    std::vector<std::thread> workers_;
    std::atomic<bool> stop_{false};

    // Work distribution and result collection
    std::unique_ptr<ActionQueue> action_queue_;
    std::unique_ptr<ResultStaging> staging_;

    // Maps batch position -> env_id for last recv() result
    std::vector<int> last_recv_env_ids_;

    // Error handling
    std::atomic<bool> has_error_{false};
    std::exception_ptr error_;
    std::mutex error_mutex_;

    // Track first batch for slot release
    bool first_batch_{true};

    /// Worker thread main loop
    void worker_loop(int thread_id);

    /// Execute one environment step or reset
    void execute_env(const Action& action);

    /// Set thread CPU affinity
    void set_thread_affinity(int thread_affinity_offset);

    /// Record an error from a worker thread
    void set_error(std::exception_ptr e);

    /// Check if an error occurred and rethrow
    void check_error();
};

}  // namespace ale::vector

#endif  // ALE_VECTOR_ENV_VECTORIZER_HPP_
