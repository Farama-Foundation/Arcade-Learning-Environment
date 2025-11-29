#include "env_vectorizer.hpp"

#if defined(__linux__)
    #include <pthread.h>
#elif defined(_WIN32)
    #include <windows.h>
#elif defined(__APPLE__)
    #include <mach/thread_policy.h>
    #include <mach/thread_act.h>
    #include <pthread.h>
#endif

namespace ale::vector {

EnvVectorizer::EnvVectorizer(
    const fs::path& rom_path,
    int num_envs,
    int batch_size,
    int num_threads,
    int thread_affinity_offset,
    AutoresetMode autoreset_mode,
    int img_height,
    int img_width,
    int stack_num,
    bool grayscale,
    int frame_skip,
    bool maxpool,
    int noop_max,
    bool use_fire_reset,
    bool episodic_life,
    bool life_loss_info,
    bool reward_clipping,
    int max_episode_steps,
    float repeat_action_probability,
    bool full_action_space
) : num_envs_(num_envs),
    batch_size_(batch_size > 0 ? batch_size : num_envs),
    img_height_(img_height),
    img_width_(img_width),
    stack_num_(stack_num),
    grayscale_(grayscale),
    autoreset_mode_(autoreset_mode),
    last_recv_env_ids_(batch_size_ > 0 ? batch_size_ : num_envs)
{
    // Create environments
    envs_.reserve(num_envs_);
    for (int i = 0; i < num_envs_; ++i) {
        envs_.push_back(std::make_unique<PreprocessedEnv>(
            i, rom_path, img_height, img_width, frame_skip, maxpool,
            grayscale, stack_num, noop_max, use_fire_reset, episodic_life,
            life_loss_info, reward_clipping, max_episode_steps,
            repeat_action_probability, full_action_space, -1
        ));
    }

    stacked_obs_size_ = envs_[0]->stacked_obs_size();
    action_set_ = envs_[0]->action_set();

    // Create action queue (capacity = 2x num_envs for safety)
    action_queue_ = std::make_unique<ActionQueue>(num_envs_ * 2);

    // Create result staging
    bool same_step = (autoreset_mode_ == AutoresetMode::SameStep);
    staging_ = std::make_unique<ResultStaging>(batch_size_, num_envs_, stacked_obs_size_, same_step);

    // Determine thread count
    int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (num_threads <= 0) {
        num_threads_ = std::min(batch_size_, hw_threads);
    } else {
        num_threads_ = std::min(num_threads, hw_threads);
    }

    // Start worker threads
    workers_.reserve(num_threads_);
    for (int i = 0; i < num_threads_; ++i) {
        workers_.emplace_back([this, i] { worker_loop(i); });
    }

    // Set thread affinity if requested
    if (thread_affinity_offset >= 0) {
        set_thread_affinity(thread_affinity_offset);
    }
}

EnvVectorizer::~EnvVectorizer() {
    stop_.store(true);

    // Send dummy actions to wake up blocked workers
    std::vector<Action> wake_actions(workers_.size());
    for (auto& a : wake_actions) {
        a.env_id = 0;
        a.force_reset = false;
    }
    action_queue_->enqueue_bulk(wake_actions);

    // Join all workers
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

BatchResult EnvVectorizer::reset(const std::vector<int>& env_ids, const std::vector<int>& seeds) {
    if (env_ids.size() != seeds.size()) {
        throw std::invalid_argument("env_ids and seeds must have same size");
    }

    // Release slots from previous batch (but not on first batch)
    if (!first_batch_ && !staging_->is_ordered()) {
        // For unordered mode, we need to handle this carefully
        // The staging buffer was already released in recv()
    }
    first_batch_ = false;

    // Set seeds and prepare actions
    std::vector<Action> actions;
    actions.reserve(env_ids.size());
    for (std::size_t i = 0; i < env_ids.size(); ++i) {
        int env_id = env_ids[i];
        envs_[env_id]->set_seed(seeds[i]);

        Action action;
        action.env_id = env_id;
        action.action_id = 0;
        action.paddle_strength = 1.0f;
        action.force_reset = true;
        actions.push_back(action);
    }

    // Enqueue reset actions
    action_queue_->enqueue_bulk(actions);

    // Wait for results
    return recv();
}

void EnvVectorizer::send(const std::vector<Action>& actions) {
    if (actions.size() != static_cast<std::size_t>(batch_size_)) {
        throw std::invalid_argument(
            "Expected " + std::to_string(batch_size_) + " actions, got " + std::to_string(actions.size())
        );
    }

    // Map actions to correct environments using last_recv_env_ids
    std::vector<Action> mapped_actions;
    mapped_actions.reserve(actions.size());

    for (std::size_t i = 0; i < actions.size(); ++i) {
        Action mapped = actions[i];
        int actual_env_id = last_recv_env_ids_[i];
        mapped.env_id = actual_env_id;
        mapped.force_reset = false;

        // Set action on environment
        envs_[actual_env_id]->set_action(mapped.action_id, mapped.paddle_strength);

        mapped_actions.push_back(mapped);
    }

    // Enqueue actions
    action_queue_->enqueue_bulk(mapped_actions);
}

BatchResult EnvVectorizer::recv() {
    // Wait for batch to complete
    staging_->wait_for_batch();

    // Check for errors
    check_error();

    // Release batch and get results
    auto result = staging_->release_batch();

    // Remember env_ids for next send()
    std::memcpy(last_recv_env_ids_.data(), result.env_ids_data(), batch_size_ * sizeof(int));

    return result;
}

void EnvVectorizer::worker_loop(int thread_id) {
    (void)thread_id;  // For potential future use (logging, etc.)

    while (!stop_.load()) {
        try {
            Action action = action_queue_->dequeue();

            if (stop_.load()) {
                break;
            }

            execute_env(action);

        } catch (...) {
            set_error(std::current_exception());
        }
    }
}

void EnvVectorizer::execute_env(const Action& action) {
    int env_id = action.env_id;
    auto& env = *envs_[env_id];

    if (autoreset_mode_ == AutoresetMode::NextStep) {
        // NextStep mode: reset happens before step if episode was over
        if (action.force_reset || env.is_episode_over()) {
            env.reset();
        } else {
            env.step();
        }

        // Stage result
        staging_->stage_result(env_id, [&](OutputSlot& slot) {
            env.write_to(slot);
        });

    } else {  // SameStep mode
        if (action.force_reset) {
            env.reset();

            staging_->stage_result(env_id, [&](OutputSlot& slot) {
                env.write_to(slot);
            });

        } else {
            env.step();

            staging_->stage_result(env_id, [&](OutputSlot& slot) {
                if (env.is_episode_over()) {
                    // Write final observation before reset
                    env.write_obs_to(slot.final_obs);

                    // Capture pre-reset metadata
                    env.write_to(slot);
                    int pre_reward = *slot.reward;
                    bool pre_terminated = *slot.terminated;
                    bool pre_truncated = *slot.truncated;

                    // Reset and write new observation
                    env.reset();
                    env.write_to(slot);

                    // Restore pre-reset reward/terminated/truncated
                    *slot.reward = pre_reward;
                    *slot.terminated = pre_terminated;
                    *slot.truncated = pre_truncated;
                } else {
                    env.write_to(slot);
                }
            });
        }
    }
}

void EnvVectorizer::set_thread_affinity(int thread_affinity_offset) {
    int processor_count = static_cast<int>(std::thread::hardware_concurrency());

    for (std::size_t i = 0; i < workers_.size(); ++i) {
        int core_id = (thread_affinity_offset + static_cast<int>(i)) % processor_count;

#if defined(__linux__)
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);
        pthread_setaffinity_np(workers_[i].native_handle(), sizeof(cpu_set_t), &cpuset);
#elif defined(_WIN32)
        DWORD_PTR mask = (static_cast<DWORD_PTR>(1) << core_id);
        SetThreadAffinityMask(workers_[i].native_handle(), mask);
#elif defined(__APPLE__)
        thread_affinity_policy_data_t policy = { static_cast<integer_t>(core_id) };
        thread_port_t mach_thread = pthread_mach_thread_np(workers_[i].native_handle());
        thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY,
                         (thread_policy_t)&policy, THREAD_AFFINITY_POLICY_COUNT);
#endif
    }
}

void EnvVectorizer::set_error(std::exception_ptr e) {
    std::lock_guard<std::mutex> lock(error_mutex_);
    if (!has_error_.load()) {
        error_ = e;
        has_error_.store(true);
    }
}

void EnvVectorizer::check_error() {
    if (has_error_.load()) {
        std::lock_guard<std::mutex> lock(error_mutex_);
        if (error_) {
            std::rethrow_exception(error_);
        }
    }
}

}  // namespace ale::vector
