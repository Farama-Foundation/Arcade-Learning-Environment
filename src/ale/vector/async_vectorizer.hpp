#ifndef ALE_VECTOR_ASYNC_VECTORIZER_HPP_
#define ALE_VECTOR_ASYNC_VECTORIZER_HPP_

#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <array>
#include <cstdint>

#include "ThreadPool.h"
#include "utils.hpp"
#include "preprocessed_env.hpp"

#if defined(_WIN32) || defined(WIN32) || defined(_MSC_VER)
    #include <windows.h>
#endif

namespace ale {
namespace vector {

/**
 * AsyncVectorizer manages a collection of environments that can be stepped in parallel.
 * It handles the distribution of actions to environments and collection of observations.
 */
class AsyncVectorizer {
public:
    /**
     * Constructor for AsyncVectorizer
     *
     * @param num_envs The number of parallel environments to run
     * @param batch_size The number of environments to process in a batch (0 means use num_envs)
     * @param num_threads The number of worker threads to use (0 means use hardware concurrency)
     * @param thread_affinity_offset The CPU core offset for thread affinity (-1 means no affinity)
     * @param env_factory Function that creates environment instances
     */
    AsyncVectorizer(
        int num_envs,
        int batch_size = 0,
        int num_threads = 0,
        int thread_affinity_offset = -1,
        std::function<std::unique_ptr<PreprocessedAtariEnv>(int)> env_factory = nullptr
    ) : num_envs_(num_envs),
        batch_size_(batch_size <= 0 ? num_envs : batch_size),
        is_sync_(batch_size_ == num_envs_),
        stop_(false),
        stepping_env_num_(0),
        action_buffer_queue_(new ActionBufferQueue(num_envs_)),
        state_buffer_queue_(new StateBufferQueue(batch_size_, num_envs_)) {

        // Create environments
        envs_.resize(num_envs_);
        for (int i = 0; i < num_envs_; ++i) {
            envs_[i] = env_factory(i);
        }

        // Setup worker threads
        std::size_t processor_count = std::thread::hardware_concurrency();
        if (num_threads <= 0) {
            num_threads_ = min(batch_size_, static_cast<int>(processor_count));
        } else {
            num_threads_ = num_threads;
        }

        // Start worker threads
        for (int i = 0; i < num_threads_; ++i) {
            workers_.emplace_back([this] {
                worker_function();
            });
        }

        // Set thread affinity if requested
        if (thread_affinity_offset >= 0) {
            set_thread_affinity(thread_affinity_offset, processor_count);
        }
    }

    /**
     * Destructor - stops worker threads and cleans up resources
     */
    ~AsyncVectorizer() {
        stop_ = true;
        // Send empty actions to wake up and terminate all worker threads
        std::vector<ActionSlice> empty_actions(workers_.size());
        action_buffer_queue_->enqueue_bulk(empty_actions);
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    /**
     * Send actions to environments
     *
     * @param actions Vector of actions to send to environments
     */
    void send(const std::vector<Action>& actions) {
        std::vector<ActionSlice> action_slices;
        action_slices.reserve(actions.size());

        for (size_t i = 0; i < actions.size(); ++i) {
            int env_id = actions[i].env_id;
            envs_[env_id]->set_action(actions[i]);

            ActionSlice slice;
            slice.env_id = env_id;
            slice.order = is_sync_ ? static_cast<int>(i) : -1;
            slice.force_reset = false;
            action_slices.emplace_back(slice);
        }

        if (is_sync_) {
            stepping_env_num_ += actions.size();
        }

        action_buffer_queue_->enqueue_bulk(action_slices);
    }

    /**
     * Receive observations from environments
     *
     * @return Vector of observations from environments
     */
    std::vector<Observation> recv() {
        int additional_wait = 0;
        if (is_sync_ && stepping_env_num_ < batch_size_) {
            additional_wait = batch_size_ - stepping_env_num_;
        }

        std::vector<Observation> observations = state_buffer_queue_->wait(additional_wait);

        if (is_sync_) {
            stepping_env_num_ -= observations.size();
        }

        return observations;
    }

    /**
     * Reset specified environments
     *
     * @param env_ids Vector of environment IDs to reset
     */
    void reset(const std::vector<int>& env_ids) {
        std::vector<ActionSlice> reset_actions;
        reset_actions.reserve(env_ids.size());

        for (size_t i = 0; i < env_ids.size(); ++i) {
            ActionSlice slice;
            slice.env_id = env_ids[i];
            slice.order = is_sync_ ? static_cast<int>(i) : -1;
            slice.force_reset = true;
            reset_actions.emplace_back(slice);
        }

        if (is_sync_) {
            stepping_env_num_ += env_ids.size();
        }

        action_buffer_queue_->enqueue_bulk(reset_actions);
    }

private:
    /**
     * Worker thread function that processes environment steps
     */
    void worker_function() {
        while (!stop_) {
            try {
                ActionSlice action = action_buffer_queue_->dequeue();
                if (stop_) {
                    break;
                }

                int env_id = action.env_id;
                int order = action.order;
                bool reset = action.force_reset || envs_[env_id]->is_episode_over();

                if (reset) {
                    envs_[env_id]->reset();
                } else {
                    envs_[env_id]->step();
                }

                // Get observation and write to state buffer
                Observation obs = envs_[env_id]->get_observation();
                state_buffer_queue_->write(obs, order);

            } catch (const std::exception& e) {
                // Log error but continue processing
                std::cerr << "Error in worker thread: " << e.what() << std::endl;
            }
        }
    }

    /**
     * Set thread affinity for worker threads
     */
    void set_thread_affinity(int thread_affinity_offset, int processor_count) {
        for (size_t tid = 0; tid < workers_.size(); ++tid) {
            size_t core_id = (thread_affinity_offset + tid) % processor_count;

            #if defined(__linux__)
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(core_id, &cpuset);
                pthread_setaffinity_np(workers_[tid].native_handle(), sizeof(cpu_set_t), &cpuset);
            #elif defined(_WIN32)
                DWORD_PTR mask = (static_cast<DWORD_PTR>(1) << core_id);
                SetThreadAffinityMask(workers_[tid].native_handle(), mask);
            #elif defined(__APPLE__)
                thread_affinity_policy_data_t policy = { static_cast<integer_t>(core_id) };
                thread_port_t mach_thread = pthread_mach_thread_np(workers_[tid].native_handle());
                thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY,
                                (thread_policy_t)&policy, THREAD_AFFINITY_POLICY_COUNT);
            #endif
        }
    }

private:
    int num_envs_;                                    // Number of parallel environments
    int batch_size_;                                  // Batch size for processing
    int num_threads_;                                 // Number of worker threads
    bool is_sync_;                                    // Whether to operate in synchronous mode
    std::atomic<bool> stop_;                          // Signal to stop worker threads
    std::atomic<int> stepping_env_num_;               // Number of environments currently stepping
    std::vector<std::thread> workers_;                // Worker threads
    std::unique_ptr<ActionBufferQueue> action_buffer_queue_;  // Queue for actions
    std::unique_ptr<StateBufferQueue> state_buffer_queue_;    // Queue for observations
    std::vector<std::unique_ptr<PreprocessedAtariEnv>> envs_; // Environment instances
};

} // namespace vector
} // namespace ale

#endif // ALE_VECTOR_ASYNC_VECTORIZER_HPP_
