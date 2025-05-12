#ifndef ALE_VECTOR_ASYNC_VECTORIZER_HPP_
#define ALE_VECTOR_ASYNC_VECTORIZER_HPP_

#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <functional>
#include <mutex>

#include "ale/external/ThreadPool.h"
#include "utils.hpp"
#include "preprocessed_env.hpp"

#if defined(_WIN32) || defined(WIN32) || defined(_MSC_VER)
    #include <windows.h>
#endif

namespace ale::vector {
    /**
     * AsyncVectorizer manages a collection of environments that can be stepped in parallel.
     * It handles the (async) distribution of actions to environments and collection of observations.
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
        explicit AsyncVectorizer(
            const int num_envs,
            const int batch_size = 0,
            const int num_threads = 0,
            const int thread_affinity_offset = -1,
            const std::function<std::unique_ptr<PreprocessedAtariEnv>(int)> &env_factory = nullptr
        ) : num_envs_(num_envs),
            batch_size_(batch_size > 0 ? batch_size : num_envs),
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
            obs_size_ = envs_[0]->get_obs_size();

            // Setup worker threads
            const std::size_t processor_count = std::thread::hardware_concurrency();
            if (num_threads <= 0) {
                num_threads_ = std::min<int>(batch_size_, static_cast<int>(processor_count));
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
            const std::vector<ActionSlice> empty_actions(workers_.size());
            action_buffer_queue_->enqueue_bulk(empty_actions);
            for (auto& worker : workers_) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
        }

        /**
         * Reset specified environments
         *
         * @param reset_indices Vector of environment IDs to reset
         * @param seeds Vector of seeds to use on reset (use -1 to not change the environment's seed)
         */
        void reset(const std::vector<int>& reset_indices, const std::vector<int>& seeds) {
            std::vector<ActionSlice> reset_actions;
            reset_actions.reserve(reset_indices.size());

            for (size_t i = 0; i < reset_indices.size(); ++i) {
                const int env_id = reset_indices[i];
                envs_[env_id]->set_seed(seeds[i]);

                ActionSlice action;
                action.env_id = env_id;
                action.order = is_sync_ ? static_cast<int>(i) : -1;
                action.force_reset = true;

                reset_actions.emplace_back(action);
            }

            if (is_sync_) {
                stepping_env_num_ += reset_indices.size();
            }

            action_buffer_queue_->enqueue_bulk(reset_actions);
        }

        /**
         * Send actions to the sub-environments
         *
         * @param actions Vector of actions to send to the sub-environments
         */
        void send(const std::vector<EnvironmentAction>& actions) {
            std::vector<ActionSlice> action_slices;
            action_slices.reserve(actions.size());

            for (int i = 0; i < actions.size(); i++) {
                const int env_id = actions[i].env_id;
                envs_[env_id]->set_action(actions[i]);

                ActionSlice action;
                action.env_id = env_id;
                action.order = is_sync_ ? i : -1;
                action.force_reset = false;

                action_slices.emplace_back(action);
            }

            if (is_sync_) {
                stepping_env_num_ += actions.size();
            }

            action_buffer_queue_->enqueue_bulk(action_slices);
        }

        /**
         * Receive timesteps from the environments
         * This is the asynchronous version that waits for results after send()
         *
         * @return Vector of timesteps from the environments
         */
        std::vector<Timestep> recv() {
            int additional_wait = 0;
            if (is_sync_ && stepping_env_num_ < batch_size_) {
                additional_wait = batch_size_ - stepping_env_num_;
            }

            std::vector<Timestep> timesteps = state_buffer_queue_->wait(additional_wait);

            if (is_sync_) {
                stepping_env_num_ -= timesteps.size();
            }

            return timesteps;
        }

        /**
         * Step the environments with actions and wait for results
         * This is a convenience method that combines send() and recv()
         *
         * @param actions Vector of actions for the environments
         * @return Vector of timesteps from the environments
         */
        std::vector<Timestep> step(const std::vector<EnvironmentAction>& actions) {
            send(actions);
            return recv();
        }

        int get_num_envs() const {
            return num_envs_;
        }

        int get_batch_size() const {
            return batch_size_;
        }

        int get_obs_size() const {
            return obs_size_;
        }

    private:
        int num_envs_;                                    // Number of parallel environments
        int batch_size_;                                  // Batch size for processing
        int num_threads_;                                 // Number of worker threads
        bool is_sync_;                                    // Whether to operate in synchronous mode
        int obs_size_;

        std::atomic<bool> stop_;                          // Signal to stop worker threads
        std::atomic<int> stepping_env_num_;               // Number of environments currently stepping
        std::vector<std::thread> workers_;                // Worker threads
        std::unique_ptr<ActionBufferQueue> action_buffer_queue_;  // Queue for actions
        std::unique_ptr<StateBufferQueue> state_buffer_queue_;    // Queue for observations
        std::vector<std::unique_ptr<PreprocessedAtariEnv>> envs_; // Environment instances

        /**
         * Worker thread function that processes environment steps
         */
        void worker_function() const {
            while (!stop_) {
                try {
                    ActionSlice action = action_buffer_queue_->dequeue();
                    if (stop_) {
                        break;
                    }

                    const int env_id = action.env_id;
                    const int order = action.order;

                    if (action.force_reset || envs_[env_id]->is_episode_over()) {
                        envs_[env_id]->reset();
                    } else {
                        envs_[env_id]->step();
                    }

                    // Get timestep and write to state buffer
                    Timestep timestep = envs_[env_id]->get_timestep();
                    state_buffer_queue_->write(timestep, order);
                } catch (const std::exception& e) {
                    // Log error but continue processing
                    std::cerr << "Error in worker thread: " << e.what() << std::endl;
                }
            }
        }

        /**
     * Set thread affinity for worker threads
     */
        void set_thread_affinity(const int thread_affinity_offset, const int processor_count) {
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
    };
}

#endif // ALE_VECTOR_ASYNC_VECTORIZER_HPP_
