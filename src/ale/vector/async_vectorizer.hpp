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
         * @param autoreset_mode Specify how to automatically reset the sub-environments after an episode ends
         */
        explicit AsyncVectorizer(
            const int num_envs,
            const int batch_size = 0,
            const int num_threads = 0,
            const int thread_affinity_offset = -1,
            const std::function<std::unique_ptr<PreprocessedAtariEnv>(int)> &env_factory = nullptr,
            const AutoresetMode autoreset_mode = AutoresetMode::NextStep
        ) : num_envs_(num_envs),
            batch_size_(batch_size > 0 ? batch_size : num_envs),
            is_sync_(batch_size_ == num_envs_),
            autoreset_mode_(autoreset_mode),
            stop_(false),
            action_buffer_queue_(new ActionBufferQueue(num_envs_)),
            state_buffer_queue_(new StateBufferQueue(batch_size_, num_envs_)),
            final_obs_storage_(num_envs_) {

            // Create environments
            envs_.resize(num_envs_);
            for (int i = 0; i < num_envs_; ++i) {
                envs_[i] = env_factory(i);
            }
            stacked_obs_size_ = envs_[0]->get_stacked_obs_size();

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
                action.force_reset = true;

                reset_actions.emplace_back(action);
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

            for (size_t i = 0; i < actions.size(); i++) {
                const int env_id = actions[i].env_id;
                envs_[env_id]->set_action(actions[i]);

                ActionSlice action;
                action.env_id = env_id;
                action.force_reset = false;

                action_slices.emplace_back(action);
            }

            action_buffer_queue_->enqueue_bulk(action_slices);
        }

        /**
         * Receive timesteps from the environments
         * This is the asynchronous version that waits for results after send()
         *
         * @return Vector of timesteps from the environments
         */
        const std::vector<Timestep> recv() {
            std::vector<Timestep> timesteps = state_buffer_queue_->collect();
            return timesteps;
        }

        /**
         * Step the environments with actions and wait for results
         * This is a convenience method that combines send() and recv()
         *
         * @param actions Vector of actions for the environments
         * @return Vector of timesteps from the environments
         */
        const std::vector<Timestep> step(const std::vector<EnvironmentAction>& actions) {
            send(actions);
            return recv();
        }

        const int get_num_envs() const {
            return num_envs_;
        }

        const int get_batch_size() const {
            return batch_size_;
        }

        const int get_stacked_obs_size() const {
            return stacked_obs_size_;
        }

        const AutoresetMode get_autoreset() const {
            return autoreset_mode_;
        }

    private:
        int num_envs_;                                    // Number of parallel environments
        int batch_size_;                                  // Batch size for processing
        int num_threads_;                                 // Number of worker threads
        bool is_sync_;                                    // Whether to operate in synchronous mode
        int stacked_obs_size_;                            // The observation size (stack-num * width * height * channels)
        AutoresetMode autoreset_mode_;                    // How to reset sub-environments after an episode ends

        std::atomic<bool> stop_;                          // Signal to stop worker threads
        std::vector<std::thread> workers_;                // Worker threads
        std::unique_ptr<ActionBufferQueue> action_buffer_queue_;  // Queue for actions
        std::unique_ptr<StateBufferQueue> state_buffer_queue_;    // Queue for observations
        std::vector<std::unique_ptr<PreprocessedAtariEnv>> envs_; // Environment instances

        mutable std::vector<std::vector<uint8_t>> final_obs_storage_;  // For same-step autoreset

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
                    if (autoreset_mode_ == AutoresetMode::NextStep) {
                        if (action.force_reset || envs_[env_id]->is_episode_over()) {
                            envs_[env_id]->reset();
                        } else {
                            envs_[env_id]->step();
                        }

                        // Get timestep and write to state buffer
                        Timestep timestep = envs_[env_id]->get_timestep();
                        timestep.final_observation = nullptr;  // Not used in NextStep mode
                        state_buffer_queue_->write(timestep);
                    } else if (autoreset_mode_ == AutoresetMode::SameStep) {
                        if (action.force_reset) {
                            // on standard `reset`
                            envs_[env_id]->reset();
                            Timestep timestep = envs_[env_id]->get_timestep();
                            timestep.final_observation = nullptr;
                            state_buffer_queue_->write(timestep);
                        } else {
                            envs_[env_id]->step();
                            Timestep step_timestep = envs_[env_id]->get_timestep();

                            // if episode over, autoreset
                            if (envs_[env_id]->is_episode_over()) {
                                final_obs_storage_[env_id] = step_timestep.observation;

                                envs_[env_id]->reset();
                                Timestep reset_timestep = envs_[env_id]->get_timestep();

                                reset_timestep.final_observation = &final_obs_storage_[env_id];
                                reset_timestep.reward = step_timestep.reward;
                                reset_timestep.terminated = step_timestep.terminated;
                                reset_timestep.truncated = step_timestep.truncated;

                                // Write the reset timestep with the some of the step timestep data
                                state_buffer_queue_->write(reset_timestep);
                            } else {
                                step_timestep.final_observation = nullptr;
                                state_buffer_queue_->write(step_timestep);
                            }
                        }
                    } else {
                        throw std::runtime_error("Invalid autoreset mode");
                    }
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
