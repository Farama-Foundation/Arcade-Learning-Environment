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
     * Result from recv() - caller takes ownership of allocated buffers.
     */
    struct RecvResult {
        uint8_t* obs_data;                      // Newly allocated, caller owns
        std::vector<TimestepMetadata> metadata; // Copied from internal buffer
        uint8_t* final_obs_data;                // nullptr or newly allocated, caller owns
        std::vector<uint8_t> has_final_obs;     // Which slots have final_obs (uint8_t for compatibility)
        std::size_t batch_size;                 // Number of results
    };

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
            autoreset_mode_(autoreset_mode),
            stop_(false),
            action_queue_(new ActionQueue(num_envs_)),
            pending_obs_buffer_(nullptr) {

            // Create environments
            envs_.resize(num_envs_);
            for (int i = 0; i < num_envs_; ++i) {
                envs_[i] = env_factory(i);
            }
            stacked_obs_size_ = envs_[0]->get_stacked_obs_size();

            // Create state buffer with observation size
            state_buffer_ = std::make_unique<StateBuffer>(batch_size_, num_envs_, stacked_obs_size_);

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
            action_queue_->enqueue_bulk(empty_actions);
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
            // Allocate output buffer BEFORE enqueueing (prevents race condition)
            const std::size_t total_obs_size = batch_size_ * stacked_obs_size_;
            pending_obs_buffer_ = new uint8_t[total_obs_size];
            state_buffer_->set_output_buffer(pending_obs_buffer_);

            // Prepare reset actions
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

            // Enqueue actions - workers can now safely write to buffer
            action_queue_->enqueue_bulk(reset_actions);
        }

        /**
         * Send actions to the sub-environments
         *
         * @param actions Vector of actions to send to the sub-environments
         */
        void send(const std::vector<EnvironmentAction>& actions) {
            // Allocate output buffer BEFORE enqueueing (prevents race condition)
            const std::size_t total_obs_size = batch_size_ * stacked_obs_size_;
            pending_obs_buffer_ = new uint8_t[total_obs_size];
            state_buffer_->set_output_buffer(pending_obs_buffer_);

            // Prepare action slices
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

            // Enqueue actions - workers can now safely write to buffer
            action_queue_->enqueue_bulk(action_slices);
        }

        /**
         * Receive timesteps from the environments.
         * Returns ownership of allocated observation buffer to caller.
         *
         * @return RecvResult containing observation data and metadata
         */
        RecvResult recv() {
            // Wait for all workers to complete
            state_buffer_->wait_for_batch();

            // Build result
            RecvResult result;
            result.obs_data = pending_obs_buffer_;  // Transfer ownership
            result.batch_size = batch_size_;
            pending_obs_buffer_ = nullptr;

            // Copy metadata (small - ~32 bytes per env)
            result.metadata.resize(batch_size_);
            std::memcpy(
                result.metadata.data(),
                state_buffer_->get_metadata(),
                batch_size_ * sizeof(TimestepMetadata)
            );

            // Handle final_obs for SameStep mode
            if (autoreset_mode_ == AutoresetMode::SameStep) {
                const uint8_t* has_final = state_buffer_->get_has_final_obs();
                bool any_final = false;
                for (std::size_t i = 0; i < batch_size_; i++) {
                    if (has_final[i]) {
                        any_final = true;
                        break;
                    }
                }

                if (any_final) {
                    const std::size_t total_obs_size = batch_size_ * stacked_obs_size_;
                    result.final_obs_data = new uint8_t[total_obs_size];
                    std::memcpy(result.final_obs_data, state_buffer_->get_final_obs_buffer(), total_obs_size);
                    result.has_final_obs.assign(has_final, has_final + batch_size_);
                } else {
                    result.final_obs_data = nullptr;
                }
            } else {
                result.final_obs_data = nullptr;
            }

            // Reset state buffer for next batch
            state_buffer_->reset();

            return result;
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
        int stacked_obs_size_;                            // The observation size (stack-num * width * height * channels)
        AutoresetMode autoreset_mode_;                    // How to reset sub-environments after an episode ends

        std::atomic<bool> stop_;                          // Signal to stop worker threads
        std::vector<std::thread> workers_;                // Worker threads
        std::unique_ptr<ActionQueue> action_queue_;       // Queue for actions
        std::unique_ptr<StateBuffer> state_buffer_;       // Buffer for observations and metadata
        std::vector<std::unique_ptr<PreprocessedAtariEnv>> envs_; // Environment instances

        uint8_t* pending_obs_buffer_;                     // Buffer allocated in send(), returned in recv()

        /**
         * Worker thread function that processes environment steps.
         * Writes results directly to pre-allocated output buffer.
         */
        void worker_function() {
            while (!stop_) {
                try {
                    ActionSlice action = action_queue_->dequeue();
                    if (stop_) {
                        break;
                    }

                    const int env_id = action.env_id;

                    // Get write slot - pointers are into the pre-allocated output buffer
                    WriteSlot slot = state_buffer_->allocate_write_slot(env_id);

                    if (autoreset_mode_ == AutoresetMode::NextStep) {
                        if (action.force_reset || envs_[env_id]->is_episode_over()) {
                            envs_[env_id]->reset();
                        } else {
                            envs_[env_id]->step();
                        }

                        // Write directly to output buffer (single copy: linearize frame stack)
                        envs_[env_id]->write_timestep_to(slot.obs_dest, *slot.meta);

                    } else if (autoreset_mode_ == AutoresetMode::SameStep) {
                        if (action.force_reset) {
                            envs_[env_id]->reset();
                            envs_[env_id]->write_timestep_to(slot.obs_dest, *slot.meta);
                        } else {
                            envs_[env_id]->step();

                            if (envs_[env_id]->is_episode_over()) {
                                // Save final observation before reset
                                envs_[env_id]->write_observation_to(slot.final_obs_dest);
                                state_buffer_->mark_slot_has_final_obs(slot.slot_index);

                                // Capture pre-reset metadata
                                TimestepMetadata pre_reset_meta;
                                envs_[env_id]->write_metadata_to(pre_reset_meta);

                                // Reset and write new observation
                                envs_[env_id]->reset();
                                envs_[env_id]->write_timestep_to(slot.obs_dest, *slot.meta);

                                // Restore pre-reset reward/terminated/truncated
                                slot.meta->reward = pre_reset_meta.reward;
                                slot.meta->terminated = pre_reset_meta.terminated;
                                slot.meta->truncated = pre_reset_meta.truncated;
                            } else {
                                envs_[env_id]->write_timestep_to(slot.obs_dest, *slot.meta);
                            }
                        }
                    } else {
                        throw std::runtime_error("Invalid autoreset mode");
                    }

                    state_buffer_->mark_complete();

                } catch (const std::exception& e) {
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
