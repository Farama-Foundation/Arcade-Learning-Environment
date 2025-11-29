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
     * Batch data from recv() - caller takes ownership of allocated buffers.
     */
    struct BatchData {
        int* env_ids;                  // Newly allocated, caller owns
        uint8_t* observations;         // Newly allocated, caller owns
        int* rewards;                  // Newly allocated, caller owns
        bool* terminations;            // Newly allocated, caller owns
        bool* truncations;             // Newly allocated, caller owns
        int* lives;                    // Newly allocated, caller owns
        int* frame_numbers;            // Newly allocated, caller owns
        int* episode_frame_numbers;    // Newly allocated, caller owns

        uint8_t* final_observations;   // nullptr or newly allocated, caller owns
        std::size_t batch_size;        // Number of results
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
            first_batch_(true),
            action_queue_(new ActionQueue(num_envs_)),
            pending_obs_buffer_(nullptr),
            pending_final_obs_(nullptr),
            pending_env_ids_(nullptr),
            pending_rewards_(nullptr),
            pending_terminations_(nullptr),
            pending_truncations_(nullptr),
            pending_lives_(nullptr),
            pending_frame_numbers_(nullptr),
            pending_episode_frame_numbers_(nullptr) {

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
            // Allocate output buffers BEFORE enqueueing (prevents race condition)
            const std::size_t total_obs_size = batch_size_ * stacked_obs_size_;
            pending_obs_buffer_ = new uint8_t[total_obs_size];
            state_buffer_->set_output_buffer(pending_obs_buffer_);

            // Release slots from previous batch (but not on first batch)
            if (!first_batch_) {
                state_buffer_->release_slots();
            }
            first_batch_ = false;

            // Allocate metadata buffers
            pending_env_ids_ = new int[batch_size_];
            pending_rewards_ = new int[batch_size_];
            pending_terminations_ = new bool[batch_size_];
            pending_truncations_ = new bool[batch_size_];
            pending_lives_ = new int[batch_size_];
            pending_frame_numbers_ = new int[batch_size_];
            pending_episode_frame_numbers_ = new int[batch_size_];
            state_buffer_->set_metadata_buffers(
                pending_env_ids_,
                pending_rewards_,
                pending_terminations_,
                pending_truncations_,
                pending_lives_,
                pending_frame_numbers_,
                pending_episode_frame_numbers_
            );

            // In SameStep mode, also allocate final_obs buffer
            if (autoreset_mode_ == AutoresetMode::SameStep) {
                pending_final_obs_ = new uint8_t[total_obs_size];
                state_buffer_->set_final_obs_buffer(pending_final_obs_);
            }

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
            // Allocate output buffers BEFORE enqueueing (prevents race condition)
            const std::size_t total_obs_size = batch_size_ * stacked_obs_size_;

            pending_obs_buffer_ = new uint8_t[total_obs_size];
            pending_env_ids_ = new int[batch_size_];
            pending_rewards_ = new int[batch_size_];
            pending_terminations_ = new bool[batch_size_];
            pending_truncations_ = new bool[batch_size_];
            pending_lives_ = new int[batch_size_];
            pending_frame_numbers_ = new int[batch_size_];
            pending_episode_frame_numbers_ = new int[batch_size_];

            state_buffer_->set_output_buffer(pending_obs_buffer_);
            state_buffer_->set_metadata_buffers(
                pending_env_ids_,
                pending_rewards_,
                pending_terminations_,
                pending_truncations_,
                pending_lives_,
                pending_frame_numbers_,
                pending_episode_frame_numbers_
            );

            // Release slots from previous batch (but not on first batch)
            if (!first_batch_) {
                state_buffer_->release_slots();
            }
            first_batch_ = false;

            // In SameStep mode, also allocate final_obs buffer
            if (autoreset_mode_ == AutoresetMode::SameStep) {
                pending_final_obs_ = new uint8_t[total_obs_size];
                state_buffer_->set_final_obs_buffer(pending_final_obs_);
            }

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
         * @return BatchData containing observation data and metadata
         */
        BatchData recv() {
            // Wait for all workers to complete
            state_buffer_->wait_for_batch();

            // Build result - transfer ownership of all buffers (no copying!)
            BatchData result;
            result.observations = pending_obs_buffer_;
            result.final_observations = pending_final_obs_;
            result.env_ids = pending_env_ids_;
            result.rewards = pending_rewards_;
            result.terminations = pending_terminations_;
            result.truncations = pending_truncations_;
            result.lives = pending_lives_;
            result.frame_numbers = pending_frame_numbers_;
            result.episode_frame_numbers = pending_episode_frame_numbers_;
            result.batch_size = batch_size_;

            // Clear pending pointers (ownership transferred)
            pending_obs_buffer_ = nullptr;
            pending_final_obs_ = nullptr;
            pending_env_ids_ = nullptr;
            pending_rewards_ = nullptr;
            pending_terminations_ = nullptr;
            pending_truncations_ = nullptr;
            pending_lives_ = nullptr;
            pending_frame_numbers_ = nullptr;
            pending_episode_frame_numbers_ = nullptr;

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
        bool first_batch_;                                // Track if this is the first batch (don't release permits)
        std::vector<std::thread> workers_;                // Worker threads
        std::unique_ptr<ActionQueue> action_queue_;       // Queue for actions
        std::unique_ptr<StateBuffer> state_buffer_;       // Buffer for observations and metadata
        std::vector<std::unique_ptr<PreprocessedAtariEnv>> envs_; // Environment instances

        // Pending buffers allocated in send()/reset(), returned in recv()
        uint8_t* pending_obs_buffer_;                     // Observations buffer
        uint8_t* pending_final_obs_;                      // Final observations buffer (SameStep mode only)
        int* pending_env_ids_;                            // Env IDs metadata buffer
        int* pending_rewards_;                            // Rewards metadata buffer
        bool* pending_terminations_;                      // Terminations metadata buffer
        bool* pending_truncations_;                       // Truncations metadata buffer
        int* pending_lives_;                              // Lives metadata buffer
        int* pending_frame_numbers_;                      // Frame numbers metadata buffer
        int* pending_episode_frame_numbers_;              // Episode frame numbers metadata buffer

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
                    if (autoreset_mode_ == AutoresetMode::NextStep) {
                        if (action.force_reset || envs_[env_id]->is_episode_over()) {
                            envs_[env_id]->reset();
                        } else {
                            envs_[env_id]->step();
                        }

                        // Get write slot - pointers are into the pre-allocated output buffer (after the reset or step occurs)
                        WriteSlot slot = state_buffer_->allocate_write_slot(env_id);
                        envs_[env_id]->write_timestep_to(
                            slot.obs_dest,
                            slot.env_id_dest,
                            slot.reward_dest,
                            slot.terminated_dest,
                            slot.truncated_dest,
                            slot.lives_dest,
                            slot.frame_number_dest,
                            slot.episode_frame_number_dest
                        );
                    } else if (autoreset_mode_ == AutoresetMode::SameStep) {
                        if (action.force_reset) {
                            envs_[env_id]->reset();

                            // Get write slot - pointers are into the pre-allocated output buffer (after the force reset)
                            WriteSlot slot = state_buffer_->allocate_write_slot(env_id);
                            envs_[env_id]->write_timestep_to(
                                slot.obs_dest,
                                slot.env_id_dest,
                                slot.reward_dest,
                                slot.terminated_dest,
                                slot.truncated_dest,
                                slot.lives_dest,
                                slot.frame_number_dest,
                                slot.episode_frame_number_dest
                            );
                        } else {
                            envs_[env_id]->step();

                            // Get write slot - pointers are into the pre-allocated output buffer (after the step)
                            WriteSlot slot = state_buffer_->allocate_write_slot(env_id);

                            if (envs_[env_id]->is_episode_over()) {
                                // Write current (final) observation before reset
                                envs_[env_id]->write_observation_to(slot.final_obs_dest);

                                // Capture pre-reset metadata temporarily (for reward/terminated/truncated)
                                int pre_reward;
                                bool pre_terminated, pre_truncated;
                                envs_[env_id]->write_metadata_to(
                                    slot.env_id_dest,
                                    &pre_reward,
                                    &pre_terminated,
                                    &pre_truncated,
                                    slot.lives_dest,
                                    slot.frame_number_dest,
                                    slot.episode_frame_number_dest
                                );

                                // Reset and write new observation
                                envs_[env_id]->reset();
                                envs_[env_id]->write_timestep_to(
                                    slot.obs_dest,
                                    slot.env_id_dest,  // overwrites with same value
                                    slot.reward_dest,
                                    slot.terminated_dest,
                                    slot.truncated_dest,
                                    slot.lives_dest,   // overwrites with reset lives
                                    slot.frame_number_dest,
                                    slot.episode_frame_number_dest
                                );

                                // Restore pre-reset reward/terminated/truncated
                                *slot.reward_dest = pre_reward;
                                *slot.terminated_dest = pre_terminated;
                                *slot.truncated_dest = pre_truncated;
                            } else {
                                // No episode over
                                envs_[env_id]->write_timestep_to(
                                    slot.obs_dest,
                                    slot.env_id_dest,
                                    slot.reward_dest,
                                    slot.terminated_dest,
                                    slot.truncated_dest,
                                    slot.lives_dest,
                                    slot.frame_number_dest,
                                    slot.episode_frame_number_dest
                                );
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
