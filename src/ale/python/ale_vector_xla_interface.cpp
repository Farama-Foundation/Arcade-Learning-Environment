#include "xla/ffi/api/ffi.h"

#include "ale/vector/env_vectorizer.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <vector>
#include <cstring>  // For memcpy
#include <iostream>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace ffi = xla::ffi;
namespace nb = nanobind;

// CPU version of XLAReset
ffi::Error XLAResetImpl(
    ffi::Buffer<ffi::U8> handle_buffer,
    ffi::Buffer<ffi::S32> reset_indices_buffer,
    ffi::Buffer<ffi::S32> reset_seeds_buffer,
    ffi::ResultBuffer<ffi::U8> new_handle_buffer,
    ffi::ResultBuffer<ffi::U8> observations_buffer,
    ffi::ResultBuffer<ffi::S32> env_ids_buffer,
    ffi::ResultBuffer<ffi::S32> lives_buffer,
    ffi::ResultBuffer<ffi::S32> frame_numbers_buffer,
    ffi::ResultBuffer<ffi::S32> episode_frame_numbers_buffer
) {
    // Validate handle buffer size
    if (handle_buffer.element_count() != sizeof(ale::vector::EnvVectorizer*)) {
        return ffi::Error::Internal("Incorrect handle buffer size in reset");
    }

    // Safely extract the vectorizer pointer from the handle buffer
    ale::vector::EnvVectorizer* vectorizer = nullptr;
    std::memcpy(&vectorizer, handle_buffer.typed_data(), sizeof(vectorizer));
    if (!vectorizer) {
        return ffi::Error::Internal("Invalid vectorizer pointer in reset");
    }

    // Copy handle to output (needed for state management)
    if (new_handle_buffer->element_count() != handle_buffer.element_count()) {
        return ffi::Error::Internal("Incorrect new handle buffer size in reset");
    }
    std::memcpy(new_handle_buffer->typed_data(),
                handle_buffer.typed_data(),
                handle_buffer.element_count());

    try {
        // Extract reset indices and seeds
        std::vector<int> reset_indices(
            reset_indices_buffer.typed_data(),
            reset_indices_buffer.typed_data() + reset_indices_buffer.element_count());
        std::vector<int> reset_seeds(
            reset_seeds_buffer.typed_data(),
            reset_seeds_buffer.typed_data() + reset_seeds_buffer.element_count());

        // Reset the environments (returns BatchResult directly)
        auto result = vectorizer->reset(reset_indices, reset_seeds);

        size_t batch_size = result.batch_size();
        size_t stacked_obs_size = vectorizer->stacked_obs_size();

        // Check if the observations buffer is large enough
        if (observations_buffer->element_count() != batch_size * stacked_obs_size) {
            return ffi::Error::Internal("Observations buffer is the wrong size");
        }

        // Copy data from BatchResult to output buffers
        std::memcpy(observations_buffer->typed_data(), result.obs_data(),
                    batch_size * stacked_obs_size);
        std::memcpy(env_ids_buffer->typed_data(), result.env_ids_data(),
                    batch_size * sizeof(int32_t));
        std::memcpy(lives_buffer->typed_data(), result.lives_data(),
                    batch_size * sizeof(int32_t));
        std::memcpy(frame_numbers_buffer->typed_data(), result.frame_numbers_data(),
                    batch_size * sizeof(int32_t));
        std::memcpy(episode_frame_numbers_buffer->typed_data(), result.episode_frame_numbers_data(),
                    batch_size * sizeof(int32_t));

        return ffi::Error::Success();
    }
    catch (const std::exception& e) {
        std::string error_msg = "Exception during reset: ";
        error_msg += e.what();
        std::cerr << error_msg << std::endl;
        return ffi::Error::Internal(error_msg);
    }
    catch (...) {
        std::cerr << "Unknown exception during reset" << std::endl;
        return ffi::Error::Internal("Unknown exception during reset");
    }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AtariVectorEnvXLAReset, XLAResetImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::U8>>()   // handle
        .Arg<ffi::Buffer<ffi::S32>>()  // reset_indices
        .Arg<ffi::Buffer<ffi::S32>>()  // reset_seeds
        .Ret<ffi::Buffer<ffi::U8>>()   // new_handle
        .Ret<ffi::Buffer<ffi::U8>>()   // observations
        .Ret<ffi::Buffer<ffi::S32>>()  // env_ids
        .Ret<ffi::Buffer<ffi::S32>>()  // lives
        .Ret<ffi::Buffer<ffi::S32>>()  // frame_numbers
        .Ret<ffi::Buffer<ffi::S32>>()  // episode_frame_numbers
);

#ifdef __CUDACC__
// GPU version of XLAReset with CUDA stream support
ffi::Error XLAResetGPUImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::U8> handle_buffer,
    ffi::Buffer<ffi::S32> reset_indices_buffer,
    ffi::Buffer<ffi::S32> reset_seeds_buffer,
    ffi::ResultBuffer<ffi::U8> new_handle_buffer,
    ffi::ResultBuffer<ffi::U8> observations_buffer,
    ffi::ResultBuffer<ffi::S32> env_ids_buffer,
    ffi::ResultBuffer<ffi::S32> lives_buffer,
    ffi::ResultBuffer<ffi::S32> frame_numbers_buffer,
    ffi::ResultBuffer<ffi::S32> episode_frame_numbers_buffer
) {
    // Validate handle buffer size
    if (handle_buffer.element_count() != sizeof(ale::vector::EnvVectorizer*)) {
        return ffi::Error::Internal("Incorrect handle buffer size in reset (GPU)");
    }

    // Allocate host memory for handle
    std::vector<uint8_t> host_handle(handle_buffer.element_count());
    cudaError_t err = cudaMemcpyAsync(host_handle.data(), handle_buffer.typed_data(),
                                       handle_buffer.element_count(),
                                       cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA memcpy failed (handle D2H): ") + cudaGetErrorString(err));
    }

    // Wait for the transfer to complete
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA stream sync failed: ") + cudaGetErrorString(err));
    }

    // Extract the vectorizer pointer
    ale::vector::EnvVectorizer* vectorizer = nullptr;
    std::memcpy(&vectorizer, host_handle.data(), sizeof(vectorizer));
    if (!vectorizer) {
        return ffi::Error::Internal("Invalid vectorizer pointer in reset (GPU)");
    }

    // Copy handle to output
    if (new_handle_buffer->element_count() != handle_buffer.element_count()) {
        return ffi::Error::Internal("Incorrect new handle buffer size in reset (GPU)");
    }
    err = cudaMemcpyAsync(new_handle_buffer->typed_data(), host_handle.data(),
                          handle_buffer.element_count(),
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA memcpy failed (new handle H2D): ") + cudaGetErrorString(err));
    }

    try {
        // Allocate and copy reset indices and seeds from GPU to CPU
        std::vector<int32_t> host_reset_indices(reset_indices_buffer.element_count());
        std::vector<int32_t> host_reset_seeds(reset_seeds_buffer.element_count());

        err = cudaMemcpyAsync(host_reset_indices.data(), reset_indices_buffer.typed_data(),
                              reset_indices_buffer.element_count() * sizeof(int32_t),
                              cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) {
            return ffi::Error::Internal(std::string("CUDA memcpy failed (indices D2H): ") + cudaGetErrorString(err));
        }

        err = cudaMemcpyAsync(host_reset_seeds.data(), reset_seeds_buffer.typed_data(),
                              reset_seeds_buffer.element_count() * sizeof(int32_t),
                              cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) {
            return ffi::Error::Internal(std::string("CUDA memcpy failed (seeds D2H): ") + cudaGetErrorString(err));
        }

        // Wait for transfers to complete
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            return ffi::Error::Internal(std::string("CUDA stream sync failed: ") + cudaGetErrorString(err));
        }

        // Convert to int for vectorizer API
        std::vector<int> reset_indices(host_reset_indices.begin(), host_reset_indices.end());
        std::vector<int> reset_seeds(host_reset_seeds.begin(), host_reset_seeds.end());

        // Reset the environments (returns BatchResult directly)
        auto result = vectorizer->reset(reset_indices, reset_seeds);

        size_t batch_size = result.batch_size();
        size_t stacked_obs_size = vectorizer->stacked_obs_size();

        // Check if the observations buffer is large enough
        if (observations_buffer->element_count() != batch_size * stacked_obs_size) {
            return ffi::Error::Internal("Observations buffer is the wrong size (GPU)");
        }

        // Copy data from BatchResult to GPU buffers via host memory
        err = cudaMemcpyAsync(observations_buffer->typed_data(), result.obs_data(),
                              batch_size * stacked_obs_size,
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            return ffi::Error::Internal(std::string("CUDA memcpy failed (observations H2D): ") + cudaGetErrorString(err));
        }

        err = cudaMemcpyAsync(env_ids_buffer->typed_data(), result.env_ids_data(),
                              batch_size * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            return ffi::Error::Internal(std::string("CUDA memcpy failed (env_ids H2D): ") + cudaGetErrorString(err));
        }

        err = cudaMemcpyAsync(lives_buffer->typed_data(), result.lives_data(),
                              batch_size * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            return ffi::Error::Internal(std::string("CUDA memcpy failed (lives H2D): ") + cudaGetErrorString(err));
        }

        err = cudaMemcpyAsync(frame_numbers_buffer->typed_data(), result.frame_numbers_data(),
                              batch_size * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            return ffi::Error::Internal(std::string("CUDA memcpy failed (frame_numbers H2D): ") + cudaGetErrorString(err));
        }

        err = cudaMemcpyAsync(episode_frame_numbers_buffer->typed_data(), result.episode_frame_numbers_data(),
                              batch_size * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            return ffi::Error::Internal(std::string("CUDA memcpy failed (episode_frame_numbers H2D): ") + cudaGetErrorString(err));
        }

        // Check for any CUDA errors
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return ffi::Error::Internal(std::string("CUDA error after operations: ") + cudaGetErrorString(err));
        }

        return ffi::Error::Success();
    }
    catch (const std::exception& e) {
        std::string error_msg = "Exception during reset (GPU): ";
        error_msg += e.what();
        std::cerr << error_msg << std::endl;
        return ffi::Error::Internal(error_msg);
    }
    catch (...) {
        std::cerr << "Unknown exception during reset (GPU)" << std::endl;
        return ffi::Error::Internal("Unknown exception during reset (GPU)");
    }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AtariVectorEnvXLAResetGPU, XLAResetGPUImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()  // CUDA stream
        .Arg<ffi::Buffer<ffi::U8>>()   // handle
        .Arg<ffi::Buffer<ffi::S32>>()  // reset_indices
        .Arg<ffi::Buffer<ffi::S32>>()  // reset_seeds
        .Ret<ffi::Buffer<ffi::U8>>()   // new_handle
        .Ret<ffi::Buffer<ffi::U8>>()   // observations
        .Ret<ffi::Buffer<ffi::S32>>()  // env_ids
        .Ret<ffi::Buffer<ffi::S32>>()  // lives
        .Ret<ffi::Buffer<ffi::S32>>()  // frame_numbers
        .Ret<ffi::Buffer<ffi::S32>>()  // episode_frame_numbers
);
#endif  // __CUDACC__

// CPU version of XLAStep
ffi::Error XLAStepImpl(
    ffi::Buffer<ffi::U8> handle_buffer,
    ffi::Buffer<ffi::S32> action_id_buffer,
    ffi::Buffer<ffi::F32> paddle_strength_buffer,
    ffi::ResultBuffer<ffi::U8> new_handle_buffer,
    ffi::ResultBuffer<ffi::U8> observations_buffer,
    ffi::ResultBuffer<ffi::S32> rewards_buffer,
    ffi::ResultBuffer<ffi::PRED> terminations_buffer,
    ffi::ResultBuffer<ffi::PRED> truncations_buffer,
    ffi::ResultBuffer<ffi::S32> env_id_buffer,
    ffi::ResultBuffer<ffi::S32> lives_buffer,
    ffi::ResultBuffer<ffi::S32> frame_numbers_buffer,
    ffi::ResultBuffer<ffi::S32> episode_frame_numbers_buffer
) {
    // Validate handle buffer size
    if (handle_buffer.element_count() != sizeof(ale::vector::EnvVectorizer*)) {
        return ffi::Error::Internal("Incorrect handle buffer size in step");
    }

    // Safely extract the vectorizer pointer from the handle buffer
    ale::vector::EnvVectorizer* vectorizer = nullptr;
    std::memcpy(&vectorizer, handle_buffer.typed_data(), sizeof(vectorizer));
    if (!vectorizer) {
        return ffi::Error::Internal("Invalid vectorizer pointer in step");
    }

    // Copy handle to output
    if (new_handle_buffer->element_count() != handle_buffer.element_count()) {
        return ffi::Error::Internal("New handle buffer is the wrong size");
    }
    std::memcpy(new_handle_buffer->typed_data(),
                handle_buffer.typed_data(),
                handle_buffer.element_count());

    try {
        size_t num_envs = vectorizer->num_envs();

        if (action_id_buffer.element_count() != num_envs) {
            return ffi::Error::Internal("Action id buffer is the wrong size");
        } else if (paddle_strength_buffer.element_count() != num_envs) {
            return ffi::Error::Internal("Paddle strength buffer is the wrong size");
        }

        std::vector<ale::vector::Action> actions(num_envs);
        for (size_t i = 0; i < num_envs; ++i) {
            actions[i].env_id = static_cast<int>(i);
            actions[i].action_id = action_id_buffer.typed_data()[i];
            actions[i].paddle_strength = paddle_strength_buffer.typed_data()[i];
            actions[i].force_reset = false;
        }

        // Step the environments
        vectorizer->send(actions);

        // Receive the results
        auto result = vectorizer->recv();

        size_t batch_size = result.batch_size();
        size_t stacked_obs_size = vectorizer->stacked_obs_size();

        // Check if the observations buffer is large enough
        if (observations_buffer->element_count() != batch_size * stacked_obs_size) {
            return ffi::Error::Internal("Observations buffer is the wrong size");
        }

        // Copy data from BatchResult to output buffers
        std::memcpy(observations_buffer->typed_data(), result.obs_data(),
                    batch_size * stacked_obs_size);
        std::memcpy(rewards_buffer->typed_data(), result.rewards_data(),
                    batch_size * sizeof(int32_t));
        std::memcpy(env_id_buffer->typed_data(), result.env_ids_data(),
                    batch_size * sizeof(int32_t));
        std::memcpy(lives_buffer->typed_data(), result.lives_data(),
                    batch_size * sizeof(int32_t));
        std::memcpy(frame_numbers_buffer->typed_data(), result.frame_numbers_data(),
                    batch_size * sizeof(int32_t));
        std::memcpy(episode_frame_numbers_buffer->typed_data(), result.episode_frame_numbers_data(),
                    batch_size * sizeof(int32_t));

        // Copy bools element-wise (bool* to PRED buffer)
        const bool* term_data = result.terminations_data();
        const bool* trunc_data = result.truncations_data();
        for (size_t i = 0; i < batch_size; ++i) {
            terminations_buffer->typed_data()[i] = term_data[i];
            truncations_buffer->typed_data()[i] = trunc_data[i];
        }

        return ffi::Error::Success();
    }
    catch (const std::exception& e) {
        std::string error_msg = "Exception during step: ";
        error_msg += e.what();
        std::cerr << error_msg << std::endl;
        return ffi::Error::Internal(error_msg);
    }
    catch (...) {
        std::cerr << "Unknown exception during step" << std::endl;
        return ffi::Error::Internal("Unknown exception during step");
    }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AtariVectorEnvXLAStep, XLAStepImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::U8>>()    // handle
        .Arg<ffi::Buffer<ffi::S32>>()   // actions
        .Arg<ffi::Buffer<ffi::F32>>()   // paddle_strength
        .Ret<ffi::Buffer<ffi::U8>>()    // new_handle
        .Ret<ffi::Buffer<ffi::U8>>()    // observations
        .Ret<ffi::Buffer<ffi::S32>>()   // rewards
        .Ret<ffi::Buffer<ffi::PRED>>()  // terminations
        .Ret<ffi::Buffer<ffi::PRED>>()  // truncations
        .Ret<ffi::Buffer<ffi::S32>>()   // env_ids
        .Ret<ffi::Buffer<ffi::S32>>()   // lives
        .Ret<ffi::Buffer<ffi::S32>>()   // frame_numbers
        .Ret<ffi::Buffer<ffi::S32>>()   // episode_frame_numbers
);

#ifdef __CUDACC__
// GPU version of XLAStep with CUDA stream support
ffi::Error XLAStepGPUImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::U8> handle_buffer,
    ffi::Buffer<ffi::S32> action_id_buffer,
    ffi::Buffer<ffi::F32> paddle_strength_buffer,
    ffi::ResultBuffer<ffi::U8> new_handle_buffer,
    ffi::ResultBuffer<ffi::U8> observations_buffer,
    ffi::ResultBuffer<ffi::S32> rewards_buffer,
    ffi::ResultBuffer<ffi::PRED> terminations_buffer,
    ffi::ResultBuffer<ffi::PRED> truncations_buffer,
    ffi::ResultBuffer<ffi::S32> env_id_buffer,
    ffi::ResultBuffer<ffi::S32> lives_buffer,
    ffi::ResultBuffer<ffi::S32> frame_numbers_buffer,
    ffi::ResultBuffer<ffi::S32> episode_frame_numbers_buffer
) {
    // Validate handle buffer size
    if (handle_buffer.element_count() != sizeof(ale::vector::EnvVectorizer*)) {
        return ffi::Error::Internal("Incorrect handle buffer size in step (GPU)");
    }

    // Allocate host memory for handle
    std::vector<uint8_t> host_handle(handle_buffer.element_count());
    cudaError_t err = cudaMemcpyAsync(host_handle.data(), handle_buffer.typed_data(),
                                       handle_buffer.element_count(),
                                       cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA memcpy failed (handle D2H): ") + cudaGetErrorString(err));
    }

    // Wait for the transfer to complete
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA stream sync failed: ") + cudaGetErrorString(err));
    }

    // Extract the vectorizer pointer
    ale::vector::EnvVectorizer* vectorizer = nullptr;
    std::memcpy(&vectorizer, host_handle.data(), sizeof(vectorizer));
    if (!vectorizer) {
        return ffi::Error::Internal("Invalid vectorizer pointer in step (GPU)");
    }

    // Copy handle to output
    if (new_handle_buffer->element_count() != handle_buffer.element_count()) {
        return ffi::Error::Internal("New handle buffer is the wrong size (GPU)");
    }
    err = cudaMemcpyAsync(new_handle_buffer->typed_data(), host_handle.data(),
                          handle_buffer.element_count(),
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA memcpy failed (new handle H2D): ") + cudaGetErrorString(err));
    }

    try {
        size_t num_envs = vectorizer->num_envs();

        if (action_id_buffer.element_count() != num_envs) {
            return ffi::Error::Internal("Action id buffer is the wrong size (GPU)");
        } else if (paddle_strength_buffer.element_count() != num_envs) {
            return ffi::Error::Internal("Paddle strength buffer is the wrong size (GPU)");
        }

        // Copy action IDs and paddle strength from GPU to CPU
        std::vector<int32_t> host_action_ids(num_envs);
        std::vector<float> host_paddle_strength(num_envs);

        err = cudaMemcpyAsync(host_action_ids.data(), action_id_buffer.typed_data(),
                              num_envs * sizeof(int32_t),
                              cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) {
            return ffi::Error::Internal(std::string("CUDA memcpy failed (actions D2H): ") + cudaGetErrorString(err));
        }

        err = cudaMemcpyAsync(host_paddle_strength.data(), paddle_strength_buffer.typed_data(),
                              num_envs * sizeof(float),
                              cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) {
            return ffi::Error::Internal(std::string("CUDA memcpy failed (paddle D2H): ") + cudaGetErrorString(err));
        }

        // Wait for transfers to complete
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            return ffi::Error::Internal(std::string("CUDA stream sync failed: ") + cudaGetErrorString(err));
        }

        // Prepare actions for vectorizer
        std::vector<ale::vector::Action> actions(num_envs);
        for (size_t i = 0; i < num_envs; ++i) {
            actions[i].env_id = static_cast<int>(i);
            actions[i].action_id = host_action_ids[i];
            actions[i].paddle_strength = host_paddle_strength[i];
            actions[i].force_reset = false;
        }

        // Step the environments (CPU operation)
        vectorizer->send(actions);

        // Receive the results
        auto result = vectorizer->recv();

        size_t batch_size = result.batch_size();
        size_t stacked_obs_size = vectorizer->stacked_obs_size();

        // Check if the observations buffer is large enough
        if (observations_buffer->element_count() != batch_size * stacked_obs_size) {
            return ffi::Error::Internal("Observations buffer is the wrong size (GPU)");
        }

        // Copy data from BatchResult to GPU buffers via host memory
        err = cudaMemcpyAsync(observations_buffer->typed_data(), result.obs_data(),
                              batch_size * stacked_obs_size,
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            return ffi::Error::Internal(std::string("CUDA memcpy failed (observations H2D): ") + cudaGetErrorString(err));
        }

        err = cudaMemcpyAsync(rewards_buffer->typed_data(), result.rewards_data(),
                              batch_size * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            return ffi::Error::Internal(std::string("CUDA memcpy failed (rewards H2D): ") + cudaGetErrorString(err));
        }

        err = cudaMemcpyAsync(env_id_buffer->typed_data(), result.env_ids_data(),
                              batch_size * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            return ffi::Error::Internal(std::string("CUDA memcpy failed (env_ids H2D): ") + cudaGetErrorString(err));
        }

        err = cudaMemcpyAsync(lives_buffer->typed_data(), result.lives_data(),
                              batch_size * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            return ffi::Error::Internal(std::string("CUDA memcpy failed (lives H2D): ") + cudaGetErrorString(err));
        }

        err = cudaMemcpyAsync(frame_numbers_buffer->typed_data(), result.frame_numbers_data(),
                              batch_size * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            return ffi::Error::Internal(std::string("CUDA memcpy failed (frame_numbers H2D): ") + cudaGetErrorString(err));
        }

        err = cudaMemcpyAsync(episode_frame_numbers_buffer->typed_data(), result.episode_frame_numbers_data(),
                              batch_size * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            return ffi::Error::Internal(std::string("CUDA memcpy failed (episode_frame_numbers H2D): ") + cudaGetErrorString(err));
        }

        // Copy bools element-wise to temporary buffer, then to GPU
        std::vector<uint8_t> host_terminations(batch_size);
        std::vector<uint8_t> host_truncations(batch_size);
        const bool* term_data = result.terminations_data();
        const bool* trunc_data = result.truncations_data();
        for (size_t i = 0; i < batch_size; ++i) {
            host_terminations[i] = term_data[i] ? 1 : 0;
            host_truncations[i] = trunc_data[i] ? 1 : 0;
        }

        err = cudaMemcpyAsync(terminations_buffer->typed_data(), host_terminations.data(),
                              batch_size * sizeof(uint8_t),
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            return ffi::Error::Internal(std::string("CUDA memcpy failed (terminations H2D): ") + cudaGetErrorString(err));
        }

        err = cudaMemcpyAsync(truncations_buffer->typed_data(), host_truncations.data(),
                              batch_size * sizeof(uint8_t),
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            return ffi::Error::Internal(std::string("CUDA memcpy failed (truncations H2D): ") + cudaGetErrorString(err));
        }

        // Check for any CUDA errors
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return ffi::Error::Internal(std::string("CUDA error after operations: ") + cudaGetErrorString(err));
        }

        return ffi::Error::Success();
    }
    catch (const std::exception& e) {
        std::string error_msg = "Exception during step (GPU): ";
        error_msg += e.what();
        std::cerr << error_msg << std::endl;
        return ffi::Error::Internal(error_msg);
    }
    catch (...) {
        std::cerr << "Unknown exception during step (GPU)" << std::endl;
        return ffi::Error::Internal("Unknown exception during step (GPU)");
    }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AtariVectorEnvXLAStepGPU, XLAStepGPUImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()  // CUDA stream
        .Arg<ffi::Buffer<ffi::U8>>()    // handle
        .Arg<ffi::Buffer<ffi::S32>>()   // actions
        .Arg<ffi::Buffer<ffi::F32>>()   // paddle_strength
        .Ret<ffi::Buffer<ffi::U8>>()    // new_handle
        .Ret<ffi::Buffer<ffi::U8>>()    // observations
        .Ret<ffi::Buffer<ffi::S32>>()   // rewards
        .Ret<ffi::Buffer<ffi::PRED>>()  // terminations
        .Ret<ffi::Buffer<ffi::PRED>>()  // truncations
        .Ret<ffi::Buffer<ffi::S32>>()   // env_ids
        .Ret<ffi::Buffer<ffi::S32>>()   // lives
        .Ret<ffi::Buffer<ffi::S32>>()   // frame_numbers
        .Ret<ffi::Buffer<ffi::S32>>()   // episode_frame_numbers
);
#endif  // __CUDACC__

template <typename T>
nb::capsule EncapsulateFFICall(T *fn) {
    // This check is optional, but it can be helpful for avoiding invalid handlers.
    static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                  "Encapsulated function must be and XLA FFI handler");
    return nb::capsule(reinterpret_cast<void *>(fn));
}

void init_vector_module_xla(nb::module_& m) {
    // CPU handlers
    m.def("VectorXLAReset", [] {return EncapsulateFFICall(AtariVectorEnvXLAReset); });
    m.def("VectorXLAStep", [] {return EncapsulateFFICall(AtariVectorEnvXLAStep); });

#ifdef __CUDACC__
    // GPU handlers
    m.def("VectorXLAResetGPU", [] {return EncapsulateFFICall(AtariVectorEnvXLAResetGPU); });
    m.def("VectorXLAStepGPU", [] {return EncapsulateFFICall(AtariVectorEnvXLAStepGPU); });
#endif
}
