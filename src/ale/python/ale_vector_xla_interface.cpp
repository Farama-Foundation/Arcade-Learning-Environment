#include "xla/ffi/api/ffi.h"

#include "ale/vector/env_vectorizer.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <vector>
#include <cstring>  // For memcpy
#include <iostream>

#ifdef BUILD_VECTOR_XLA_GPU
#include <cuda_runtime.h>
#endif

// ExternalBuffers types the integer outputs as `int` (what BatchResult and
// OutputSlot use) while the FFI types them as int32_t; the step_into wiring
// hands one straight to the other.
static_assert(sizeof(int) == sizeof(int32_t),
              "ALE vector XLA step_into requires int to be 32-bit");

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

#ifdef BUILD_VECTOR_XLA_GPU
// Waits for the stream on scope exit, so host buffers that queued
// cudaMemcpyAsync calls read from outlive those copies on every return path,
// including early error returns and exceptions. Declare it after the buffers.
struct StreamSyncOnExit {
    cudaStream_t stream;
    ~StreamSyncOnExit() { cudaStreamSynchronize(stream); }
};

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
    // The handle is already on the device, so copy it across directly.
    err = cudaMemcpyAsync(new_handle_buffer->typed_data(), handle_buffer.typed_data(),
                          handle_buffer.element_count(),
                          cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA memcpy failed (new handle D2D): ") + cudaGetErrorString(err));
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
        StreamSyncOnExit sync_on_exit{stream};

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

        // The copies above read out of the BatchResult, which is destroyed when
        // this scope ends, so they must complete before we return. Sync here as
        // well as in sync_on_exit so that a failure is reported.
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            return ffi::Error::Internal(std::string("CUDA stream sync failed (reset results): ") + cudaGetErrorString(err));
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
#endif  // BUILD_VECTOR_XLA_GPU


#ifdef BUILD_VECTOR_XLA_GPU
namespace {

/// Page-locked host buffers the ALE workers write their results straight into,
/// plus the event that guards them.
///
/// Two things this buys over the copy-out path it replaces. H2D from
/// page-locked memory runs at roughly twice the bandwidth of pageable memory,
/// because the driver no longer has to stage the transfer through its own
/// bounce buffer. And because the buffers are owned by the vectorizer rather
/// than being stack temporaries, the copies issued from them stay valid after
/// the handler returns - so the handler does not have to synchronize the
/// stream, and the transfer overlaps the next round's emulation.
///
/// What makes that safe is h2d_done: the workers may not overwrite the buffers
/// until the previous round's transfer has drained them, so the next call waits
/// on the event before handing them to step_into().
struct PinnedStaging {
    std::size_t batch_size = 0;
    std::size_t obs_size = 0;
    uint8_t* obs = nullptr;
    int* env_ids = nullptr;
    int* rewards = nullptr;
    bool* terminations = nullptr;
    bool* truncations = nullptr;
    int* lives = nullptr;
    int* frame_numbers = nullptr;
    int* episode_frame_numbers = nullptr;
    cudaEvent_t h2d_done = nullptr;
    bool h2d_pending = false;

    PinnedStaging() = default;
    PinnedStaging(const PinnedStaging&) = delete;
    PinnedStaging& operator=(const PinnedStaging&) = delete;

    ~PinnedStaging() {
        // Never free page-locked memory with a transfer still reading it.
        if (h2d_done) {
            if (h2d_pending) {
                cudaEventSynchronize(h2d_done);
            }
            cudaEventDestroy(h2d_done);
        }
        cudaFreeHost(obs);
        cudaFreeHost(env_ids);
        cudaFreeHost(rewards);
        cudaFreeHost(terminations);
        cudaFreeHost(truncations);
        cudaFreeHost(lives);
        cudaFreeHost(frame_numbers);
        cudaFreeHost(episode_frame_numbers);
    }

    template <typename T>
    bool alloc(T** dst, std::size_t count) {
        void* p = nullptr;
        if (cudaHostAlloc(&p, count * sizeof(T), cudaHostAllocDefault) != cudaSuccess) {
            return false;
        }
        *dst = static_cast<T*>(p);
        return true;
    }

    bool allocate(std::size_t bs, std::size_t os) {
        batch_size = bs;
        obs_size = os;
        return alloc(&obs, bs * os) && alloc(&env_ids, bs) && alloc(&rewards, bs) &&
               alloc(&terminations, bs) && alloc(&truncations, bs) && alloc(&lives, bs) &&
               alloc(&frame_numbers, bs) && alloc(&episode_frame_numbers, bs) &&
               cudaEventCreateWithFlags(&h2d_done, cudaEventDisableTiming) == cudaSuccess;
    }

    ale::vector::BatchResult::ExternalBuffers as_external() const {
        ale::vector::BatchResult::ExternalBuffers b;
        b.observations = obs;
        b.env_ids = env_ids;
        b.rewards = rewards;
        b.terminations = terminations;
        b.truncations = truncations;
        b.lives = lives;
        b.frame_numbers = frame_numbers;
        b.episode_frame_numbers = episode_frame_numbers;
        b.final_observations = nullptr;  // the XLA path is NextStep-only
        return b;
    }
};

/// Fetch (or lazily build) the pinned staging owned by this vectorizer.
/// Returns nullptr if page-locked memory is unavailable, in which case the
/// caller falls back to copying out of the BatchResult.
PinnedStaging* pinned_staging_for(
    ale::vector::EnvVectorizer* vectorizer, std::size_t batch_size, std::size_t obs_size
) {
    auto& slot = vectorizer->backend_staging();
    auto* existing = static_cast<PinnedStaging*>(slot.get());
    if (existing && existing->batch_size == batch_size && existing->obs_size == obs_size) {
        return existing;
    }
    auto fresh = std::make_shared<PinnedStaging>();
    if (!fresh->allocate(batch_size, obs_size)) {
        return nullptr;
    }
    slot = fresh;  // shared_ptr<void> keeps PinnedStaging's deleter
    return fresh.get();
}

}  // namespace
#endif  // BUILD_VECTOR_XLA_GPU

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
        // send() and recv() operate on batch_size actions/results, not num_envs.
        // Python-side xla_step sends action buffers sized to batch_size and
        // allocates result buffers sized to batch_size, so match here.
        size_t batch_size = vectorizer->batch_size();

        if (action_id_buffer.element_count() != batch_size) {
            return ffi::Error::Internal("Action id buffer is the wrong size");
        } else if (paddle_strength_buffer.element_count() != batch_size) {
            return ffi::Error::Internal("Paddle strength buffer is the wrong size");
        }

        std::vector<ale::vector::Action> actions(batch_size);
        for (size_t i = 0; i < batch_size; ++i) {
            actions[i].env_id = static_cast<int>(i);
            actions[i].action_id = action_id_buffer.typed_data()[i];
            actions[i].paddle_strength = paddle_strength_buffer.typed_data()[i];
            actions[i].force_reset = false;
        }

        size_t stacked_obs_size = vectorizer->stacked_obs_size();

        // Check if the observations buffer is large enough
        if (observations_buffer->element_count() != batch_size * stacked_obs_size) {
            return ffi::Error::Internal("Observations buffer is the wrong size");
        }

        // Ask the workers to write this round straight into XLA's output
        // buffers. Without this the batch is written once into a BatchResult
        // and then copied out again in full - measured at 8.4-8.7% of this
        // handler at 256-512 envs, since the observations alone are 14.5 MB per
        // step at n=512. XLA owns these buffers and they outlive the call, and
        // recv() below returns a non-owning view over them.
        //
        // Declined in async mode (batch_size < num_envs), where envs still in
        // flight from the previous round would stage into these buffers after
        // the handler has returned; the copy-out below is the fallback.
        ale::vector::BatchResult::ExternalBuffers out;
        out.observations = observations_buffer->typed_data();
        out.env_ids = env_id_buffer->typed_data();
        out.rewards = rewards_buffer->typed_data();
        out.terminations = terminations_buffer->typed_data();
        out.truncations = truncations_buffer->typed_data();
        out.lives = lives_buffer->typed_data();
        out.frame_numbers = frame_numbers_buffer->typed_data();
        out.episode_frame_numbers = episode_frame_numbers_buffer->typed_data();
        out.final_observations = nullptr;  // the XLA path is NextStep-only

        const bool wrote_in_place = vectorizer->step_into(out);

        // Step the environments
        vectorizer->send(actions);

        // Receive the results
        auto result = vectorizer->recv();

        if (wrote_in_place) {
            // Workers already filled every output buffer; nothing to copy.
            return ffi::Error::Success();
        }

        size_t result_batch_size = result.batch_size();

        // Copy data from BatchResult to output buffers
        std::memcpy(observations_buffer->typed_data(), result.obs_data(),
                    result_batch_size * stacked_obs_size);
        std::memcpy(rewards_buffer->typed_data(), result.rewards_data(),
                    result_batch_size * sizeof(int32_t));
        std::memcpy(env_id_buffer->typed_data(), result.env_ids_data(),
                    result_batch_size * sizeof(int32_t));
        std::memcpy(lives_buffer->typed_data(), result.lives_data(),
                    result_batch_size * sizeof(int32_t));
        std::memcpy(frame_numbers_buffer->typed_data(), result.frame_numbers_data(),
                    result_batch_size * sizeof(int32_t));
        std::memcpy(episode_frame_numbers_buffer->typed_data(), result.episode_frame_numbers_data(),
                    result_batch_size * sizeof(int32_t));

        // Copy bools element-wise (bool* to PRED buffer)
        const bool* term_data = result.terminations_data();
        const bool* trunc_data = result.truncations_data();
        for (size_t i = 0; i < result_batch_size; ++i) {
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

#ifdef BUILD_VECTOR_XLA_GPU
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
    // The handle is already on the device, so copy it across directly.
    err = cudaMemcpyAsync(new_handle_buffer->typed_data(), handle_buffer.typed_data(),
                          handle_buffer.element_count(),
                          cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA memcpy failed (new handle D2D): ") + cudaGetErrorString(err));
    }

    try {
        // send() and recv() operate on batch_size actions/results, not num_envs.
        size_t batch_size = vectorizer->batch_size();

        if (action_id_buffer.element_count() != batch_size) {
            return ffi::Error::Internal("Action id buffer is the wrong size (GPU)");
        } else if (paddle_strength_buffer.element_count() != batch_size) {
            return ffi::Error::Internal("Paddle strength buffer is the wrong size (GPU)");
        }

        // Copy action IDs and paddle strength from GPU to CPU
        std::vector<int32_t> host_action_ids(batch_size);
        std::vector<float> host_paddle_strength(batch_size);

        err = cudaMemcpyAsync(host_action_ids.data(), action_id_buffer.typed_data(),
                              batch_size * sizeof(int32_t),
                              cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) {
            return ffi::Error::Internal(std::string("CUDA memcpy failed (actions D2H): ") + cudaGetErrorString(err));
        }

        err = cudaMemcpyAsync(host_paddle_strength.data(), paddle_strength_buffer.typed_data(),
                              batch_size * sizeof(float),
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
        std::vector<ale::vector::Action> actions(batch_size);
        for (size_t i = 0; i < batch_size; ++i) {
            actions[i].env_id = static_cast<int>(i);
            actions[i].action_id = host_action_ids[i];
            actions[i].paddle_strength = host_paddle_strength[i];
            actions[i].force_reset = false;
        }

        const size_t stacked_obs_size = vectorizer->stacked_obs_size();
        if (observations_buffer->element_count() != batch_size * stacked_obs_size) {
            return ffi::Error::Internal("Observations buffer is the wrong size (GPU)");
        }

        // Have the workers write this round straight into page-locked host
        // memory, so the batch is written once and the H2D that follows runs at
        // pinned bandwidth. Declined in async mode (see stage_into) and when
        // page-locked memory is unavailable; the copy-out below is the fallback.
        // stage_into is ordered-mode only, so in async mode there is no point
        // page-locking buffers it will refuse - that would be 14.5 MB per
        // vectorizer of pinned memory doing nothing.
        const bool ordered = vectorizer->batch_size() == vectorizer->num_envs();
        PinnedStaging* pinned = ordered
            ? pinned_staging_for(vectorizer, batch_size, stacked_obs_size)
            : nullptr;
        bool wrote_pinned = false;
        if (pinned != nullptr) {
            // The previous round's transfer still reads these buffers - it must
            // drain before the workers overwrite them.
            if (pinned->h2d_pending) {
                err = cudaEventSynchronize(pinned->h2d_done);
                if (err != cudaSuccess) {
                    return ffi::Error::Internal(std::string("CUDA event sync failed: ") + cudaGetErrorString(err));
                }
                pinned->h2d_pending = false;
            }
            const auto external = pinned->as_external();
            wrote_pinned = vectorizer->step_into(external);
        }

        // Step the environments (CPU operation)
        vectorizer->send(actions);

        // Receive the results
        auto result = vectorizer->recv();

        size_t result_batch_size = result.batch_size();

        if (wrote_pinned) {
            const struct { void* dst; const void* src; size_t bytes; const char* what; } copies[] = {
                {observations_buffer->typed_data(), pinned->obs, result_batch_size * stacked_obs_size, "observations"},
                {rewards_buffer->typed_data(), pinned->rewards, result_batch_size * sizeof(int32_t), "rewards"},
                {env_id_buffer->typed_data(), pinned->env_ids, result_batch_size * sizeof(int32_t), "env_ids"},
                {lives_buffer->typed_data(), pinned->lives, result_batch_size * sizeof(int32_t), "lives"},
                {frame_numbers_buffer->typed_data(), pinned->frame_numbers, result_batch_size * sizeof(int32_t), "frame_numbers"},
                {episode_frame_numbers_buffer->typed_data(), pinned->episode_frame_numbers, result_batch_size * sizeof(int32_t), "episode_frame_numbers"},
                {terminations_buffer->typed_data(), pinned->terminations, result_batch_size * sizeof(bool), "terminations"},
                {truncations_buffer->typed_data(), pinned->truncations, result_batch_size * sizeof(bool), "truncations"},
            };
            for (const auto& c : copies) {
                err = cudaMemcpyAsync(c.dst, c.src, c.bytes, cudaMemcpyHostToDevice, stream);
                if (err != cudaSuccess) {
                    return ffi::Error::Internal(std::string("CUDA memcpy failed (") + c.what +
                                                " pinned H2D): " + cudaGetErrorString(err));
                }
            }
            // No stream synchronize: the source buffers belong to the vectorizer
            // and outlive this call, so the transfer may overlap the next
            // round's emulation. The event is what stops the workers from
            // overwriting the buffers before it has finished.
            err = cudaEventRecord(pinned->h2d_done, stream);
            if (err != cudaSuccess) {
                return ffi::Error::Internal(std::string("CUDA event record failed: ") + cudaGetErrorString(err));
            }
            pinned->h2d_pending = true;
            return ffi::Error::Success();
        }

        // Copy bools element-wise to temporary buffers, sent to the GPU below
        std::vector<uint8_t> host_terminations(result_batch_size);
        std::vector<uint8_t> host_truncations(result_batch_size);
        const bool* term_data = result.terminations_data();
        const bool* trunc_data = result.truncations_data();
        for (size_t i = 0; i < result_batch_size; ++i) {
            host_terminations[i] = term_data[i] ? 1 : 0;
            host_truncations[i] = trunc_data[i] ? 1 : 0;
        }

        StreamSyncOnExit sync_on_exit{stream};

        // Copy data from BatchResult to GPU buffers via host memory
        err = cudaMemcpyAsync(observations_buffer->typed_data(), result.obs_data(),
                              result_batch_size * stacked_obs_size,
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            return ffi::Error::Internal(std::string("CUDA memcpy failed (observations H2D): ") + cudaGetErrorString(err));
        }

        err = cudaMemcpyAsync(rewards_buffer->typed_data(), result.rewards_data(),
                              result_batch_size * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            return ffi::Error::Internal(std::string("CUDA memcpy failed (rewards H2D): ") + cudaGetErrorString(err));
        }

        err = cudaMemcpyAsync(env_id_buffer->typed_data(), result.env_ids_data(),
                              result_batch_size * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            return ffi::Error::Internal(std::string("CUDA memcpy failed (env_ids H2D): ") + cudaGetErrorString(err));
        }

        err = cudaMemcpyAsync(lives_buffer->typed_data(), result.lives_data(),
                              result_batch_size * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            return ffi::Error::Internal(std::string("CUDA memcpy failed (lives H2D): ") + cudaGetErrorString(err));
        }

        err = cudaMemcpyAsync(frame_numbers_buffer->typed_data(), result.frame_numbers_data(),
                              result_batch_size * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            return ffi::Error::Internal(std::string("CUDA memcpy failed (frame_numbers H2D): ") + cudaGetErrorString(err));
        }

        err = cudaMemcpyAsync(episode_frame_numbers_buffer->typed_data(), result.episode_frame_numbers_data(),
                              result_batch_size * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            return ffi::Error::Internal(std::string("CUDA memcpy failed (episode_frame_numbers H2D): ") + cudaGetErrorString(err));
        }

        err = cudaMemcpyAsync(terminations_buffer->typed_data(), host_terminations.data(),
                              result_batch_size * sizeof(uint8_t),
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            return ffi::Error::Internal(std::string("CUDA memcpy failed (terminations H2D): ") + cudaGetErrorString(err));
        }

        err = cudaMemcpyAsync(truncations_buffer->typed_data(), host_truncations.data(),
                              result_batch_size * sizeof(uint8_t),
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            return ffi::Error::Internal(std::string("CUDA memcpy failed (truncations H2D): ") + cudaGetErrorString(err));
        }

        // Check for any CUDA errors
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return ffi::Error::Internal(std::string("CUDA error after operations: ") + cudaGetErrorString(err));
        }

        // Every copy above reads from memory that dies with this call - the
        // BatchResult and the two host_* vectors - so the transfers have to
        // finish before we return. Sync here as well as in sync_on_exit so that
        // a failure is reported.
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            return ffi::Error::Internal(std::string("CUDA stream sync failed (step results): ") + cudaGetErrorString(err));
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
#endif  // BUILD_VECTOR_XLA_GPU

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

#ifdef BUILD_VECTOR_XLA_GPU
    // GPU handlers
    m.def("VectorXLAResetGPU", [] {return EncapsulateFFICall(AtariVectorEnvXLAResetGPU); });
    m.def("VectorXLAStepGPU", [] {return EncapsulateFFICall(AtariVectorEnvXLAStepGPU); });
#endif
}
