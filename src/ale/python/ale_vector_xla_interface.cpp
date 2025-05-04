
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "ale/vector/async_vectorizer.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>

namespace ffi = xla::ffi;
namespace py = pybind11;

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
    ale::vector::AsyncVectorizer* vectorizer = reinterpret_cast<ale::vector::AsyncVectorizer*>(handle_buffer.typed_data());

    std::copy(handle_buffer.typed_data(),
              handle_buffer.typed_data() + 8,
              new_handle_buffer->typed_data());

    const std::vector<int> reset_indices(reset_indices_buffer.typed_data(), reset_indices_buffer.typed_data() + reset_indices_buffer.element_count());
    const std::vector<int> reset_seeds(reset_seeds_buffer.typed_data(), reset_seeds_buffer.typed_data() + reset_seeds_buffer.element_count());
    vectorizer->reset(reset_indices, reset_seeds);
    auto timesteps = vectorizer->recv();

    // Copy data to output buffers
    uint8_t* obs_ptr = observations_buffer->typed_data();
    int32_t* env_ids_ptr = env_ids_buffer->typed_data();
    int32_t* lives_ptr = lives_buffer->typed_data();
    int32_t* frame_numbers_ptr = frame_numbers_buffer->typed_data();
    int32_t* episode_frame_numbers_ptr = episode_frame_numbers_buffer->typed_data();

    // Copy data to output buffers
    size_t obs_size = vectorizer->get_obs_size();
    for (size_t i = 0; i < timesteps.size(); ++i) {
        const auto& timestep = timesteps[i];

        std::copy(timestep.observation.data(),
                  timestep.observation.data() + obs_size,
                  obs_ptr + i * obs_size);
        env_ids_ptr[i] = timestep.env_id;
        lives_ptr[i] = timestep.lives;
        frame_numbers_ptr[i] = timestep.frame_number;
        episode_frame_numbers_ptr[i] = timestep.episode_frame_number;
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AtariVectorEnvXLAReset, XLAResetImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::U8>>()  // handle
        .Arg<ffi::Buffer<ffi::S32>>()  // reset_indices
        .Arg<ffi::Buffer<ffi::S32>>()  // reset_seeds
        .Ret<ffi::Buffer<ffi::U8>>()  // new_handle
        .Ret<ffi::Buffer<ffi::U8>>()   // observations
        .Ret<ffi::Buffer<ffi::S32>>()  // env_ids
        .Ret<ffi::Buffer<ffi::S32>>()  // lives
        .Ret<ffi::Buffer<ffi::S32>>()  // frame_numbers
        .Ret<ffi::Buffer<ffi::S32>>()  // episode_frame_numbers
);

ffi::Error XLAStepImpl(
    ffi::Buffer<ffi::U8> handle_buffer,
    ffi::Buffer<ffi::S32> actions_buffer,
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
    ale::vector::AsyncVectorizer* vectorizer = reinterpret_cast<ale::vector::AsyncVectorizer*>(handle_buffer.typed_data());

    // Copy handle to output
    std::copy(handle_buffer.typed_data(),
              handle_buffer.typed_data() + 8,
              new_handle_buffer->typed_data());

    // Create actions for the vectorizer
    std::vector<ale::vector::EnvironmentAction> actions(vectorizer->get_num_envs());
    for (int i = 0; i < vectorizer->get_num_envs(); ++i) {
        actions[i].env_id = i;
        actions[i].action_id = actions_buffer.typed_data()[i];
        actions[i].paddle_strength = paddle_strength_buffer.typed_data()[i];
    }

    // Step the environments
    vectorizer->send(actions);
    auto timesteps = vectorizer->recv();

    // Get shape info to properly format the observations
    size_t obs_size = vectorizer->get_obs_size();

    // Copy data to output buffers
    uint8_t* obs_ptr = observations_buffer->typed_data();
    int* rewards_ptr = rewards_buffer->typed_data();
    bool* terminations_ptr = terminations_buffer->typed_data();
    bool* truncations_ptr = truncations_buffer->typed_data();
    int* env_ids_ptr = env_id_buffer->typed_data();
    int* lives_ptr = lives_buffer->typed_data();
    int* frame_numbers_ptr = frame_numbers_buffer->typed_data();
    int* episode_frame_numbers_ptr = episode_frame_numbers_buffer->typed_data();

    for (int i = 0; i < vectorizer->get_num_envs(); ++i) {
        const auto& timestep = timesteps[i];

        // Copy observation data
        std::copy(timestep.observation.data(),
                  timestep.observation.data() + obs_size,
                  obs_ptr + i * obs_size);

        // Copy other fields
        rewards_ptr[i] = timestep.reward;
        terminations_ptr[i] = timestep.terminated ? 1 : 0;
        truncations_ptr[i] = timestep.truncated ? 1 : 0;
        env_ids_ptr[i] = timestep.env_id;
        lives_ptr[i] = timestep.lives;
        frame_numbers_ptr[i] = timestep.frame_number;
        episode_frame_numbers_ptr[i] = timestep.episode_frame_number;
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AtariVectorEnvXLAStep, XLAStepImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::U8>>()  // handle
        .Arg<ffi::Buffer<ffi::S32>>()  // actions
        .Arg<ffi::Buffer<ffi::F32>>()  // paddle_strength (optional)
        .Ret<ffi::Buffer<ffi::U8>>()  // new_handle
        .Ret<ffi::Buffer<ffi::U8>>()   // observations
        .Ret<ffi::Buffer<ffi::S32>>()  // rewards
        .Ret<ffi::Buffer<ffi::PRED>>()   // terminations
        .Ret<ffi::Buffer<ffi::PRED>>()   // truncations
        .Ret<ffi::Buffer<ffi::S32>>()  // env_ids
        .Ret<ffi::Buffer<ffi::S32>>()  // lives
        .Ret<ffi::Buffer<ffi::S32>>()  // frame_numbers
        .Ret<ffi::Buffer<ffi::S32>>()  // episode_frame_numbers
);


template <typename T>
py::capsule EncapsulateFFICall(T *fn) {
    // This check is optional, but it can be helpful for avoiding invalid handlers.
    static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                  "Encapsulated function must be and XLA FFI handler");
    return py::capsule(reinterpret_cast<void *>(fn));
}


void init_xla_module(py::module& m) {
    m.def("VectorXLAReset", [] {return EncapsulateFFICall(AtariVectorEnvXLAReset); });
    m.def("VectorXLAStep", [] {return EncapsulateFFICall(AtariVectorEnvXLAStep);});
}
