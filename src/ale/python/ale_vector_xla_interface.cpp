#include "xla/ffi/api/ffi.h"

#include "ale/vector/async_vectorizer.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <cstring>  // For memcpy
#include <iostream>

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
    // Validate handle buffer size
    if (handle_buffer.element_count() != sizeof(ale::vector::AsyncVectorizer*)) {
        return ffi::Error::Internal("Incorrect handle buffer size in reset");
    }

    // Safely extract the vectorizer pointer from the handle buffer
    ale::vector::AsyncVectorizer* vectorizer = nullptr;
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

        // Reset the environments
        vectorizer->reset(reset_indices, reset_seeds);

        // Receive the observations after reset
        auto timesteps = vectorizer->recv();

        if (timesteps.empty()) {
            return ffi::Error::Internal("No timesteps received after step");
        } else if (timesteps.size() != vectorizer->get_batch_size()) {
            return ffi::Error::Internal("Number of timesteps is wrong");
        }

        size_t obs_size = vectorizer->get_obs_size();

        // Check if the observations buffer is large enough
        if (observations_buffer->element_count() != vectorizer->get_batch_size() * obs_size) {
            return ffi::Error::Internal("Observations buffer is the wrong size");
        }

        for (size_t i = 0; i < vectorizer->get_batch_size(); ++i) {
            const auto& timestep = timesteps[i];

            std::memcpy(
                observations_buffer->typed_data() + i * obs_size,
                timestep.observation.data(),
                obs_size
            );
            env_ids_buffer->typed_data()[i] = timestep.env_id;
            lives_buffer->typed_data()[i] = timestep.lives;
            frame_numbers_buffer->typed_data()[i] = timestep.frame_number;
            episode_frame_numbers_buffer->typed_data()[i] = timestep.episode_frame_number;
        }

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
    if (handle_buffer.element_count() != sizeof(ale::vector::AsyncVectorizer*)) {
        return ffi::Error::Internal("Incorrect handle buffer size in step");
    }

    // Safely extract the vectorizer pointer from the handle buffer
    ale::vector::AsyncVectorizer* vectorizer = nullptr;
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
        size_t num_envs = vectorizer->get_batch_size();

        if (action_id_buffer.element_count() != num_envs) {
            return ffi::Error::Internal("Action id buffer is the wrong size");
        } else if (paddle_strength_buffer.element_count() != num_envs) {
            return ffi::Error::Internal("Paddle strength buffer is the wrong size");
        }

        std::vector<ale::vector::EnvironmentAction> actions(num_envs);
        for (size_t i = 0; i < num_envs; ++i) {
            actions[i].env_id = i;
            actions[i].action_id = action_id_buffer.typed_data()[i];
            actions[i].paddle_strength = paddle_strength_buffer.typed_data()[i];
        }

        // Step the environments
        vectorizer->send(actions);

        // Receive the timesteps
        auto timesteps = vectorizer->recv();

        if (timesteps.empty()) {
            return ffi::Error::Internal("No timesteps received after step");
        } else if (timesteps.size() != vectorizer->get_batch_size()) {
            return ffi::Error::Internal("Number of timesteps is wrong");
        }

        size_t obs_size = vectorizer->get_obs_size();

        // Check if the observations buffer is large enough
        if (observations_buffer->element_count() != vectorizer->get_batch_size() * obs_size) {
            return ffi::Error::Internal("Observations buffer is the wrong size");
        }

        for (size_t i = 0; i < vectorizer->get_batch_size(); ++i) {
            const auto& timestep = timesteps[i];

            std::memcpy(
                observations_buffer->typed_data() + i * obs_size,
                timestep.observation.data(),
                obs_size
            );
            rewards_buffer->typed_data()[i] = timestep.reward;
            terminations_buffer->typed_data()[i] = timestep.terminated;
            truncations_buffer->typed_data()[i] = timestep.truncated;
            env_id_buffer->typed_data()[i] = timestep.env_id;
            lives_buffer->typed_data()[i] = timestep.lives;
            frame_numbers_buffer->typed_data()[i] = timestep.frame_number;
            episode_frame_numbers_buffer->typed_data()[i] = timestep.episode_frame_number;
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
        .Arg<ffi::Buffer<ffi::U8>>()  // handle
        .Arg<ffi::Buffer<ffi::S32>>()  // actions
        .Arg<ffi::Buffer<ffi::F32>>()  // paddle_strength
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

void init_vector_module_xla(py::module& m) {
    m.def("VectorXLAReset", [] {return EncapsulateFFICall(AtariVectorEnvXLAReset); });
    m.def("VectorXLAStep", [] {return EncapsulateFFICall(AtariVectorEnvXLAStep);});
}
