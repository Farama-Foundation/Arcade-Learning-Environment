#include "ale_vector_python_interface.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/filesystem.h>
#include <vector>
#include <cmath>
#include <tuple>

namespace nb = nanobind;

// Function to add vector environment bindings to an existing module
void init_vector_module(nb::module_& m) {
    // Define ALEVectorInterface class
    nb::class_<ale::vector::ALEVectorInterface>(m, "ALEVectorInterface")
        .def(nb::init<const fs::path, int, int, int, int, int, bool, bool, int, bool, bool, bool, bool, int, float, bool, int, int, int>(),
             "rom_path"_a,
             "num_envs"_a,
             "frame_skip"_a = 4,
             "stack_num"_a = 4,
             "img_height"_a = 84,
             "img_width"_a = 84,
             "grayscale"_a = true,
             "maxpool"_a = true,
             "noop_max"_a = 30,
             "use_fire_reset"_a = true,
             "episodic_life"_a = false,
             "life_loss_info"_a = false,
             "reward_clipping"_a = true,
             "max_episode_steps"_a = 108000,
             "repeat_action_probability"_a = 0.0f,
             "full_action_space"_a = false,
             "batch_size"_a = 0,
             "num_threads"_a = 0,
             "thread_affinity_offset"_a = -1)
        .def("reset", [](ale::vector::ALEVectorInterface& self, const std::vector<int> reset_indices, const std::vector<int> reset_seeds) {
            // Call C++ reset method with GIL released
            nb::gil_scoped_release release;
            auto timesteps = self.reset(reset_indices, reset_seeds);
            nb::gil_scoped_acquire acquire;

            // Get shape information
            int num_envs = timesteps.size();
            auto obs_shape = self.get_observation_shape();
            int stack_num = std::get<0>(obs_shape);
            int height = std::get<1>(obs_shape);
            int width = std::get<2>(obs_shape);

            // Create a single NumPy array for all observations
            nb::ndarray<nb::numpy, uint8_t> observations;
            if (self.is_grayscale()) {
                observations = nb::ndarray<nb::numpy, uint8_t>(
                    nullptr, {num_envs, stack_num, height, width});
            } else {
                observations = nb::ndarray<nb::numpy, uint8_t>(
                    nullptr, {num_envs, stack_num, height, width, 3});
            }
            auto observations_ptr = observations.data();

            // Create arrays for info fields
            auto env_ids = nb::ndarray<nb::numpy, int>(nullptr, {num_envs});
            auto lives = nb::ndarray<nb::numpy, int>(nullptr, {num_envs});
            auto frame_numbers = nb::ndarray<nb::numpy, int>(nullptr, {num_envs});
            auto episode_frame_numbers = nb::ndarray<nb::numpy, int>(nullptr, {num_envs});

            auto env_ids_ptr = env_ids.data();
            auto lives_ptr = lives.data();
            auto frame_numbers_ptr = frame_numbers.data();
            auto episode_frame_numbers_ptr = episode_frame_numbers.data();

            // Copy data from observations to NumPy arrays
            int channels = self.is_grayscale() ? 1 : 3;
            size_t obs_size = stack_num * height * width * channels;
            for (int i = 0; i < num_envs; i++) {
                const auto& timestep = timesteps[i];

                // Copy screen data
                std::memcpy(
                    observations_ptr + i * obs_size,
                    timestep.observation.data(),
                    obs_size * sizeof(uint8_t)
                );

                // Copy info fields
                env_ids_ptr[i] = timestep.env_id;
                lives_ptr[i] = timestep.lives;
                frame_numbers_ptr[i] = timestep.frame_number;
                episode_frame_numbers_ptr[i] = timestep.episode_frame_number;
            }

            // Create info dict
            nb::dict info;
            info["env_id"] = env_ids;
            info["lives"] = lives;
            info["frame_number"] = frame_numbers;
            info["episode_frame_number"] = episode_frame_numbers;

            return nb::make_tuple(observations, info);
        })
        .def("send", [](ale::vector::ALEVectorInterface& self, const std::vector<int> action_ids, const std::vector<float> paddle_strengths) {
            self.send(action_ids, paddle_strengths);
        })
        .def("recv", [](ale::vector::ALEVectorInterface& self) {
            const auto timesteps = self.recv();
            nb::gil_scoped_acquire acquire;

            // Get shape information
            int num_envs = timesteps.size();
            const auto shape_info = self.get_observation_shape();
            int stack_num = std::get<0>(shape_info);
            int height = std::get<1>(shape_info);
            int width = std::get<2>(shape_info);

            // Create NumPy arrays
            nb::ndarray<nb::numpy, uint8_t> observations;
            if (self.is_grayscale()) {
                observations = nb::ndarray<nb::numpy, uint8_t>(
                    nullptr, {num_envs, stack_num, height, width});
            } else {
                observations = nb::ndarray<nb::numpy, uint8_t>(
                    nullptr, {num_envs, stack_num, height, width, 3});
            }

            auto rewards = nb::ndarray<nb::numpy, int>(nullptr, {num_envs});
            auto terminations = nb::ndarray<nb::numpy, bool>(nullptr, {num_envs});
            auto truncations = nb::ndarray<nb::numpy, bool>(nullptr, {num_envs});
            auto env_ids = nb::ndarray<nb::numpy, int>(nullptr, {num_envs});
            auto lives = nb::ndarray<nb::numpy, int>(nullptr, {num_envs});
            auto frame_numbers = nb::ndarray<nb::numpy, int>(nullptr, {num_envs});
            auto episode_frame_numbers = nb::ndarray<nb::numpy, int>(nullptr, {num_envs});

            // Get pointers to the arrays' data
            auto observations_ptr = observations.data();
            auto rewards_ptr = rewards.data();
            auto terminations_ptr = terminations.data();
            auto truncations_ptr = truncations.data();
            auto env_ids_ptr = env_ids.data();
            auto lives_ptr = lives.data();
            auto frame_numbers_ptr = frame_numbers.data();
            auto episode_frame_numbers_ptr = episode_frame_numbers.data();

            // Copy data from observations to NumPy arrays
            int channels = self.is_grayscale() ? 1 : 3;
            const size_t obs_size = stack_num * height * width * channels;
            for (int i = 0; i < num_envs; i++) {
                const auto& timestep = timesteps[i];

                // Copy screen data
                std::memcpy(
                    observations_ptr + i * obs_size,
                    timestep.observation.data(),
                    obs_size * sizeof(uint8_t)
                );

                // Copy other fields
                rewards_ptr[i] = timestep.reward;
                terminations_ptr[i] = timestep.terminated;
                truncations_ptr[i] = timestep.truncated;
                env_ids_ptr[i] = timestep.env_id;
                lives_ptr[i] = timestep.lives;
                frame_numbers_ptr[i] = timestep.frame_number;
                episode_frame_numbers_ptr[i] = timestep.episode_frame_number;
            }

            // Create info dict
            nb::dict info;
            info["env_id"] = env_ids;
            info["lives"] = lives;
            info["frame_number"] = frame_numbers;
            info["episode_frame_number"] = episode_frame_numbers;

            return nb::make_tuple(observations, rewards, terminations, truncations, info);
        })::numpy, uint8_t>(
                    nb::steal(nb::detail::ndarray_new(
                        nb::handle(nb::type<uint8_t>().raw_type()),
                        5,
                        new size_t[5]{size_t(num_envs), size_t(stack_num), size_t(height), size_t(width), size_t(3)},
                        nb::handle(),
                        nullptr,
                        nb::dtype<uint8_t>(),
                        nb::device::cpu::value,
                        0,
                        'C'
                    ))
                );
            }

            auto rewards = nb::ndarray<nb::numpy, int>(
                nb::steal(nb::detail::ndarray_new(
                    nb::handle(nb::type<int>().raw_type()),
                    1,
                    new size_t[1]{size_t(num_envs)},
                    nb::handle(),
                    nullptr,
                    nb::dtype<int>(),
                    nb::device::cpu::value,
                    0,
                    'C'
                ))
            );
            auto terminations = nb::ndarray<nb::numpy, bool>(
                nb::steal(nb::detail::ndarray_new(
                    nb::handle(nb::type<bool>().raw_type()),
                    1,
                    new size_t[1]{size_t(num_envs)},
                    nb::handle(),
                    nullptr,
                    nb::dtype<bool>(),
                    nb::device::cpu::value,
                    0,
                    'C'
                ))
            );
            auto truncations = nb::ndarray<nb::numpy, bool>(
                nb::steal(nb::detail::ndarray_new(
                    nb::handle(nb::type<bool>().raw_type()),
                    1,
                    new size_t[1]{size_t(num_envs)},
                    nb::handle(),
                    nullptr,
                    nb::dtype<bool>(),
                    nb::device::cpu::value,
                    0,
                    'C'
                ))
            );
            auto env_ids = nb::ndarray<nb::numpy, int>(
                nb::steal(nb::detail::ndarray_new(
                    nb::handle(nb::type<int>().raw_type()),
                    1,
                    new size_t[1]{size_t(num_envs)},
                    nb::handle(),
                    nullptr,
                    nb::dtype<int>(),
                    nb::device::cpu::value,
                    0,
                    'C'
                ))
            );
            auto lives = nb::ndarray<nb::numpy, int>(
                nb::steal(nb::detail::ndarray_new(
                    nb::handle(nb::type<int>().raw_type()),
                    1,
                    new size_t[1]{size_t(num_envs)},
                    nb::handle(),
                    nullptr,
                    nb::dtype<int>(),
                    nb::device::cpu::value,
                    0,
                    'C'
                ))
            );
            auto frame_numbers = nb::ndarray<nb::numpy, int>(
                nb::steal(nb::detail::ndarray_new(
                    nb::handle(nb::type<int>().raw_type()),
                    1,
                    new size_t[1]{size_t(num_envs)},
                    nb::handle(),
                    nullptr,
                    nb::dtype<int>(),
                    nb::device::cpu::value,
                    0,
                    'C'
                ))
            );
            auto episode_frame_numbers = nb::ndarray<nb::numpy, int>(
                nb::steal(nb::detail::ndarray_new(
                    nb::handle(nb::type<int>().raw_type()),
                    1,
                    new size_t[1]{size_t(num_envs)},
                    nb::handle(),
                    nullptr,
                    nb::dtype<int>(),
                    nb::device::cpu::value,
                    0,
                    'C'
                ))
            );

            // Get pointers to the arrays' data
            auto observations_ptr = observations.data();
            auto rewards_ptr = rewards.data();
            auto terminations_ptr = terminations.data();
            auto truncations_ptr = truncations.data();
            auto env_ids_ptr = env_ids.data();
            auto lives_ptr = lives.data();
            auto frame_numbers_ptr = frame_numbers.data();
            auto episode_frame_numbers_ptr = episode_frame_numbers.data();

            // Copy data from observations to NumPy arrays
            const size_t obs_size = stack_num * height * width * channels;
            for (int i = 0; i < num_envs; i++) {
                const auto& timestep = timesteps[i];

                // Copy screen data
                std::memcpy(
                    observations_ptr + i * obs_size,
                    timestep.observation.data(),
                    obs_size * sizeof(uint8_t)
                );

                // Copy other fields
                rewards_ptr[i] = timestep.reward;
                terminations_ptr[i] = timestep.terminated;
                truncations_ptr[i] = timestep.truncated;
                env_ids_ptr[i] = timestep.env_id;
                lives_ptr[i] = timestep.lives;
                frame_numbers_ptr[i] = timestep.frame_number;
                episode_frame_numbers_ptr[i] = timestep.episode_frame_number;
            }

            // Create info dict
            nb::dict info;
            info["env_id"] = env_ids;
            info["lives"] = lives;
            info["frame_number"] = frame_numbers;
            info["episode_frame_number"] = episode_frame_numbers;

            return nb::make_tuple(observations, rewards, terminations, truncations, info);
        })
        .def("get_action_set", &ale::vector::ALEVectorInterface::get_action_set)
        .def("get_num_envs", &ale::vector::ALEVectorInterface::get_num_envs)
        .def("get_observation_shape", [](ale::vector::ALEVectorInterface& self) {
            auto shape = self.get_observation_shape();
            if (self.is_grayscale()) {
                return nb::make_tuple(std::get<0>(shape), std::get<1>(shape), std::get<2>(shape));
            } else {
                return nb::make_tuple(std::get<0>(shape), std::get<1>(shape), std::get<2>(shape), std::get<3>(shape));
            }
        })
        .def("handle", [](ale::vector::ALEVectorInterface& self) {
            // Get the raw pointer to the AsyncVectorizer
            auto ptr = self.get_vectorizer();

            // Create a NumPy array with the correct size to hold the pointer
            auto handle_array = nb::ndarray<nb::numpy, uint8_t>(nullptr, {sizeof(ptr)});
            auto handle_ptr = handle_array.data();

            // Copy the pointer value into the byte array
            std::memcpy(handle_ptr, &ptr, sizeof(ptr));

            return handle_array;
        });
}