#include "ale_vector_python_interface.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/ndarray.h>

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
            int channels = self.is_grayscale() ? 1 : 3;

            // Create a single NumPy array for all observations
            nb::ndarray<nb::numpy, uint8_t> observations;
            if (self.is_grayscale()) {
                size_t shape[4] = {static_cast<size_t>(num_envs), static_cast<size_t>(stack_num), static_cast<size_t>(height), static_cast<size_t>(width)};
                observations = nb::steal(nb::detail::ndarray_new(
                    nb::handle(nb::dtype<uint8_t>().raw_dtype()),
                    4, shape, nb::handle(nullptr), nullptr, nb::dtype<uint8_t>()));
            } else {
                size_t shape[5] = {static_cast<size_t>(num_envs), static_cast<size_t>(stack_num), static_cast<size_t>(height), static_cast<size_t>(width), 3};
                observations = nb::steal(nb::detail::ndarray_new(
                    nb::handle(nb::dtype<uint8_t>().raw_dtype()),
                    5, shape, nb::handle(nullptr), nullptr, nb::dtype<uint8_t>()));
            }
            auto observations_ptr = observations.data();

            // Create arrays for info fields
            size_t env_shape[1] = {static_cast<size_t>(num_envs)};

            auto env_ids = nb::steal(nb::detail::ndarray_new(
                nb::handle(nb::dtype<int>().raw_dtype()),
                1, env_shape, nb::handle(nullptr), nullptr, nb::dtype<int>()));
            auto lives = nb::steal(nb::detail::ndarray_new(
                nb::handle(nb::dtype<int>().raw_dtype()),
                1, env_shape, nb::handle(nullptr), nullptr, nb::dtype<int>()));
            auto frame_numbers = nb::steal(nb::detail::ndarray_new(
                nb::handle(nb::dtype<int>().raw_dtype()),
                1, env_shape, nb::handle(nullptr), nullptr, nb::dtype<int>()));
            auto episode_frame_numbers = nb::steal(nb::detail::ndarray_new(
                nb::handle(nb::dtype<int>().raw_dtype()),
                1, env_shape, nb::handle(nullptr), nullptr, nb::dtype<int>()));

            auto env_ids_array = nb::cast<nb::ndarray<nb::numpy, int>>(env_ids);
            auto lives_array = nb::cast<nb::ndarray<nb::numpy, int>>(lives);
            auto frame_numbers_array = nb::cast<nb::ndarray<nb::numpy, int>>(frame_numbers);
            auto episode_frame_numbers_array = nb::cast<nb::ndarray<nb::numpy, int>>(episode_frame_numbers);

            // Copy data from observations to NumPy arrays
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
                env_ids_array.data()[i] = timestep.env_id;
                lives_array.data()[i] = timestep.lives;
                frame_numbers_array.data()[i] = timestep.frame_number;
                episode_frame_numbers_array.data()[i] = timestep.episode_frame_number;
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
            int channels = self.is_grayscale() ? 1 : 3;

            // Create NumPy arrays
            nb::ndarray<nb::numpy, uint8_t> observations;
            if (self.is_grayscale()) {
                size_t shape[4] = {static_cast<size_t>(num_envs), static_cast<size_t>(stack_num), static_cast<size_t>(height), static_cast<size_t>(width)};
                observations = nb::steal(nb::detail::ndarray_new(
                    nb::handle(nb::dtype<uint8_t>().raw_dtype()),
                    4, shape, nb::handle(nullptr), nullptr, nb::dtype<uint8_t>()));
            } else {
                size_t shape[5] = {static_cast<size_t>(num_envs), static_cast<size_t>(stack_num), static_cast<size_t>(height), static_cast<size_t>(width), 3};
                observations = nb::steal(nb::detail::ndarray_new(
                    nb::handle(nb::dtype<uint8_t>().raw_dtype()),
                    5, shape, nb::handle(nullptr), nullptr, nb::dtype<uint8_t>()));
            }

            size_t env_shape[1] = {static_cast<size_t>(num_envs)};

            auto rewards = nb::steal(nb::detail::ndarray_new(
                nb::handle(nb::dtype<int>().raw_dtype()),
                1, env_shape, nb::handle(nullptr), nullptr, nb::dtype<int>()));
            auto terminations = nb::steal(nb::detail::ndarray_new(
                nb::handle(nb::dtype<bool>().raw_dtype()),
                1, env_shape, nb::handle(nullptr), nullptr, nb::dtype<bool>()));
            auto truncations = nb::steal(nb::detail::ndarray_new(
                nb::handle(nb::dtype<bool>().raw_dtype()),
                1, env_shape, nb::handle(nullptr), nullptr, nb::dtype<bool>()));
            auto env_ids = nb::steal(nb::detail::ndarray_new(
                nb::handle(nb::dtype<int>().raw_dtype()),
                1, env_shape, nb::handle(nullptr), nullptr, nb::dtype<int>()));
            auto lives = nb::steal(nb::detail::ndarray_new(
                nb::handle(nb::dtype<int>().raw_dtype()),
                1, env_shape, nb::handle(nullptr), nullptr, nb::dtype<int>()));
            auto frame_numbers = nb::steal(nb::detail::ndarray_new(
                nb::handle(nb::dtype<int>().raw_dtype()),
                1, env_shape, nb::handle(nullptr), nullptr, nb::dtype<int>()));
            auto episode_frame_numbers = nb::steal(nb::detail::ndarray_new(
                nb::handle(nb::dtype<int>().raw_dtype()),
                1, env_shape, nb::handle(nullptr), nullptr, nb::dtype<int>()));

            // Cast to typed arrays
            auto rewards_array = nb::cast<nb::ndarray<nb::numpy, int>>(rewards);
            auto terminations_array = nb::cast<nb::ndarray<nb::numpy, bool>>(terminations);
            auto truncations_array = nb::cast<nb::ndarray<nb::numpy, bool>>(truncations);
            auto env_ids_array = nb::cast<nb::ndarray<nb::numpy, int>>(env_ids);
            auto lives_array = nb::cast<nb::ndarray<nb::numpy, int>>(lives);
            auto frame_numbers_array = nb::cast<nb::ndarray<nb::numpy, int>>(frame_numbers);
            auto episode_frame_numbers_array = nb::cast<nb::ndarray<nb::numpy, int>>(episode_frame_numbers);

            // Copy data from observations to NumPy arrays
            const size_t obs_size = stack_num * height * width * channels;
            auto observations_ptr = observations.data();

            for (int i = 0; i < num_envs; i++) {
                const auto& timestep = timesteps[i];

                // Copy screen data
                std::memcpy(
                    observations_ptr + i * obs_size,
                    timestep.observation.data(),
                    obs_size * sizeof(uint8_t)
                );

                // Copy other fields
                rewards_array.data()[i] = timestep.reward;
                terminations_array.data()[i] = timestep.terminated;
                truncations_array.data()[i] = timestep.truncated;
                env_ids_array.data()[i] = timestep.env_id;
                lives_array.data()[i] = timestep.lives;
                frame_numbers_array.data()[i] = timestep.frame_number;
                episode_frame_numbers_array.data()[i] = timestep.episode_frame_number;
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
            size_t shape[1] = {sizeof(ptr)};
            auto handle_array = nb::steal(nb::detail::ndarray_new(
                nb::handle(nb::dtype<uint8_t>().raw_dtype()),
                1, shape, nb::handle(nullptr), nullptr, nb::dtype<uint8_t>()));

            auto handle_typed = nb::cast<nb::ndarray<nb::numpy, uint8_t>>(handle_array);
            auto handle_ptr = handle_typed.data();

            // Copy the pointer value into the byte array
            std::memcpy(handle_ptr, &ptr, sizeof(ptr));

            return handle_array;
        });
}
