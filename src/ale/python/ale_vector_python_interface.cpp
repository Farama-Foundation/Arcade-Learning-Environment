#include "ale_vector_python_interface.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <tuple>

namespace py = pybind11;

// Function to add vector environment bindings to an existing module
void init_vector_module(py::module& m) {
    // Define ALEVectorInterface class
    py::class_<ale::vector::ALEVectorInterface>(m, "ALEVectorInterface")
        .def(py::init<const fs::path, int, int, int, int, int, bool, bool, int, bool, bool, bool, bool, int, float, bool, int, int, int>(),
             py::arg("rom_path"),
             py::arg("num_envs"),
             py::arg("frame_skip") = 4,
             py::arg("stack_num") = 4,
             py::arg("img_height") = 84,
             py::arg("img_width") = 84,
             py::arg("grayscale") = true,
             py::arg("maxpool") = true,
             py::arg("noop_max") = 30,
             py::arg("use_fire_reset") = true,
             py::arg("episodic_life") = false,
             py::arg("life_loss_info") = false,
             py::arg("reward_clipping") = true,
             py::arg("max_episode_steps") = 108000,
             py::arg("repeat_action_probability") = 0.0f,
             py::arg("full_action_space") = false,
             py::arg("batch_size") = 0,
             py::arg("num_threads") = 0,
             py::arg("thread_affinity_offset") = -1)
        .def("reset", [](ale::vector::ALEVectorInterface& self, const std::vector<int> reset_indices, const std::vector<int> reset_seeds) {
            // Call C++ reset method with GIL released
            py::gil_scoped_release release;
            auto timesteps = self.reset(reset_indices, reset_seeds);
            py::gil_scoped_acquire acquire;

            // Get shape information
            int num_envs = timesteps.size();
            auto obs_shape = self.get_observation_shape();
            int stack_num = std::get<0>(obs_shape);
            int height = std::get<1>(obs_shape);
            int width = std::get<2>(obs_shape);
            int channels = self.is_grayscale() ? 1 : 3;

            // Create a single NumPy array for all observations
            py::array_t<uint8_t> observations;
            if (self.is_grayscale()) {
                observations = py::array_t<uint8_t>({num_envs, stack_num, height, width});
            } else {
                observations = py::array_t<uint8_t>({num_envs, stack_num, height, width, 3});
            }
            auto observations_ptr = static_cast<uint8_t*>(observations.mutable_data());

            // Create arrays for info fields
            py::array_t<int> env_ids(num_envs);
            py::array_t<int> lives(num_envs);
            py::array_t<int> frame_numbers(num_envs);
            py::array_t<int> episode_frame_numbers(num_envs);

            auto env_ids_ptr = static_cast<int*>(env_ids.mutable_data());
            auto lives_ptr = static_cast<int*>(lives.mutable_data());
            auto frame_numbers_ptr = static_cast<int*>(frame_numbers.mutable_data());
            auto episode_frame_numbers_ptr = static_cast<int*>(episode_frame_numbers.mutable_data());

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
                env_ids_ptr[i] = timestep.env_id;
                lives_ptr[i] = timestep.lives;
                frame_numbers_ptr[i] = timestep.frame_number;
                episode_frame_numbers_ptr[i] = timestep.episode_frame_number;
            }

            // Create info dict
            py::dict info;
            info["env_id"] = env_ids;
            info["lives"] = lives;
            info["frame_number"] = frame_numbers;
            info["episode_frame_number"] = episode_frame_numbers;

            return py::make_tuple(observations, info);
        })
        .def("send", [](ale::vector::ALEVectorInterface& self, const std::vector<int> action_ids, const std::vector<float> paddle_strengths) {
            self.send(action_ids, paddle_strengths);
        })
        .def("recv", [](ale::vector::ALEVectorInterface& self) {
            const auto timesteps = self.recv();
            py::gil_scoped_acquire acquire;

            // Get shape information
            int num_envs = timesteps.size();
            const auto shape_info = self.get_observation_shape();
            int stack_num = std::get<0>(shape_info);
            int height = std::get<1>(shape_info);
            int width = std::get<2>(shape_info);
            int channels = self.is_grayscale() ? 1 : 3;

            // Create NumPy arrays
            py::array_t<uint8_t> observations;
            if (self.is_grayscale()) {
                observations = py::array_t<uint8_t>({num_envs, stack_num, height, width});
            } else {
                observations = py::array_t<uint8_t>({num_envs, stack_num, height, width, 3});
            }
            py::array_t<int> rewards(num_envs);
            py::array_t<bool> terminations(num_envs);
            py::array_t<bool> truncations(num_envs);
            py::array_t<int> env_ids(num_envs);
            py::array_t<int> lives(num_envs);
            py::array_t<int> frame_numbers(num_envs);
            py::array_t<int> episode_frame_numbers(num_envs);

            // Get pointers to the arrays' data
            auto observations_ptr = static_cast<uint8_t*>(observations.mutable_data());
            auto rewards_ptr = static_cast<int*>(rewards.mutable_data());
            auto terminations_ptr = static_cast<bool*>(terminations.mutable_data());
            auto truncations_ptr = static_cast<bool*>(truncations.mutable_data());
            auto env_ids_ptr = static_cast<int*>(env_ids.mutable_data());
            auto lives_ptr = static_cast<int*>(lives.mutable_data());
            auto frame_numbers_ptr = static_cast<int*>(frame_numbers.mutable_data());
            auto episode_frame_numbers_ptr = static_cast<int*>(episode_frame_numbers.mutable_data());

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
            py::dict info;
            info["env_id"] = env_ids;
            info["lives"] = lives;
            info["frame_number"] = frame_numbers;
            info["episode_frame_number"] = episode_frame_numbers;

            return py::make_tuple(observations, rewards, terminations, truncations, info);
        })
        .def("get_action_set", &ale::vector::ALEVectorInterface::get_action_set)
        .def("get_num_envs", &ale::vector::ALEVectorInterface::get_num_envs)
        .def("get_observation_shape", [](ale::vector::ALEVectorInterface& self) {
            auto shape = self.get_observation_shape();
            if (self.is_grayscale()) {
                return py::make_tuple(std::get<0>(shape), std::get<1>(shape), std::get<2>(shape));
            } else {
                return py::make_tuple(std::get<0>(shape), std::get<1>(shape), std::get<2>(shape), std::get<3>(shape));
            }
        })
        .def("handle", [](ale::vector::ALEVectorInterface& self) {
            // Get the raw pointer to the AsyncVectorizer
            auto ptr = self.get_vectorizer();

            // Create a NumPy array with the correct size to hold the pointer
            py::array_t<uint8_t> handle_array(sizeof(ptr));
            auto handle_ptr = static_cast<uint8_t*>(handle_array.mutable_data());

            // Copy the pointer value into the byte array
            std::memcpy(handle_ptr, &ptr, sizeof(ptr));

            return handle_array;
        });
}
