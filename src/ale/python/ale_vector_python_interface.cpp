#include "ale_vector_python_interface.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/ndarray.h>
#include <vector>
#include <cmath>
#include <tuple>
#include <string>

namespace nb = nanobind;

// Function to add vector environment bindings to an existing module
void init_vector_module(nb::module_& m) {
    // Define ALEVectorInterface class
    nb::class_<ale::vector::ALEVectorInterface>(m, "ALEVectorInterface")
        .def(nb::init<const fs::path, int, int, int, int, int, bool, bool, int, bool, bool, bool, bool, int, float, bool, int, int, int, std::string>(),
             nb::arg("rom_path"),
             nb::arg("num_envs"),
             nb::arg("frame_skip") = 4,
             nb::arg("stack_num") = 4,
             nb::arg("img_height") = 84,
             nb::arg("img_width") = 84,
             nb::arg("grayscale") = true,
             nb::arg("maxpool") = true,
             nb::arg("noop_max") = 30,
             nb::arg("use_fire_reset") = true,
             nb::arg("episodic_life") = false,
             nb::arg("life_loss_info") = false,
             nb::arg("reward_clipping") = true,
             nb::arg("max_episode_steps") = 108000,
             nb::arg("repeat_action_probability") = 0.0f,
             nb::arg("full_action_space") = false,
             nb::arg("batch_size") = 0,
             nb::arg("num_threads") = 0,
             nb::arg("thread_affinity_offset") = -1,
             nb::arg("autoreset_mode") = "NextStep")
        .def("reset", [](ale::vector::ALEVectorInterface& self, const std::vector<int> reset_indices, const std::vector<int> reset_seeds) {
            // Call C++ reset method with GIL released
            nb::gil_scoped_release release;
            auto result = self.reset(reset_indices, reset_seeds);
            nb::gil_scoped_acquire acquire;

            // Get shape information
            const int batch_size = result.batch_size;
            const auto obs_shape = self.get_observation_shape();
            const int stack_num = std::get<0>(obs_shape);
            const int height = std::get<1>(obs_shape);
            const int width = std::get<2>(obs_shape);
            const bool grayscale = self.is_grayscale();

            // Wrap observation buffer - capsule takes ownership
            nb::capsule obs_owner(result.observations, [](void *p) noexcept {
                delete[] static_cast<uint8_t*>(p);
            });

            nb::ndarray<nb::numpy, uint8_t> observations;
            if (grayscale) {
                size_t shape[4] = {(size_t)batch_size, (size_t)stack_num, (size_t)height, (size_t)width};
                observations = nb::ndarray<nb::numpy, uint8_t>(result.observations, 4, shape, obs_owner);
            } else {
                size_t shape[5] = {(size_t)batch_size, (size_t)stack_num, (size_t)height, (size_t)width, 3};
                observations = nb::ndarray<nb::numpy, uint8_t>(result.observations, 5, shape, obs_owner);
            }

            // Create capsules - ownership transferred from BatchData
            nb::capsule env_ids_owner(result.env_ids, [](void *p) noexcept { delete[] (int*)p; });
            nb::capsule lives_owner(result.lives, [](void *p) noexcept { delete[] (int*)p; });
            nb::capsule frame_numbers_owner(result.frame_numbers, [](void *p) noexcept { delete[] (int*)p; });
            nb::capsule episode_frame_numbers_owner(result.episode_frame_numbers, [](void *p) noexcept { delete[] (int*)p; });

            // Create numpy arrays (zero-copy - direct from BatchData)
            size_t info_shape[1] = {(size_t)batch_size};
            auto env_ids = nb::ndarray<nb::numpy, int>(result.env_ids, 1, info_shape, env_ids_owner);
            auto lives = nb::ndarray<nb::numpy, int>(result.lives, 1, info_shape, lives_owner);
            auto frame_numbers = nb::ndarray<nb::numpy, int>(result.frame_numbers, 1, info_shape, frame_numbers_owner);
            auto episode_frame_numbers = nb::ndarray<nb::numpy, int>(result.episode_frame_numbers, 1, info_shape, episode_frame_numbers_owner);

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
            // Release GIL while waiting for workers
            nb::gil_scoped_release release;
            auto result = self.recv();
            nb::gil_scoped_acquire acquire;

            // Get shape info
            const auto obs_shape_info = self.get_observation_shape();
            const int stack_num = std::get<0>(obs_shape_info);
            const int height = std::get<1>(obs_shape_info);
            const int width = std::get<2>(obs_shape_info);
            const int batch_size = result.batch_size;
            const bool grayscale = self.is_grayscale();

            // Wrap obs buffer - capsule takes ownership and will delete[]
            nb::capsule obs_owner(result.observations, [](void *p) noexcept { delete[] static_cast<uint8_t*>(p); });
            nb::ndarray<nb::numpy, uint8_t> observations;
            if (grayscale) {
                size_t shape[4] = {(size_t)batch_size, (size_t)stack_num, (size_t)height, (size_t)width};
                observations = nb::ndarray<nb::numpy, uint8_t>(result.observations, 4, shape, obs_owner);
            } else {
                size_t shape[5] = {(size_t)batch_size, (size_t)stack_num, (size_t)height, (size_t)width, 3};
                observations = nb::ndarray<nb::numpy, uint8_t>(result.observations, 5, shape, obs_owner);
            }

            // Create capsules - ownership transferred from BatchData
            nb::capsule rewards_owner(result.rewards, [](void *p) noexcept { delete[] (int*)p; });
            nb::capsule terminations_owner(result.terminations, [](void *p) noexcept { delete[] (bool*)p; });
            nb::capsule truncations_owner(result.truncations, [](void *p) noexcept { delete[] (bool*)p; });
            nb::capsule env_ids_owner(result.env_ids, [](void *p) noexcept { delete[] (int*)p; });
            nb::capsule lives_owner(result.lives, [](void *p) noexcept { delete[] (int*)p; });
            nb::capsule frame_numbers_owner(result.frame_numbers, [](void *p) noexcept { delete[] (int*)p; });
            nb::capsule episode_frame_numbers_owner(result.episode_frame_numbers, [](void *p) noexcept { delete[] (int*)p; });

            // Create numpy arrays (zero-copy - direct from BatchData)
            size_t info_shape[1] = {(size_t)batch_size};
            auto rewards = nb::ndarray<nb::numpy, int>(result.rewards, 1, info_shape, rewards_owner);
            auto terminations = nb::ndarray<nb::numpy, bool>(result.terminations, 1, info_shape, terminations_owner);
            auto truncations = nb::ndarray<nb::numpy, bool>(result.truncations, 1, info_shape, truncations_owner);
            auto env_ids = nb::ndarray<nb::numpy, int>(result.env_ids, 1, info_shape, env_ids_owner);
            auto lives = nb::ndarray<nb::numpy, int>(result.lives, 1, info_shape, lives_owner);
            auto frame_numbers = nb::ndarray<nb::numpy, int>(result.frame_numbers, 1, info_shape, frame_numbers_owner);
            auto episode_frame_numbers = nb::ndarray<nb::numpy, int>(result.episode_frame_numbers, 1, info_shape, episode_frame_numbers_owner);

            // Build info dict
            nb::dict info;
            info["env_id"] = env_ids;
            info["lives"] = lives;
            info["frame_number"] = frame_numbers;
            info["episode_frame_number"] = episode_frame_numbers;

            // Handle final_obs for SameStep mode - only include if any env terminated/truncated
            if (result.final_observations != nullptr) {
                // Check if any environment actually terminated or truncated
                bool any_done = false;
                for (size_t i = 0; i < batch_size; i++) {
                    if (result.terminations[i] || result.truncations[i]) {
                        any_done = true;
                        break;
                    }
                }

                if (any_done) {
                    // Wrap the buffer directly - workers have already filled in all slots
                    nb::capsule final_obs_owner(result.final_observations, [](void *p) noexcept {
                        delete[] static_cast<uint8_t*>(p);
                    });

                    nb::ndarray<nb::numpy, uint8_t> final_observations;
                    if (grayscale) {
                        size_t shape[4] = {(size_t)batch_size, (size_t)stack_num, (size_t)height, (size_t)width};
                        final_observations = nb::ndarray<nb::numpy, uint8_t>(result.final_observations, 4, shape, final_obs_owner);
                    } else {
                        size_t shape[5] = {(size_t)batch_size, (size_t)stack_num, (size_t)height, (size_t)width, 3};
                        final_observations = nb::ndarray<nb::numpy, uint8_t>(result.final_observations, 5, shape, final_obs_owner);
                    }
                    info["final_obs"] = final_observations;
                } else {
                    // No environments terminated - delete the unused buffer
                    delete[] result.final_observations;
                }
            }

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

            // Allocate memory for handle array
            uint8_t* handle_data = new uint8_t[sizeof(ptr)];
            std::memcpy(handle_data, &ptr, sizeof(ptr));

            // Create capsule for cleanup
            nb::capsule handle_owner(handle_data, [](void *p) noexcept { delete[] (uint8_t *) p; });

            // Create numpy array
            size_t shape[1] = {sizeof(ptr)};
            return nb::ndarray<nb::numpy, uint8_t>(handle_data, 1, shape, handle_owner);
        });
}
