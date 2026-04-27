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
        .def(nb::init<const std::vector<fs::path>&, int, int, int, int, bool, bool, int, bool, bool, bool, bool, int, float, bool, int, int, int, std::string>(),
             nb::arg("rom_paths"),
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
            auto timesteps = self.reset(reset_indices, reset_seeds);
            nb::gil_scoped_acquire acquire;

            // Get shape information
            int batch_size = timesteps.size();
            auto obs_shape = self.get_observation_shape();
            int stack_num = std::get<0>(obs_shape);
            int height = std::get<1>(obs_shape);
            int width = std::get<2>(obs_shape);
            int channels = self.is_grayscale() ? 1 : 3;

            // Create a single NumPy array for all observations
            size_t obs_total_size = batch_size * stack_num * height * width * channels;
            uint8_t* obs_data = new uint8_t[obs_total_size];

            // Create arrays for info fields
            int* env_ids_data = new int[batch_size];
            int* lives_data = new int[batch_size];
            int* frame_numbers_data = new int[batch_size];
            int* episode_frame_numbers_data = new int[batch_size];

            // Copy data from observations to arrays
            size_t obs_size = stack_num * height * width * channels;
            for (int i = 0; i < batch_size; i++) {
                const auto& timestep = timesteps[i];

                // Copy screen data
                std::memcpy(
                    obs_data + i * obs_size,
                    timestep.observation.data(),
                    obs_size * sizeof(uint8_t)
                );

                // Copy info fields
                env_ids_data[i] = timestep.env_id;
                lives_data[i] = timestep.lives;
                frame_numbers_data[i] = timestep.frame_number;
                episode_frame_numbers_data[i] = timestep.episode_frame_number;
            }

            // Create capsules for cleanup
            nb::capsule obs_owner(obs_data, [](void *p) noexcept { delete[] (uint8_t *) p; });
            nb::capsule env_ids_owner(env_ids_data, [](void *p) noexcept { delete[] (int *) p; });
            nb::capsule lives_owner(lives_data, [](void *p) noexcept { delete[] (int *) p; });
            nb::capsule frame_numbers_owner(frame_numbers_data, [](void *p) noexcept { delete[] (int *) p; });
            nb::capsule episode_frame_numbers_owner(episode_frame_numbers_data, [](void *p) noexcept { delete[] (int *) p; });

            // Create numpy arrays with allocated data
            nb::ndarray<nb::numpy, uint8_t> observations;
            if (self.is_grayscale()) {
                size_t shape[4] = {(size_t)batch_size, (size_t)stack_num, (size_t)height, (size_t)width};
                observations = nb::ndarray<nb::numpy, uint8_t>(obs_data, 4, shape, obs_owner);
            } else {
                size_t shape[5] = {(size_t)batch_size, (size_t)stack_num, (size_t)height, (size_t)width, 3};
                observations = nb::ndarray<nb::numpy, uint8_t>(obs_data, 5, shape, obs_owner);
            }

            size_t info_shape[1] = {(size_t)batch_size};
            auto env_ids = nb::ndarray<nb::numpy, int>(env_ids_data, 1, info_shape, env_ids_owner);
            auto lives = nb::ndarray<nb::numpy, int>(lives_data, 1, info_shape, lives_owner);
            auto frame_numbers = nb::ndarray<nb::numpy, int>(frame_numbers_data, 1, info_shape, frame_numbers_owner);
            auto episode_frame_numbers = nb::ndarray<nb::numpy, int>(episode_frame_numbers_data, 1, info_shape, episode_frame_numbers_owner);

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
        }, nb::arg("actions_per_rom"), nb::arg("paddle_strengths"))
        .def("send", [](ale::vector::ALEVectorInterface& self,
                const std::vector<std::vector<int>>& action_id_sequences,
                nb::object paddle_arg,
                nb::object gamma_arg) {
            const size_t n = action_id_sequences.size();

            std::vector<std::vector<float>> paddle_strength_per_action(n);
            if (nb::isinstance<nb::float_>(paddle_arg) || nb::isinstance<nb::int_>(paddle_arg)) {
                float p = nb::cast<float>(paddle_arg);
                for (size_t i = 0; i < n; i++)
                    paddle_strength_per_action[i].assign(action_id_sequences[i].size(), p);
            } else {
                auto outer = nb::cast<nb::sequence>(paddle_arg);
                if (n > 0 && (nb::isinstance<nb::float_>(outer[0]) || nb::isinstance<nb::int_>(outer[0]))) {
                    auto per_env = nb::cast<std::vector<float>>(paddle_arg);
                    for (size_t i = 0; i < n; i++)
                        paddle_strength_per_action[i].assign(action_id_sequences[i].size(), per_env[i]);
                } else if (n > 0) {
                    paddle_strength_per_action = nb::cast<std::vector<std::vector<float>>>(paddle_arg);
                }
            }

            std::vector<float> gammas;
            if (nb::isinstance<nb::float_>(gamma_arg) || nb::isinstance<nb::int_>(gamma_arg)) {
                float g = nb::cast<float>(gamma_arg);
                if (g != 1.0f) {
                    for (size_t i = 0; i < n; i++) {
                        if (action_id_sequences[i].empty()) {
                            throw std::invalid_argument(
                                "scalar gamma != 1.0 cannot be used when env " + std::to_string(i) +
                                " has an empty sequence; use gamma=1.0 or pass a list of per-env gammas");
                        }
                    }
                }
                gammas.assign(n, g);
            } else {
                gammas = nb::cast<std::vector<float>>(gamma_arg);
            }
            self.send(action_id_sequences, paddle_strength_per_action, gammas);
        }, nb::arg("actions_per_rom"), nb::arg("paddle_strength_per_action"), nb::arg("gamma"))
        .def("recv", [](ale::vector::ALEVectorInterface& self,
                nb::object obs_arg, nb::object rewards_arg, nb::object terms_arg,
                nb::object truncs_arg, nb::object steps_arg) -> nb::object {
            const bool into = !obs_arg.is_none();

            nb::gil_scoped_release release;
            const auto timesteps = self.recv();
            nb::gil_scoped_acquire acquire;

            const int batch_size = timesteps.size();
            const auto shape_info = self.get_observation_shape();
            const int stack_num = std::get<0>(shape_info);
            const int height = std::get<1>(shape_info);
            const int width = std::get<2>(shape_info);
            const int channels = self.is_grayscale() ? 1 : 3;
            const size_t obs_size = stack_num * height * width * channels;
            const ale::vector::AutoresetMode autoreset_mode = self.get_autoreset_mode();

            // Always-allocated small info fields
            int* env_ids = new int[batch_size];
            int* lives   = new int[batch_size];
            int* frame_nums    = new int[batch_size];
            int* ep_frame_nums = new int[batch_size];

            // Main output pointers - either caller buffers or newly allocated
            uint8_t* obs_dst;
            double*  rew_dst;
            bool*    ter_dst;
            bool*    tru_dst;
            int*     stp_dst;

            nb::capsule obs_cap, rew_cap, ter_cap, tru_cap, stp_cap;

            if (into) {
                obs_dst = nb::cast<nb::ndarray<nb::numpy, uint8_t>>(obs_arg).data();
                rew_dst = nb::cast<nb::ndarray<nb::numpy, double>>(rewards_arg).data();
                ter_dst = nb::cast<nb::ndarray<nb::numpy, bool>>(terms_arg).data();
                tru_dst = nb::cast<nb::ndarray<nb::numpy, bool>>(truncs_arg).data();
                stp_dst = nb::cast<nb::ndarray<nb::numpy, int>>(steps_arg).data();
            } else {
                uint8_t* o = new uint8_t[batch_size * obs_size];
                double*  r = new double[batch_size];
                bool*    t = new bool[batch_size];
                bool*    u = new bool[batch_size];
                int*     s = new int[batch_size];
                obs_cap = nb::capsule(o, [](void *p) noexcept { delete[] (uint8_t *) p; });
                rew_cap = nb::capsule(r, [](void *p) noexcept { delete[] (double *) p; });
                ter_cap = nb::capsule(t, [](void *p) noexcept { delete[] (bool *) p; });
                tru_cap = nb::capsule(u, [](void *p) noexcept { delete[] (bool *) p; });
                stp_cap = nb::capsule(s, [](void *p) noexcept { delete[] (int *) p; });
                obs_dst = o; rew_dst = r; ter_dst = t; tru_dst = u; stp_dst = s;
            }

            for (int i = 0; i < batch_size; i++) {
                const auto& ts = timesteps[i];
                std::memcpy(obs_dst + i * obs_size, ts.observation.data(), obs_size);
                rew_dst[i] = ts.reward;
                ter_dst[i] = ts.terminated;
                tru_dst[i] = ts.truncated;
                stp_dst[i] = ts.steps_taken;
                env_ids[i] = ts.env_id;
                lives[i]   = ts.lives;
                frame_nums[i]    = ts.frame_number;
                ep_frame_nums[i] = ts.episode_frame_number;
            }

            nb::capsule env_ids_cap(env_ids, [](void *p) noexcept { delete[] (int *) p; });
            nb::capsule lives_cap(lives,     [](void *p) noexcept { delete[] (int *) p; });
            nb::capsule frame_cap(frame_nums, [](void *p) noexcept { delete[] (int *) p; });
            nb::capsule ep_cap(ep_frame_nums, [](void *p) noexcept { delete[] (int *) p; });

            size_t shape1[1] = {(size_t)batch_size};
            nb::dict info;
            info["env_id"]               = nb::ndarray<nb::numpy, int>(env_ids,      1, shape1, env_ids_cap);
            info["lives"]                = nb::ndarray<nb::numpy, int>(lives,         1, shape1, lives_cap);
            info["frame_number"]         = nb::ndarray<nb::numpy, int>(frame_nums,    1, shape1, frame_cap);
            info["episode_frame_number"] = nb::ndarray<nb::numpy, int>(ep_frame_nums, 1, shape1, ep_cap);
            if (into) {
                info["steps_taken"] = steps_arg;
            } else {
                info["steps_taken"] = nb::ndarray<nb::numpy, int>(stp_dst, 1, shape1, stp_cap);
            }

            if (autoreset_mode == ale::vector::AutoresetMode::SameStep) {
                bool any_done = false;
                for (int i = 0; i < batch_size; i++) {
                    if (ter_dst[i] || tru_dst[i]) { any_done = true; break; }
                }
                if (any_done) {
                    uint8_t* final = new uint8_t[batch_size * obs_size];
                    for (int i = 0; i < batch_size; i++) {
                        const auto& ts = timesteps[i];
                        const std::vector<uint8_t>* src = (ts.terminated || ts.truncated) ?
                            ts.final_observation : &ts.observation;
                        std::memcpy(final + i * obs_size, src->data(), obs_size);
                    }
                    nb::capsule final_cap(final, [](void *p) noexcept { delete[] (uint8_t *) p; });
                    nb::ndarray<nb::numpy, uint8_t> final_obs;
                    if (self.is_grayscale()) {
                        size_t shape[4] = {(size_t)batch_size, (size_t)stack_num, (size_t)height, (size_t)width};
                        final_obs = nb::ndarray<nb::numpy, uint8_t>(final, 4, shape, final_cap);
                    } else {
                        size_t shape[5] = {(size_t)batch_size, (size_t)stack_num, (size_t)height, (size_t)width, 3};
                        final_obs = nb::ndarray<nb::numpy, uint8_t>(final, 5, shape, final_cap);
                    }
                    info["final_obs"] = final_obs;
                }
            }

            if (into) {
                return nb::object(info);
            }

            nb::ndarray<nb::numpy, uint8_t> obs_arr;
            if (self.is_grayscale()) {
                size_t shape[4] = {(size_t)batch_size, (size_t)stack_num, (size_t)height, (size_t)width};
                obs_arr = nb::ndarray<nb::numpy, uint8_t>(obs_dst, 4, shape, obs_cap);
            } else {
                size_t shape[5] = {(size_t)batch_size, (size_t)stack_num, (size_t)height, (size_t)width, 3};
                obs_arr = nb::ndarray<nb::numpy, uint8_t>(obs_dst, 5, shape, obs_cap);
            }
            return nb::object(nb::make_tuple(
                obs_arr,
                nb::ndarray<nb::numpy, double>(rew_dst, 1, shape1, rew_cap),
                nb::ndarray<nb::numpy, bool>(ter_dst,   1, shape1, ter_cap),
                nb::ndarray<nb::numpy, bool>(tru_dst,   1, shape1, tru_cap),
                info
            ));
        },
        nb::arg("obs") = nb::none(), nb::arg("rewards") = nb::none(),
        nb::arg("terminations") = nb::none(), nb::arg("truncations") = nb::none(),
        nb::arg("steps_taken") = nb::none())
        .def("num_actions", &ale::vector::ALEVectorInterface::num_actions)
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
