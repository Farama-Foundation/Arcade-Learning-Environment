#include "ale_vector_python_interface.hpp"
#include "ale/vector/env_vectorizer.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
namespace fs = std::filesystem;

using ale::vector::EnvVectorizer;
using ale::vector::BatchResult;
using ale::vector::AutoresetMode;
using ale::vector::Action;

namespace {

/// Helper to create numpy array from raw pointer with capsule ownership
template<typename T>
nb::ndarray<nb::numpy, T> make_numpy_array(T* data, std::vector<std::size_t> shape) {
    nb::capsule owner(data, [](void* p) noexcept {
        delete[] static_cast<T*>(p);
    });
    return nb::ndarray<nb::numpy, T>(data, shape.size(), shape.data(), owner);
}

/// Convert BatchResult to Python tuple for reset: (observations, info)
nb::tuple wrap_reset_result(EnvVectorizer& vec, BatchResult&& result) {
    const std::size_t batch_size = result.batch_size();
    auto [stack_num, height, width, channels] = vec.observation_shape();

    // Build observation shape
    std::vector<std::size_t> obs_shape;
    if (vec.is_grayscale()) {
        obs_shape = {batch_size, static_cast<std::size_t>(stack_num),
                     static_cast<std::size_t>(height), static_cast<std::size_t>(width)};
    } else {
        obs_shape = {batch_size, static_cast<std::size_t>(stack_num),
                     static_cast<std::size_t>(height), static_cast<std::size_t>(width), 3};
    }

    std::vector<std::size_t> info_shape = {batch_size};

    // Create numpy arrays (transfers ownership via release)
    auto observations = make_numpy_array(result.release_observations(), obs_shape);
    auto env_ids = make_numpy_array(result.release_env_ids(), info_shape);
    auto lives = make_numpy_array(result.release_lives(), info_shape);
    auto frame_numbers = make_numpy_array(result.release_frame_numbers(), info_shape);
    auto episode_frame_numbers = make_numpy_array(result.release_episode_frame_numbers(), info_shape);

    // Clean up unreleased arrays (rewards, terminations, truncations not used in reset)
    // BatchResult destructor handles this

    // Build info dict
    nb::dict info;
    info["env_id"] = env_ids;
    info["lives"] = lives;
    info["frame_number"] = frame_numbers;
    info["episode_frame_number"] = episode_frame_numbers;

    return nb::make_tuple(observations, info);
}

/// Convert BatchResult to Python tuple for step: (observations, rewards, terminations, truncations, info)
nb::tuple wrap_step_result(EnvVectorizer& vec, BatchResult&& result) {
    const std::size_t batch_size = result.batch_size();
    auto [stack_num, height, width, channels] = vec.observation_shape();

    // Build observation shape
    std::vector<std::size_t> obs_shape;
    if (vec.is_grayscale()) {
        obs_shape = {batch_size, static_cast<std::size_t>(stack_num),
                     static_cast<std::size_t>(height), static_cast<std::size_t>(width)};
    } else {
        obs_shape = {batch_size, static_cast<std::size_t>(stack_num),
                     static_cast<std::size_t>(height), static_cast<std::size_t>(width), 3};
    }

    std::vector<std::size_t> info_shape = {batch_size};

    // Create numpy arrays
    auto observations = make_numpy_array(result.release_observations(), obs_shape);
    auto rewards = make_numpy_array(result.release_rewards(), info_shape);
    auto terminations = make_numpy_array(result.release_terminations(), info_shape);
    auto truncations = make_numpy_array(result.release_truncations(), info_shape);
    auto env_ids = make_numpy_array(result.release_env_ids(), info_shape);
    auto lives = make_numpy_array(result.release_lives(), info_shape);
    auto frame_numbers = make_numpy_array(result.release_frame_numbers(), info_shape);
    auto episode_frame_numbers = make_numpy_array(result.release_episode_frame_numbers(), info_shape);

    // Build info dict
    nb::dict info;
    info["env_id"] = env_ids;
    info["lives"] = lives;
    info["frame_number"] = frame_numbers;
    info["episode_frame_number"] = episode_frame_numbers;

    // Handle final_obs for SameStep mode
    if (result.has_final_obs()) {
        // Check if any environment terminated or truncated
        bool any_done = false;
        bool* term_data = terminations.data();
        bool* trunc_data = truncations.data();
        for (std::size_t i = 0; i < batch_size; ++i) {
            if (term_data[i] || trunc_data[i]) {
                any_done = true;
                break;
            }
        }

        if (any_done) {
            auto final_obs = make_numpy_array(result.release_final_observations(), obs_shape);
            info["final_obs"] = final_obs;
        }
        // If no envs done, final_obs buffer will be cleaned up by BatchResult destructor
    }

    return nb::make_tuple(observations, rewards, terminations, truncations, info);
}

}  // anonymous namespace

void init_vector_module(nb::module_& m) {
    nb::class_<EnvVectorizer>(m, "ALEVectorInterface")
        .def("__init__", [](EnvVectorizer* t,
                const fs::path& rom_path,
                int num_envs,
                int frame_skip,
                int stack_num,
                int img_height,
                int img_width,
                bool grayscale,
                bool maxpool,
                int noop_max,
                bool use_fire_reset,
                bool episodic_life,
                bool life_loss_info,
                bool reward_clipping,
                int max_episode_steps,
                float repeat_action_probability,
                bool full_action_space,
                int batch_size,
                int num_threads,
                int thread_affinity_offset,
                const std::string& autoreset_mode_str
            ) {
                AutoresetMode autoreset_mode;
                if (autoreset_mode_str == "NextStep") {
                    autoreset_mode = AutoresetMode::NextStep;
                } else if (autoreset_mode_str == "SameStep") {
                    autoreset_mode = AutoresetMode::SameStep;
                } else {
                    throw std::invalid_argument("Invalid autoreset_mode: " + autoreset_mode_str);
                }

                new (t) EnvVectorizer(
                    rom_path, num_envs, batch_size, num_threads, thread_affinity_offset,
                    autoreset_mode, img_height, img_width, stack_num, grayscale,
                    frame_skip, maxpool, noop_max, use_fire_reset, episodic_life,
                    life_loss_info, reward_clipping, max_episode_steps,
                    repeat_action_probability, full_action_space
                );
            },
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

        .def("reset", [](EnvVectorizer& self,
                         const std::vector<int>& reset_indices,
                         const std::vector<int>& reset_seeds) {
            nb::gil_scoped_release release;
            auto result = self.reset(reset_indices, reset_seeds);
            nb::gil_scoped_acquire acquire;
            return wrap_reset_result(self, std::move(result));
        })

        .def("send", [](EnvVectorizer& self,
                        const std::vector<int>& action_ids,
                        const std::vector<float>& paddle_strengths) {
            if (action_ids.size() != paddle_strengths.size()) {
                throw std::invalid_argument("action_ids and paddle_strengths must have same size");
            }

            std::vector<Action> actions;
            actions.reserve(action_ids.size());
            for (std::size_t i = 0; i < action_ids.size(); ++i) {
                Action a;
                a.env_id = static_cast<int>(i);  // Will be remapped in send()
                a.action_id = action_ids[i];
                a.paddle_strength = paddle_strengths[i];
                a.force_reset = false;
                actions.push_back(a);
            }

            self.send(actions);
        })

        .def("recv", [](EnvVectorizer& self) {
            nb::gil_scoped_release release;
            auto result = self.recv();
            nb::gil_scoped_acquire acquire;
            return wrap_step_result(self, std::move(result));
        })

        .def("get_action_set", &EnvVectorizer::action_set)

        .def("get_num_envs", &EnvVectorizer::num_envs)

        .def("get_observation_shape", [](EnvVectorizer& self) {
            auto [stack, h, w, c] = self.observation_shape();
            if (self.is_grayscale()) {
                return nb::make_tuple(stack, h, w);
            } else {
                return nb::make_tuple(stack, h, w, c);
            }
        })

        .def("handle", [](EnvVectorizer& self) {
            const void* ptr = self.handle();
            std::size_t ptr_size = sizeof(ptr);

            uint8_t* handle_data = new uint8_t[ptr_size];
            std::memcpy(handle_data, &ptr, ptr_size);

            nb::capsule owner(handle_data, [](void* p) noexcept {
                delete[] static_cast<uint8_t*>(p);
            });

            std::vector<std::size_t> shape = {ptr_size};
            return nb::ndarray<nb::numpy, uint8_t>(handle_data, shape.size(), shape.data(), owner);
        });

    // Expose AutoresetMode enum
    nb::enum_<AutoresetMode>(m, "AutoresetMode")
        .value("NextStep", AutoresetMode::NextStep)
        .value("SameStep", AutoresetMode::SameStep);
}
