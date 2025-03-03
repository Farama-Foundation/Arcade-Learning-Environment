#include "ale_vector_python_interface.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


// Python module definition
PYBIND11_MODULE(_vector, m) {
    // Module docstring
    m.doc() = "Vector environment support for ALE";

    // Define ActionSlice class
    py::class_<ActionSlice>(m, "ActionSlice")
        .def(py::init<>())
        .def_readwrite("env_id", &ActionSlice::env_id)
        .def_readwrite("order", &ActionSlice::order)
        .def_readwrite("force_reset", &ActionSlice::force_reset);

    // Define Action class
    py::class_<Action>(m, "Action")
        .def(py::init<>())
        .def(py::init([](const py::dict& d) {
            Action a;
            a.env_id = d["env_id"].cast<int>();
            a.action_id = d["action_id"].cast<int>();
            if (d.contains("paddle_strength")) {
                a.paddle_strength = d["paddle_strength"].cast<float>();
            } else {
                a.paddle_strength = 1.0f;
            }
            return a;
        }))
        .def_readwrite("env_id", &Action::env_id)
        .def_readwrite("action_id", &Action::action_id)
        .def_readwrite("paddle_strength", &Action::paddle_strength);

    // Define Observation class
    py::class_<Observation>(m, "Observation")
        .def(py::init<>())
        .def_property_readonly("env_id", [](const Observation& o) { return o.env_id; })
        .def_property_readonly("screen", [](const Observation& o) { return o.screen; })
        .def_property_readonly("reward", [](const Observation& o) { return o.reward; })
        .def_property_readonly("done", [](const Observation& o) { return o.done; })
        .def_property_readonly("truncated", [](const Observation& o) { return o.truncated; })
        .def_property_readonly("lives", [](const Observation& o) { return o.lives; })
        .def_property_readonly("frame_number", [](const Observation& o) { return o.frame_number; })
        .def_property_readonly("episode_frame_number", [](const Observation& o) { return o.episode_frame_number; });

    // Define ALEVectorInterface class
    py::class_<ALEVectorInterface>(m, "ALEVectorInterface")
        .def(py::init<int, const std::string&, int, bool, int, int, int, int, bool, bool, int, float, bool, int, int, int, int>(),
             py::arg("num_envs"),
             py::arg("rom_path"),
             py::arg("frame_skip") = 4,
             py::arg("gray_scale") = true,
             py::arg("stack_num") = 4,
             py::arg("img_height") = 84,
             py::arg("img_width") = 84,
             py::arg("noop_max") = 30,
             py::arg("fire_reset") = true,
             py::arg("episodic_life") = false,
             py::arg("max_episode_steps") = 108000,
             py::arg("repeat_action_probability") = 0.0f,
             py::arg("full_action_space") = false,
             py::arg("batch_size") = 0,
             py::arg("num_threads") = 0,
             py::arg("seed") = 0,
             py::arg("thread_affinity_offset") = -1)
        .def("reset_all", &ALEVectorInterface::reset_all)
        .def("reset", &ALEVectorInterface::reset)
        .def("step", &ALEVectorInterface::step)
        .def("get_action_set", &ALEVectorInterface::get_action_set)
        .def("get_num_envs", &ALEVectorInterface::get_num_envs)
        .def("get_observation_dims", &ALEVectorInterface::get_observation_dims);
}
