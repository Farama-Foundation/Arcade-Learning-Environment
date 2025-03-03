#include "ale_vector_python_interface.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Function to add vector environment bindings to an existing module
void init_vector_module(py::module& m) {
    // Create a submodule for vector environments
    py::module vector_module = m.def_submodule("vector", "Vector environment support for ALE");

    // Define ActionSlice class
    py::class_<ale::vector::ActionSlice>(vector_module, "ActionSlice")
        .def(py::init<>())
        .def_readwrite("env_id", &ale::vector::ActionSlice::env_id)
        .def_readwrite("order", &ale::vector::ActionSlice::order)
        .def_readwrite("force_reset", &ale::vector::ActionSlice::force_reset);

    // Define Action class
    py::class_<ale::vector::Action>(vector_module, "Action")
        .def(py::init<>())
        .def(py::init([](const py::dict& d) {
            ale::vector::Action a;
            a.env_id = d["env_id"].cast<int>();
            a.action_id = d["action_id"].cast<int>();
            if (d.contains("paddle_strength")) {
                a.paddle_strength = d["paddle_strength"].cast<float>();
            } else {
                a.paddle_strength = 1.0f;
            }
            return a;
        }))
        .def_readwrite("env_id", &ale::vector::Action::env_id)
        .def_readwrite("action_id", &ale::vector::Action::action_id)
        .def_readwrite("paddle_strength", &ale::vector::Action::paddle_strength);

    // Define Observation class
    py::class_<ale::vector::Observation>(vector_module, "Observation")
        .def(py::init<>())
        .def_property_readonly("env_id", [](const ale::vector::Observation& o) { return o.env_id; })
        .def_property_readonly("screen", [](const ale::vector::Observation& o) { return o.screen; })
        .def_property_readonly("reward", [](const ale::vector::Observation& o) { return o.reward; })
        .def_property_readonly("terminated", [](const ale::vector::Observation& o) { return o.terminated; })
        .def_property_readonly("truncated", [](const ale::vector::Observation& o) { return o.truncated; })
        .def_property_readonly("lives", [](const ale::vector::Observation& o) { return o.lives; })
        .def_property_readonly("frame_number", [](const ale::vector::Observation& o) { return o.frame_number; })
        .def_property_readonly("episode_frame_number", [](const ale::vector::Observation& o) { return o.episode_frame_number; });

    // Define ALEVectorInterface class
    py::class_<ale::vector::ALEVectorInterface>(vector_module, "ALEVectorInterface")
        .def(py::init<const std::string&, int, int, bool, int, int, int, int, bool, bool, int, float, bool, int, int, int, int>(),
             py::arg("rom_path"),
             py::arg("num_envs"),
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
        .def("reset", &ale::vector::ALEVectorInterface::reset)
        .def("step", &ale::vector::ALEVectorInterface::step)
        .def("get_action_set", &ale::vector::ALEVectorInterface::get_action_set)
        .def("get_num_envs", &ale::vector::ALEVectorInterface::get_num_envs)
        .def("get_observation_shape", &ale::vector::ALEVectorInterface::get_observation_shape);
}