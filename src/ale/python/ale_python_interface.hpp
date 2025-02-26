/* *****************************************************************************
 * A.L.E (Arcade Learning Environment)
 * Copyright (c) 2009-2013 by Yavar Naddaf, Joel Veness, Marc G. Bellemare and
 *   the Reinforcement Learning and Artificial Intelligence Laboratory
 * Released under the GNU General Public License; see License.txt for details.
 *
 * Based on: Stella  --  "An Atari 2600 VCS Emulator"
 * Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
 *
 * *****************************************************************************
 *  ale_python_interface.hpp
 *
 *  Bindings for the ALE Python Interface.
 *
 **************************************************************************** */
#ifndef __ALE_PYTHON_INTERFACE_HPP__
#define __ALE_PYTHON_INTERFACE_HPP__

#include <sstream>
#include <optional>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "ale/ale_interface.hpp"
#include "version.hpp"

namespace py = pybind11;
using namespace py::literals;

namespace ale {

class ALEPythonInterface : public ALEInterface {
 public:
  using ALEInterface::ALEInterface;

  void getScreen(py::array_t<pixel_t, py::array::c_style>& buffer);
  void getScreenRGB(py::array_t<pixel_t, py::array::c_style>& buffer);
  void getScreenGrayscale(py::array_t<pixel_t, py::array::c_style>& buffer);

  py::array_t<pixel_t, py::array::c_style> getScreen();
  py::array_t<pixel_t, py::array::c_style> getScreenRGB();
  py::array_t<pixel_t, py::array::c_style> getScreenGrayscale();

  inline reward_t act(unsigned int action, float paddle_strength = 1.0) {
    return ALEInterface::act((Action)action, paddle_strength);
  }

  inline py::tuple getScreenDims() {
    const ALEScreen& screen = ALEInterface::getScreen();
    return py::make_tuple(screen.height(), screen.width());
  }

  inline uint32_t getAudioSize() { return ALEInterface::getAudio().size(); }
  const py::array_t<uint8_t, py::array::c_style> getAudio();
  void getAudio(py::array_t<uint8_t, py::array::c_style> &buffer);

  // Implicitely cast std::string -> fs::path
  inline void loadROM(std::string rom_file) {
    return ALEInterface::loadROM(rom_file);
  }

  // Implicitely cast std::string -> fs::path
  static inline std::optional<std::string> isSupportedROM(const std::string& rom_file) {
    return ALEInterface::isSupportedROM(rom_file);
  }

  inline uint32_t getRAMSize() { return ALEInterface::getRAM().size(); }
  const py::array_t<uint8_t, py::array::c_style> getRAM();
  void getRAM(py::array_t<uint8_t, py::array::c_style>& buffer);
};

// Add to existing file, at the end before the PYBIND11_MODULE section

#ifdef BUILD_VECTOR_LIB
#include "ale/vector/atari_env.h"

void init_vector_bindings(py::module& m) {
    py::module vector = m.def_submodule("_vector", "Vector ALE bindings");

    py::class_<atari::AtariEnvPool>(vector, "AtariEnvPool")
        .def(py::init([](const py::dict& config) {
            // Convert Python dict to C++ config
            auto c = atari::AtariEnvFns::DefaultConfig();
            // Set config values from Python dict
            if (config.contains("num_envs"))
                c["num_envs"_] = config["num_envs"].cast<int>();
            if (config.contains("game"))
                c["task"_] = config["game"].cast<std::string>();
            if (config.contains("frame_skip"))
                c["frame_skip"_] = config["frame_skip"].cast<int>();
            if (config.contains("repeat_action_probability"))
                c["repeat_action_probability"_] = config["repeat_action_probability"].cast<float>();
            if (config.contains("minimal_actions"))
                c["full_action_space"_] = !config["minimal_actions"].cast<bool>();
            if (config.contains("max_episode_steps"))
                c["max_episode_steps"_] = config["max_episode_steps"].cast<int>();
            if (config.contains("num_threads"))
                c["num_threads"_] = config["num_threads"].cast<int>();
            if (config.contains("seed"))
                c["seed"_] = config["seed"].cast<int>();
            if (config.contains("obs_type")) {
                std::string obs_type = config["obs_type"].cast<std::string>();
                c["gray_scale"_] = (obs_type == "grayscale");
            }

            // Create spec and environment pool
            atari::AtariEnvSpec spec(c.AllValues());
            return std::make_unique<atari::AtariEnvPool>(spec);
        }))
        .def("reset", &atari::AtariEnvPool::PyReset)
        .def("step", &atari::AtariEnvPool::PySend, &atari::AtariEnvPool::PyRecv)
        .def("close", [](atari::AtariEnvPool& self) {
            // Cleanup resources
        })
        .def("get_action_set", [](atari::AtariEnvPool& self) {
            // Return full action set
            // Implementation needed
            return py::list();
        })
        .def("get_minimal_action_set", [](atari::AtariEnvPool& self) {
            // Return minimal action set
            // Implementation needed
            return py::list();
        });
}
#endif

} // namespace ale

PYBIND11_MODULE(_ale_py, m) {
  m.attr("__version__") = py::str(ALE_VERSION);
#ifdef ALE_SDL_SUPPORT
  m.attr("SDL_SUPPORT") = py::bool_(true);
#else
  m.attr("SDL_SUPPORT") = py::bool_(false);
#endif
#ifdef BUILD_VECTOR_LIB
    init_vector_bindings(m);
#endif

  py::enum_<ale::Action>(m, "Action")
      .value("NOOP", ale::PLAYER_A_NOOP)
      .value("FIRE", ale::PLAYER_A_FIRE)
      .value("UP", ale::PLAYER_A_UP)
      .value("RIGHT", ale::PLAYER_A_RIGHT)
      .value("LEFT", ale::PLAYER_A_LEFT)
      .value("DOWN", ale::PLAYER_A_DOWN)
      .value("UPRIGHT", ale::PLAYER_A_UPRIGHT)
      .value("UPLEFT", ale::PLAYER_A_UPLEFT)
      .value("DOWNRIGHT", ale::PLAYER_A_DOWNRIGHT)
      .value("DOWNLEFT", ale::PLAYER_A_DOWNLEFT)
      .value("UPFIRE", ale::PLAYER_A_UPFIRE)
      .value("RIGHTFIRE", ale::PLAYER_A_RIGHTFIRE)
      .value("LEFTFIRE", ale::PLAYER_A_LEFTFIRE)
      .value("DOWNFIRE", ale::PLAYER_A_DOWNFIRE)
      .value("UPRIGHTFIRE", ale::PLAYER_A_UPRIGHTFIRE)
      .value("UPLEFTFIRE", ale::PLAYER_A_UPLEFTFIRE)
      .value("DOWNRIGHTFIRE", ale::PLAYER_A_DOWNRIGHTFIRE)
      .value("DOWNLEFTFIRE", ale::PLAYER_A_DOWNLEFTFIRE)
      .export_values();

  py::enum_<ale::Logger::mode>(m, "LoggerMode")
      .value("Info", ale::Logger::mode::Info)
      .value("Warning", ale::Logger::mode::Warning)
      .value("Error", ale::Logger::mode::Error)
      .export_values();

  py::class_<ale::ALEState>(m, "ALEState")
      .def(py::init<>())
      .def(py::init<const ale::ALEState&, const std::string&>())
      .def(py::init<const std::string&>())
      .def("equals", &ale::ALEState::equals)
      .def("getFrameNumber", &ale::ALEState::getFrameNumber)
      .def("getEpisodeFrameNumber", &ale::ALEState::getEpisodeFrameNumber)
      .def("getDifficulty", &ale::ALEState::getDifficulty)
      .def("getCurrentMode", &ale::ALEState::getCurrentMode)
      .def("serialize", &ale::ALEState::serialize)
      .def("__eq__", &ale::ALEState::equals)
      .def(py::pickle(
          [](ale::ALEState& a) {
            return py::make_tuple(py::bytes(a.serialize()));
          },
          [](py::tuple t) {
            if (t.size() != 1)
              throw std::runtime_error("Invalid ALEState state...");

            ale::ALEState state(t[0].cast<std::string>());
            return state;
          }));

  py::class_<ale::ALEPythonInterface>(m, "ALEInterface")
      .def(py::init<>())
      .def("getString", &ale::ALEPythonInterface::getString)
      .def("getInt", &ale::ALEPythonInterface::getInt)
      .def("getBool", &ale::ALEPythonInterface::getBool)
      .def("getFloat", &ale::ALEPythonInterface::getFloat)
      .def("setString", &ale::ALEPythonInterface::setString)
      .def("setInt", &ale::ALEPythonInterface::setInt)
      .def("setBool", &ale::ALEPythonInterface::setBool)
      .def("setFloat", &ale::ALEPythonInterface::setFloat)
      .def("loadROM", &ale::ALEPythonInterface::loadROM)
      .def("loadROM", &ale::ALEInterface::loadROM)
      .def_static("isSupportedROM", &ale::ALEPythonInterface::isSupportedROM)
      .def_static("isSupportedROM", &ale::ALEInterface::isSupportedROM)
      .def("act", (ale::reward_t(ale::ALEPythonInterface::*)(uint32_t, float)) &
                      ale::ALEPythonInterface::act,
                      py::arg("action"),
                      py::arg("paddle_strength") = 1.0)
      .def("act", (ale::reward_t(ale::ALEInterface::*)(ale::Action, float)) &
                      ale::ALEInterface::act,
                      py::arg("action"),
                      py::arg("paddle_strength") = 1.0)
      .def("game_over", &ale::ALEPythonInterface::game_over, py::kw_only(), py::arg("with_truncation") = py::bool_(true))
      .def("game_truncated", &ale::ALEPythonInterface::game_truncated)
      .def("reset_game", &ale::ALEPythonInterface::reset_game)
      .def("getAvailableModes", &ale::ALEPythonInterface::getAvailableModes)
      .def("setMode", &ale::ALEPythonInterface::setMode)
      .def("getAvailableDifficulties",
           &ale::ALEPythonInterface::getAvailableDifficulties)
      .def("setDifficulty", &ale::ALEPythonInterface::setDifficulty)
      .def("getLegalActionSet", &ale::ALEPythonInterface::getLegalActionSet)
      .def("getMinimalActionSet", &ale::ALEPythonInterface::getMinimalActionSet)
      .def("getFrameNumber", &ale::ALEPythonInterface::getFrameNumber)
      .def("lives", &ale::ALEPythonInterface::lives)
      .def("getEpisodeFrameNumber",
           &ale::ALEPythonInterface::getEpisodeFrameNumber)
      .def("getScreen", (void (ale::ALEPythonInterface::*)(
                            py::array_t<ale::pixel_t, py::array::c_style>&)) &
                            ale::ALEPythonInterface::getScreen)
      .def("getScreen", (py::array_t<ale::pixel_t, py::array::c_style>(
                            ale::ALEPythonInterface::*)()) &
                            ale::ALEPythonInterface::getScreen)
      .def("getScreenRGB",
           (void (ale::ALEPythonInterface::*)(
               py::array_t<ale::pixel_t, py::array::c_style>&)) &
               ale::ALEPythonInterface::getScreenRGB)
      .def("getScreenRGB", (py::array_t<ale::pixel_t, py::array::c_style>(
                               ale::ALEPythonInterface::*)()) &
                               ale::ALEPythonInterface::getScreenRGB)
      .def("getScreenGrayscale",
           (void (ale::ALEPythonInterface::*)(
               py::array_t<ale::pixel_t, py::array::c_style>&)) &
               ale::ALEPythonInterface::getScreenGrayscale)
      .def("getScreenGrayscale",
           (py::array_t<ale::pixel_t, py::array::c_style>(
               ale::ALEPythonInterface::*)()) &
               ale::ALEPythonInterface::getScreenGrayscale)
      .def("getScreenDims", &ale::ALEPythonInterface::getScreenDims)
      .def("getAudioSize", &ale::ALEPythonInterface::getAudioSize)
      .def("getAudio", (const py::array_t<uint8_t, py::array::c_style> (
                           ale::ALEPythonInterface::*)()) &
                           ale::ALEPythonInterface::getAudio)
      .def("getAudio", (void (ale::ALEPythonInterface::*)(
                           py::array_t<uint8_t, py::array::c_style> &)) &
                           ale::ALEPythonInterface::getAudio)
      .def("getRAMSize", &ale::ALEPythonInterface::getRAMSize)
      .def("getRAM", (const py::array_t<uint8_t, py::array::c_style> (
                         ale::ALEPythonInterface::*)()) &
                         ale::ALEPythonInterface::getRAM)
      .def("getRAM", (void (ale::ALEPythonInterface::*)(
                         py::array_t<uint8_t, py::array::c_style>&)) &
                         ale::ALEPythonInterface::getRAM)
      .def("setRAM", &ale::ALEPythonInterface::setRAM)
      .def("cloneState", &ale::ALEPythonInterface::cloneState, py::kw_only(), py::arg("include_rng") = py::bool_(false))
      .def("restoreState", &ale::ALEPythonInterface::restoreState)
      .def("cloneSystemState", &ale::ALEPythonInterface::cloneSystemState)
      .def("restoreSystemState", &ale::ALEPythonInterface::restoreSystemState)
      .def("saveScreenPNG", &ale::ALEPythonInterface::saveScreenPNG)
      .def_static("setLoggerMode", &ale::Logger::setMode);
}

#endif // __ALE_PYTHON_INTERFACE_HPP__
