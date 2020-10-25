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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ale_interface.hpp"
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

  inline reward_t act(unsigned int action) {
    return ALEInterface::act((Action)action);
  }

  inline py::tuple getScreenDims() {
    const ALEScreen& screen = ALEInterface::getScreen();
    return py::make_tuple(screen.height(), screen.width());
  }

  inline uint32_t getRAMSize() { return ALEInterface::getRAM().size(); }
  const py::array_t<uint8_t, py::array::c_style> getRAM();
  void getRAM(py::array_t<uint8_t, py::array::c_style>& buffer);
};

} // namespace ale

PYBIND11_MODULE(ale_py, m) {
  m.attr("__version__") = py::str(ALE_VERSION_STR);
#ifdef __USE_SDL
  m.attr("SDL") = py::bool_(true);
#else
  m.attr("SDL") = py::bool_(false);
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
      .def_static("isSupportedRom", &ale::ALEPythonInterface::isSupportedRom)
      .def("act", (ale::reward_t(ale::ALEPythonInterface::*)(uint32_t)) &
                      ale::ALEPythonInterface::act)
      .def("act", (ale::reward_t(ale::ALEInterface::*)(ale::Action)) &
                      ale::ALEInterface::act)
      .def("game_over", &ale::ALEPythonInterface::game_over)
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
      .def("getRAMSize", &ale::ALEPythonInterface::getRAMSize)
      .def("getRAM", (const py::array_t<uint8_t, py::array::c_style> (
                         ale::ALEPythonInterface::*)()) &
                         ale::ALEPythonInterface::getRAM)
      .def("getRAM", (void (ale::ALEPythonInterface::*)(
                         py::array_t<uint8_t, py::array::c_style>&)) &
                         ale::ALEPythonInterface::getRAM)
      .def("saveState", &ale::ALEPythonInterface::saveState)
      .def("loadState", &ale::ALEPythonInterface::loadState)
      .def("cloneState", &ale::ALEPythonInterface::cloneState)
      .def("restoreState", &ale::ALEPythonInterface::restoreState)
      .def("cloneSystemState", &ale::ALEPythonInterface::cloneSystemState)
      .def("restoreSystemState", &ale::ALEPythonInterface::restoreSystemState)
      .def("saveScreenPNG", &ale::ALEPythonInterface::saveScreenPNG)
      .def_static("setLoggerMode", &ale::Logger::setMode);
}

#endif // __ALE_PYTHON_INTERFACE_HPP__
