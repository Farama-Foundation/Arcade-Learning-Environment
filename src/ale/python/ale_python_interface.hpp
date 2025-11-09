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

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/filesystem.h>

#include "ale/ale_interface.hpp"
#include "version.hpp"

#ifdef BUILD_VECTOR_LIB
    #include "ale_vector_python_interface.hpp"
#endif
#ifdef BUILD_VECTOR_XLA_SUPPORT
    #include "ale_vector_xla_interface.hpp"
#endif


namespace nb = nanobind;
void init_vector_module(nb::module_ &m);
void init_vector_module_xla(nb::module_ &m);

using namespace nb::literals;

namespace ale {

class ALEPythonInterface : public ALEInterface {
    public:
        using ALEInterface::ALEInterface;

        // Screen methods use dynamic shapes (varies by game: 210x160 or 250x160)
        void getScreen(nb::ndarray<pixel_t, nb::c_contig, nb::device::cpu>& buffer);
        void getScreenRGB(nb::ndarray<pixel_t, nb::c_contig, nb::device::cpu>& buffer);
        void getScreenGrayscale(nb::ndarray<pixel_t, nb::c_contig, nb::device::cpu>& buffer);

        nb::ndarray<nb::numpy, pixel_t> getScreen();
        nb::ndarray<nb::numpy, pixel_t> getScreenRGB();
        nb::ndarray<nb::numpy, pixel_t> getScreenGrayscale();

        inline reward_t act(unsigned int action, const float paddle_strength = 1.0) {
            return ALEInterface::act(static_cast<Action>(action), paddle_strength);
        }

        inline nb::tuple getScreenDims() const {
            const ALEScreen& screen = ALEInterface::getScreen();
            return nb::make_tuple(screen.height(), screen.width());
        }

        inline uint32_t getAudioSize() const { return ALEInterface::getAudio().size(); }
        // Audio has static shape: (512,) across all games
        nb::ndarray<nb::numpy, uint8_t, nb::shape<512>> getAudio();
        void getAudio(nb::ndarray<uint8_t, nb::shape<512>, nb::c_contig, nb::device::cpu> &buffer);

        // Implicitly cast std::string -> fs::path
        inline void loadROM(const std::string &rom_file) {
            return ALEInterface::loadROM(rom_file);
        }

        // Implicitly cast std::string -> fs::path
        static inline std::optional<std::string> isSupportedROM(const std::string& rom_file) {
            return ALEInterface::isSupportedROM(rom_file);
        }

        inline uint32_t getRAMSize() { return ALEInterface::getRAM().size(); }
        // RAM has static shape: (128,) across all games
        nb::ndarray<nb::numpy, uint8_t, nb::shape<128>> getRAM();
        void getRAM(nb::ndarray<uint8_t, nb::shape<128>, nb::c_contig, nb::device::cpu>& buffer);
};

} // namespace ale

NB_MODULE(_ale_py, m) {
    m.attr("__version__") = ALE_VERSION;
#ifdef ALE_SDL_SUPPORT
    m.attr("SDL_SUPPORT") = true;
#else
    m.attr("SDL_SUPPORT") = false;
#endif

    nb::enum_<ale::Action>(m, "Action")
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

    nb::enum_<ale::Logger::mode>(m, "LoggerMode")
        .value("Info", ale::Logger::mode::Info)
        .value("Warning", ale::Logger::mode::Warning)
        .value("Error", ale::Logger::mode::Error)
        .export_values();

    nb::class_<ale::ALEState>(m, "ALEState")
        .def(nb::init<>())
        .def(nb::init<const ale::ALEState&, const std::string&>())
        .def(nb::init<const std::string&>())
        .def("equals", &ale::ALEState::equals)
        .def("getFrameNumber", &ale::ALEState::getFrameNumber)
        .def("getEpisodeFrameNumber", &ale::ALEState::getEpisodeFrameNumber)
        .def("getDifficulty", &ale::ALEState::getDifficulty)
        .def("getCurrentMode", &ale::ALEState::getCurrentMode)
        .def("serialize", &ale::ALEState::serialize)
        .def("__eq__", &ale::ALEState::equals)
        .def("__getstate__", [](ale::ALEState& a) {
            std::string serialized = a.serialize();
            return nb::bytes(serialized.data(), serialized.size());
        })
        .def("__setstate__", [](ale::ALEState& a, nb::bytes state) {
            new (&a) ale::ALEState(std::string(state.c_str(), state.size()));
        });

    nb::class_<ale::ALEPythonInterface>(m, "ALEInterface")
        .def(nb::init<>())
        .def("getString", &ale::ALEPythonInterface::getString)
        .def("getInt", &ale::ALEPythonInterface::getInt)
        .def("getBool", &ale::ALEPythonInterface::getBool)
        .def("getFloat", &ale::ALEPythonInterface::getFloat)
        .def("setString", &ale::ALEPythonInterface::setString)
        .def("setInt", &ale::ALEPythonInterface::setInt)
        .def("setBool", &ale::ALEPythonInterface::setBool)
        .def("setFloat", &ale::ALEPythonInterface::setFloat)
        .def("loadROM", [](ale::ALEPythonInterface& self, const std::string& rom) {
            self.loadROM(rom);
        })
        .def("loadROM", [](ale::ALEPythonInterface& self, const fs::path& rom) {
            self.ALEInterface::loadROM(rom);
        })
        .def_static("isSupportedROM", [](const std::string& rom) {
            return ale::ALEPythonInterface::isSupportedROM(rom);
        })
        .def_static("isSupportedROM", [](const fs::path& rom) {
            return ale::ALEInterface::isSupportedROM(rom);
        })
        .def("act", [](ale::ALEPythonInterface& self, unsigned int action, float paddle_strength) {
            return self.act(action, paddle_strength);
        }, nb::arg("action"), nb::arg("paddle_strength") = 1.0)
        .def("act", [](ale::ALEPythonInterface& self, ale::Action action, float paddle_strength) {
            return self.ALEInterface::act(action, paddle_strength);
        }, nb::arg("action"), nb::arg("paddle_strength") = 1.0)
        .def("game_over", &ale::ALEPythonInterface::game_over,
            nb::arg("with_truncation") = true)
        .def("game_truncated", &ale::ALEPythonInterface::game_truncated)
        .def("reset_game", &ale::ALEPythonInterface::reset_game)
        .def("getAvailableModes", &ale::ALEPythonInterface::getAvailableModes)
        .def("setMode", &ale::ALEPythonInterface::setMode)
        .def("getAvailableDifficulties", &ale::ALEPythonInterface::getAvailableDifficulties)
        .def("setDifficulty", &ale::ALEPythonInterface::setDifficulty)
        .def("getLegalActionSet", &ale::ALEPythonInterface::getLegalActionSet)
        .def("getMinimalActionSet", &ale::ALEPythonInterface::getMinimalActionSet)
        .def("getFrameNumber", &ale::ALEPythonInterface::getFrameNumber)
        .def("lives", &ale::ALEPythonInterface::lives)
        .def("getEpisodeFrameNumber", &ale::ALEPythonInterface::getEpisodeFrameNumber)
        .def("getScreen", [](ale::ALEPythonInterface& self) {
            return self.getScreen();
        })
        .def("getScreen", [](ale::ALEPythonInterface& self, nb::ndarray<ale::pixel_t, nb::c_contig, nb::device::cpu>& buffer) {
            self.getScreen(buffer);
        })
        .def("getScreenRGB", [](ale::ALEPythonInterface& self) {
            return self.getScreenRGB();
        })
        .def("getScreenRGB", [](ale::ALEPythonInterface& self, nb::ndarray<ale::pixel_t, nb::c_contig, nb::device::cpu>& buffer) {
            self.getScreenRGB(buffer);
        })
        .def("getScreenGrayscale", [](ale::ALEPythonInterface& self) {
            return self.getScreenGrayscale();
        })
        .def("getScreenGrayscale", [](ale::ALEPythonInterface& self, nb::ndarray<ale::pixel_t, nb::c_contig, nb::device::cpu>& buffer) {
            self.getScreenGrayscale(buffer);
        })
        .def("getScreenDims", &ale::ALEPythonInterface::getScreenDims)
        .def("getAudioSize", &ale::ALEPythonInterface::getAudioSize)
        .def("getAudio", [](ale::ALEPythonInterface& self) {
            return self.getAudio();
        })
        .def("getAudio", [](ale::ALEPythonInterface& self, nb::ndarray<uint8_t, nb::shape<512>, nb::c_contig, nb::device::cpu>& buffer) {
            self.getAudio(buffer);
        })
        .def("getRAMSize", &ale::ALEPythonInterface::getRAMSize)
        .def("getRAM", [](ale::ALEPythonInterface& self) {
            return self.getRAM();
        })
        .def("getRAM", [](ale::ALEPythonInterface& self, nb::ndarray<uint8_t, nb::shape<128>, nb::c_contig, nb::device::cpu>& buffer) {
            self.getRAM(buffer);
        })
        .def("setRAM", &ale::ALEPythonInterface::setRAM)
        .def("cloneState", &ale::ALEPythonInterface::cloneState, nb::arg("include_rng") = false)
        .def("restoreState", &ale::ALEPythonInterface::restoreState)
        .def("cloneSystemState", &ale::ALEPythonInterface::cloneSystemState)
        .def("restoreSystemState", &ale::ALEPythonInterface::restoreSystemState)
        .def("saveScreenPNG", &ale::ALEPythonInterface::saveScreenPNG)
        .def_static("setLoggerMode", &ale::Logger::setMode);

  // Initialize the vector module if it's enabled
#ifdef BUILD_VECTOR_LIB
    init_vector_module(m);

    #ifdef BUILD_VECTOR_XLA_LIB
        init_vector_module_xla(m);
    #endif
#endif
}

#endif // __ALE_PYTHON_INTERFACE_HPP__
