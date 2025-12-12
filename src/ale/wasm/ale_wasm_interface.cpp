/* *****************************************************************************
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License version 2
 * as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 * *****************************************************************************
 * A.L.E (Arcade Learning Environment)
 * Copyright (c) 2009-2013 by Yavar Naddaf, Joel Veness, Marc G. Bellemare and
 *   the Reinforcement Learning and Artificial Intelligence Laboratory
 * Released under the GNU General Public License; see License.txt for details.
 *
 * Based on: Stella  --  "An Atari 2600 VCS Emulator"
 * Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
 *
 * *****************************************************************************
 *  ale_wasm_interface.cpp
 *
 *  WebAssembly bindings using Embind.
 **************************************************************************** */

#include <emscripten/bind.h>
#include <emscripten/val.h>
#include "ale/ale_interface.hpp"

using namespace emscripten;
using namespace ale;

namespace ale_wasm {

/**
 * Wrapper class to provide JavaScript-friendly API
 * This wraps ALEInterface and provides memory-safe access to buffers
 */
class ALEInterfaceWrapper {
private:
    ALEInterface ale;
    std::vector<unsigned char> screen_buffer;
    std::vector<unsigned char> ram_buffer;

public:
    ALEInterfaceWrapper() : ale() {
        // Reserve buffers to avoid reallocations
        screen_buffer.reserve(160 * 210 * 3);  // Max screen size RGB
        ram_buffer.reserve(128);  // RAM size
    }

    // Configuration methods
    void setBool(const std::string& key, bool value) {
        ale.setBool(key, value);
    }

    bool getBool(const std::string& key) const {
        return ale.getBool(key);
    }

    void setInt(const std::string& key, int value) {
        ale.setInt(key, value);
    }

    int getInt(const std::string& key) const {
        return ale.getInt(key);
    }

    void setFloat(const std::string& key, float value) {
        ale.setFloat(key, value);
    }

    float getFloat(const std::string& key) const {
        return ale.getFloat(key);
    }

    void setString(const std::string& key, const std::string& value) {
        ale.setString(key, value);
    }

    std::string getString(const std::string& key) const {
        return ale.getString(key);
    }

    // ROM loading (expects ROM to be in virtual filesystem)
    void loadROM(const std::string& rom_path) {
        ale.loadROM(rom_path);
    }

    // Core game loop methods
    int act(int action) {
        return ale.act(static_cast<Action>(action));
    }

    void resetGame() {
        ale.reset_game();
    }

    bool gameOver() const {
        return ale.game_over();
    }

    bool gameTruncated() const {
        return ale.game_truncated();
    }

    int lives() {
        return ale.lives();
    }

    int getFrameNumber() const {
        return ale.getFrameNumber();
    }

    int getEpisodeFrameNumber() const {
        return ale.getEpisodeFrameNumber();
    }

    // Screen access - returns Uint8ClampedArray for Canvas compatibility
    val getScreenRGB() {
        screen_buffer.clear();
        ale.getScreenRGB(screen_buffer);

        // Return as JavaScript typed array view
        return val(typed_memory_view(screen_buffer.size(), screen_buffer.data()));
    }

    val getScreenGrayscale() {
        screen_buffer.clear();
        ale.getScreenGrayscale(screen_buffer);

        return val(typed_memory_view(screen_buffer.size(), screen_buffer.data()));
    }

    // Get screen dimensions
    int getScreenWidth() const {
        return ale.getScreen().width();
    }

    int getScreenHeight() const {
        return ale.getScreen().height();
    }

    // RAM access
    val getRAM() {
        const ALERAM& ram = ale.getRAM();
        return val(typed_memory_view(ram.size(), ram.array()));
    }

    void setRAM(size_t index, uint8_t value) {
        ale.setRAM(index, value);
    }

    // Action space
    val getLegalActionSet() {
        ActionVect actions = ale.getLegalActionSet();
        std::vector<int> action_ints(actions.begin(), actions.end());
        return val::array(action_ints.begin(), action_ints.end());
    }

    val getMinimalActionSet() {
        ActionVect actions = ale.getMinimalActionSet();
        std::vector<int> action_ints(actions.begin(), actions.end());
        return val::array(action_ints.begin(), action_ints.end());
    }

    // Mode/Difficulty
    val getAvailableModes() {
        ModeVect modes = ale.getAvailableModes();
        std::vector<int> mode_ints(modes.begin(), modes.end());
        return val::array(mode_ints.begin(), mode_ints.end());
    }

    int getMode() const {
        return ale.getMode();
    }

    void setMode(int mode) {
        ale.setMode(static_cast<game_mode_t>(mode));
    }

    val getAvailableDifficulties() {
        DifficultyVect diffs = ale.getAvailableDifficulties();
        std::vector<int> diff_ints(diffs.begin(), diffs.end());
        return val::array(diff_ints.begin(), diff_ints.end());
    }

    int getDifficulty() const {
        return ale.getDifficulty();
    }

    void setDifficulty(int difficulty) {
        ale.setDifficulty(static_cast<difficulty_t>(difficulty));
    }

    // Save/restore state
    std::string saveState() {
        ALEState state = ale.cloneState();
        // Serialize state to string (simple approach - could be optimized)
        return state.serialize();
    }

    void loadState(const std::string& serialized) {
        ALEState state(serialized);
        ale.restoreState(state);
    }

    // Version info
    static std::string getVersion() {
        return ALE_VERSION;
    }
};

} // namespace ale_wasm

EMSCRIPTEN_BINDINGS(ale_module) {
    class_<ale_wasm::ALEInterfaceWrapper>("ALEInterface")
        .constructor<>()
        // Configuration
        .function("setBool", &ale_wasm::ALEInterfaceWrapper::setBool)
        .function("getBool", &ale_wasm::ALEInterfaceWrapper::getBool)
        .function("setInt", &ale_wasm::ALEInterfaceWrapper::setInt)
        .function("getInt", &ale_wasm::ALEInterfaceWrapper::getInt)
        .function("setFloat", &ale_wasm::ALEInterfaceWrapper::setFloat)
        .function("getFloat", &ale_wasm::ALEInterfaceWrapper::getFloat)
        .function("setString", &ale_wasm::ALEInterfaceWrapper::setString)
        .function("getString", &ale_wasm::ALEInterfaceWrapper::getString)
        // ROM loading
        .function("loadROM", &ale_wasm::ALEInterfaceWrapper::loadROM)
        // Game loop
        .function("act", &ale_wasm::ALEInterfaceWrapper::act)
        .function("resetGame", &ale_wasm::ALEInterfaceWrapper::resetGame)
        .function("gameOver", &ale_wasm::ALEInterfaceWrapper::gameOver)
        .function("gameTruncated", &ale_wasm::ALEInterfaceWrapper::gameTruncated)
        .function("lives", &ale_wasm::ALEInterfaceWrapper::lives)
        .function("getFrameNumber", &ale_wasm::ALEInterfaceWrapper::getFrameNumber)
        .function("getEpisodeFrameNumber", &ale_wasm::ALEInterfaceWrapper::getEpisodeFrameNumber)
        // Screen access
        .function("getScreenRGB", &ale_wasm::ALEInterfaceWrapper::getScreenRGB)
        .function("getScreenGrayscale", &ale_wasm::ALEInterfaceWrapper::getScreenGrayscale)
        .function("getScreenWidth", &ale_wasm::ALEInterfaceWrapper::getScreenWidth)
        .function("getScreenHeight", &ale_wasm::ALEInterfaceWrapper::getScreenHeight)
        // RAM access
        .function("getRAM", &ale_wasm::ALEInterfaceWrapper::getRAM)
        .function("setRAM", &ale_wasm::ALEInterfaceWrapper::setRAM)
        // Action space
        .function("getLegalActionSet", &ale_wasm::ALEInterfaceWrapper::getLegalActionSet)
        .function("getMinimalActionSet", &ale_wasm::ALEInterfaceWrapper::getMinimalActionSet)
        // Mode/Difficulty
        .function("getAvailableModes", &ale_wasm::ALEInterfaceWrapper::getAvailableModes)
        .function("getMode", &ale_wasm::ALEInterfaceWrapper::getMode)
        .function("setMode", &ale_wasm::ALEInterfaceWrapper::setMode)
        .function("getAvailableDifficulties", &ale_wasm::ALEInterfaceWrapper::getAvailableDifficulties)
        .function("getDifficulty", &ale_wasm::ALEInterfaceWrapper::getDifficulty)
        .function("setDifficulty", &ale_wasm::ALEInterfaceWrapper::setDifficulty)
        // Save/restore state
        .function("saveState", &ale_wasm::ALEInterfaceWrapper::saveState)
        .function("loadState", &ale_wasm::ALEInterfaceWrapper::loadState)
        // Static methods
        .class_function("getVersion", &ale_wasm::ALEInterfaceWrapper::getVersion)
        ;
}
