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

    // Get screen as RGBA ImageData format (ready for Canvas ImageData)
    val getScreenImageData() {
        screen_buffer.clear();
        ale.getScreenRGB(screen_buffer);

        const int width = ale.getScreen().width();
        const int height = ale.getScreen().height();

        // Convert RGB to RGBA format
        std::vector<unsigned char> rgba_buffer(width * height * 4);
        for (int i = 0; i < width * height; i++) {
            rgba_buffer[i * 4] = screen_buffer[i * 3];       // R
            rgba_buffer[i * 4 + 1] = screen_buffer[i * 3 + 1]; // G
            rgba_buffer[i * 4 + 2] = screen_buffer[i * 3 + 2]; // B
            rgba_buffer[i * 4 + 3] = 255;                       // A
        }

        // Create and return ImageData object
        val imageDataConstructor = val::global("ImageData");
        val uint8ClampedArray = val::global("Uint8ClampedArray").new_(
            typed_memory_view(rgba_buffer.size(), rgba_buffer.data())
        );
        return imageDataConstructor.new_(uint8ClampedArray, width, height);
    }

    // Render to canvas element (takes canvas ID as string)
    void renderToCanvas(const std::string& canvasId) {
        const int width = ale.getScreen().width();
        const int height = ale.getScreen().height();

        screen_buffer.clear();
        ale.getScreenRGB(screen_buffer);

        // Convert RGB to RGBA
        std::vector<unsigned char> rgba_buffer(width * height * 4);
        for (int i = 0; i < width * height; i++) {
            rgba_buffer[i * 4] = screen_buffer[i * 3];
            rgba_buffer[i * 4 + 1] = screen_buffer[i * 3 + 1];
            rgba_buffer[i * 4 + 2] = screen_buffer[i * 3 + 2];
            rgba_buffer[i * 4 + 3] = 255;
        }

        // Use embind val to interact with JavaScript DOM
        val document = val::global("document");
        val canvas = document.call<val>("getElementById", canvasId);
        if (canvas.isNull() || canvas.isUndefined()) {
            val console = val::global("console");
            console.call<void>("error", std::string("Canvas not found: ") + canvasId);
            return;
        }

        val ctx = canvas.call<val>("getContext", std::string("2d"));

        // Set canvas size if needed
        if (canvas["width"].as<int>() != width || canvas["height"].as<int>() != height) {
            canvas.set("width", width);
            canvas.set("height", height);
        }

        // Create ImageData from RGBA buffer
        val uint8ClampedArray = val::global("Uint8ClampedArray").new_(
            typed_memory_view(rgba_buffer.size(), rgba_buffer.data())
        );
        val imageData = val::global("ImageData").new_(uint8ClampedArray, width, height);
        ctx.call<void>("putImageData", imageData, 0, 0);
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
    val saveState() {
        ALEState state = ale.cloneState();
        std::string serialized = state.serialize();

        // Return as Uint8Array to preserve binary data
        std::vector<unsigned char> buffer(serialized.begin(), serialized.end());
        return val(typed_memory_view(buffer.size(), buffer.data()));
    }

    void loadState(const val& uint8Array) {
        // Convert Uint8Array to std::string
        const unsigned int length = uint8Array["length"].as<unsigned int>();
        std::string serialized;
        serialized.reserve(length);

        for (unsigned int i = 0; i < length; i++) {
            serialized.push_back(uint8Array[i].as<unsigned char>());
        }

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
        .function("getScreenImageData", &ale_wasm::ALEInterfaceWrapper::getScreenImageData)
        .function("renderToCanvas", &ale_wasm::ALEInterfaceWrapper::renderToCanvas)
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
