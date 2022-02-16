/* *****************************************************************************
 * The line 99 is based on Xitari's code, from Google Inc.
 *
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
 *  ale_interface.hpp
 *
 *  The shared library interface.
 **************************************************************************** */

#ifndef __ALE_INTERFACE_HPP__
#define __ALE_INTERFACE_HPP__

#include "emucore/OSystem.hxx"
#include "games/Roms.hpp"
#include "environment/stella_environment.hpp"
#include "common/ScreenExporter.hpp"
#include "common/Log.hpp"
#include "version.hpp"

#include <string>
#include <optional>
#include <memory>
#include <filesystem>

namespace fs = std::filesystem;

namespace ale {

/**
   This class interfaces ALE with external code for controlling agents.
 */
class ALEInterface {
 public:
  ALEInterface();
  ~ALEInterface();
  // Legacy constructor
  ALEInterface(bool display_screen);

  // Get the value of a setting.
  std::string getString(const std::string& key);
  int getInt(const std::string& key);
  bool getBool(const std::string& key);
  float getFloat(const std::string& key);

  // getStringInplace is a version of getString that returns a reference to the
  // actual, stored settings string object, without making a copy. The reference
  // is only valid until the next call of any of the setter functions below, so
  // this function must be used with care.
  const std::string& getStringInplace(const std::string& key);

  // Set the value of a setting. loadRom() must be called before the
  // setting will take effect.
  void setString(const std::string& key, const std::string& value);
  void setInt(const std::string& key, const int value);
  void setBool(const std::string& key, const bool value);
  void setFloat(const std::string& key, const float value);

  // Resets the Atari and loads a game. After this call the game
  // should be ready to play. This is necessary after changing a
  // setting for the setting to take effect. Optionally specify
  // a new ROM to load.
  void loadROM(fs::path rom_file = {});

  // Applies an action to the game and returns the reward. It is the
  // user's responsibility to check if the game has ended and reset
  // when necessary - this method will keep pressing buttons on the
  // game over screen.
  reward_t act(Action action);

  // Indicates if the game has ended.
  bool game_over() const;

  // Resets the game, but not the full system.
  void reset_game();

  // Returns the vector of modes available for the current game.
  // This should be called only after the rom is loaded.
  ModeVect getAvailableModes();

  // Sets the mode of the game.
  // The mode must be an available mode (otherwise it throws an exception).
  // This should be called only after the rom is loaded.
  void setMode(game_mode_t m);

  // Returns the game mode value last specified to the environment.
  // This may not be the exact game mode that the ROM is currently running as
  // game mode changes only take effect when the environment is reset.
  game_mode_t getMode() const { return environment->getMode(); }

  //Returns the vector of difficulties available for the current game.
  //This should be called only after the rom is loaded. Notice
  // that there are 2 levers, the right and left switches. They
  // are not tied to any specific player. In Venture, for example,
  // we have the following interpretation for the difficulties:
  // Skill          Switch
  // Level          Setting
  //   1         left B/right B
  //   2         left B/right A
  //   3         left A/right B
  //   4         left A/right A
  DifficultyVect getAvailableDifficulties();

  // Sets the difficulty of the game.
  // The difficulty must be an available mode (otherwise it throws an exception).
  // This should be called only after the rom is loaded.
  void setDifficulty(difficulty_t m);

  // Returns the current difficulty switch setting in use by the environment.
  difficulty_t getDifficulty() const { return environment->getDifficulty(); }

  // Returns the vector of legal actions. This should be called only
  // after the rom is loaded.
  ActionVect getLegalActionSet();

  // Returns the vector of the minimal set of actions needed to play
  // the game.
  ActionVect getMinimalActionSet();

  // Returns the frame number since the loading of the ROM
  int getFrameNumber();

  // The remaining number of lives.
  int lives();

  // Returns the frame number since the start of the current episode
  int getEpisodeFrameNumber() const;

  // Returns the current game screen
  const ALEScreen& getScreen();

  //This method should receive an empty vector to fill it with
  //the grayscale colours
  void getScreenGrayscale(std::vector<unsigned char>& grayscale_output_buffer);

  //This method should receive a vector to fill it with
  //the RGB colours. The first positions contain the red colours,
  //followed by the green colours and then the blue colours
  void getScreenRGB(std::vector<unsigned char>& output_rgb_buffer);

  // Returns the current RAM content
  const ALERAM& getRAM();

  // Set byte at memory address. This can be useful to change the environment
  // for example if you were trying to learn a causal model of RAM locations.
  void setRAM(size_t memory_index, byte_t value);

  // This makes a copy of the environment state. By defualt this copy does *not* include pseudorandomness
  // making it suitable for planning purposes. If `include_prng` is set to true, then the
  // pseudorandom number generator is also serialized.
  ALEState cloneState(bool include_rng = false);

  // Reverse operation of cloneState(). This will restore the ALEState, if it was
  // cloned including the RNG then the RNG will be restored. Otherwise the current
  // state of the RNG will be kept as is.
  void restoreState(const ALEState& state);

  // This makes a copy of the system & environment state, suitable for serialization. This includes
  // pseudorandomness and so is *not* suitable for planning purposes.
  // This is equivalent to calling cloneState(true) but is maintained for backwards compatibility.
  ALEState cloneSystemState();

  // Reverse operation of cloneSystemState.
  // This is maintained for backwards compatability and is equivalent to calling restoreState(state).
  void restoreSystemState(const ALEState& state);

  // Save the current screen as a png file
  void saveScreenPNG(const std::string& filename);

  // Creates a ScreenExporter object which can be used to save a sequence of frames. Ownership
  // said object is passed to the caller. Frames are saved in the directory 'path', which needs
  // to exists.
  ScreenExporter* createScreenExporter(const std::string& path) const;

 public:
  std::unique_ptr<stella::OSystem> theOSystem;
  std::unique_ptr<stella::Settings> theSettings;
  std::unique_ptr<RomSettings> romSettings;
  std::unique_ptr<StellaEnvironment> environment;
  int max_num_frames; // Maximum number of frames for each episode

 public:
  // Check if the rom with filename matches a supported MD5
  static std::optional<std::string> isSupportedROM(const fs::path& rom_file);
  // Display ALE welcome message
  static std::string welcomeMessage();
  static void disableBufferedIO();
  static void createOSystem(std::unique_ptr<stella::OSystem>& theOSystem,
                            std::unique_ptr<stella::Settings>& theSettings);
  static void loadSettings(const fs::path& romfile,
                           std::unique_ptr<stella::OSystem>& theOSystem);
};

}  // namespace ale

#endif  // __ALE_INTERFACE_HPP__
