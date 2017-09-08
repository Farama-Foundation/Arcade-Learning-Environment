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

#include "emucore/FSNode.hxx"
#include "emucore/OSystem.hxx"
#include "os_dependent/SettingsWin32.hxx"
#include "os_dependent/OSystemWin32.hxx"
#include "os_dependent/SettingsUNIX.hxx"
#include "os_dependent/OSystemUNIX.hxx"
#include "games/Roms.hpp"
#include "common/display_screen.h"
#include "environment/stella_environment.hpp"
#include "common/ScreenExporter.hpp"
#include "common/Log.hpp"

#include <string>
#include <memory>

static const std::string Version = "0.6.0";

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

  // Set the value of a setting. loadRom() must be called before the
  // setting will take effect.
  void setString(const std::string& key, const std::string& value);
  void setInt(const std::string& key, const int value);
  void setBool(const std::string& key, const bool value);
  void setFloat(const std::string& key, const float value);

  // Resets the Atari and loads a game. After this call the game
  // should be ready to play. This is necessary after changing a
  // setting for the setting to take effect.
  void loadROM(std::string rom_file);

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
  const ALEScreen &getScreen();

  //This method should receive an empty vector to fill it with
  //the grayscale colours
  void getScreenGrayscale(std::vector<unsigned char>& grayscale_output_buffer);

  //This method should receive a vector to fill it with
  //the RGB colours. The first positions contain the red colours,
  //followed by the green colours and then the blue colours
  void getScreenRGB(std::vector<unsigned char>& output_rgb_buffer);

  // Returns the current RAM content
  const ALERAM &getRAM();

  // Saves the state of the system
  void saveState();

  // Loads the state of the system
  void loadState();

  // This makes a copy of the environment state. This copy does *not* include pseudorandomness,
  // making it suitable for planning purposes. By contrast, see cloneSystemState.
  ALEState cloneState();

  // Reverse operation of cloneState(). This does not restore pseudorandomness, so that repeated
  // calls to restoreState() in the stochastic controls setting will not lead to the same outcomes.
  // By contrast, see restoreSystemState.
  void restoreState(const ALEState& state);

  // This makes a copy of the system & environment state, suitable for serialization. This includes
  // pseudorandomness and so is *not* suitable for planning purposes.
  ALEState cloneSystemState();

  // Reverse operation of cloneSystemState.
  void restoreSystemState(const ALEState& state);

  // Save the current screen as a png file
  void saveScreenPNG(const std::string& filename);

  // Creates a ScreenExporter object which can be used to save a sequence of frames. Ownership 
  // said object is passed to the caller. Frames are saved in the directory 'path', which needs
  // to exists. 
  ScreenExporter *createScreenExporter(const std::string &path) const;

 public:
  std::unique_ptr<OSystem> theOSystem;
  std::unique_ptr<Settings> theSettings;
  std::unique_ptr<RomSettings> romSettings;
  std::unique_ptr<StellaEnvironment> environment;
  int max_num_frames; // Maximum number of frames for each episode

 public:
  // Display ALE welcome message
  static std::string welcomeMessage();
  static void disableBufferedIO();
  static void createOSystem(std::unique_ptr<OSystem> &theOSystem,
                            std::unique_ptr<Settings> &theSettings);
  static void loadSettings(const std::string& romfile,
                           std::unique_ptr<OSystem> &theOSystem);

 private:
  static void checkForUnsupportedRom(std::unique_ptr<OSystem>& theOSystem);
};

#endif
