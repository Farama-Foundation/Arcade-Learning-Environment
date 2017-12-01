/* *****************************************************************************
 * The lines 201 - 204 are based on Xitari's code, from Google Inc.
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
 *  ale_interface.cpp
 *
 *  The shared library interface.
 **************************************************************************** */

#include "ale_interface.hpp"

#include <stddef.h>
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "common/ColourPalette.hpp"
#include "common/Constants.h"
#include "emucore/Console.hxx"
#include "emucore/Props.hxx"
#include "environment/ale_screen.hpp"
#include "games/RomSettings.hpp"

using namespace ale;

// Display ALE welcome message
std::string ALEInterface::welcomeMessage() {
  std::ostringstream oss;
  oss << "A.L.E: Arcade Learning Environment (version " << Version << ")\n"
      << "[Powered by Stella]\n" << "Use -help for help screen.";
  return oss.str();
}

void ALEInterface::disableBufferedIO() {
  setvbuf(stdout, NULL, _IONBF, 0);
  setvbuf(stdin, NULL, _IONBF, 0);
  std::cin.rdbuf()->pubsetbuf(0, 0);
  std::cout.rdbuf()->pubsetbuf(0, 0);
  std::cin.sync_with_stdio();
  std::cout.sync_with_stdio();
}

void ALEInterface::createOSystem(std::unique_ptr<OSystem> &theOSystem,
                          std::unique_ptr<Settings> &theSettings) {
#if (defined(WIN32) || defined(__MINGW32__))
  theOSystem.reset(new OSystemWin32());
  theSettings.reset(new SettingsWin32(theOSystem.get()));
#else
  theOSystem.reset(new OSystemUNIX());
  theSettings.reset(new SettingsUNIX(theOSystem.get()));
#endif

  theOSystem->settings().loadConfig();
}

void ALEInterface::checkForUnsupportedRom(std::unique_ptr<OSystem>& theOSystem) {
  const Properties properties = theOSystem->console().properties();
  const std::string md5 = properties.get(Cartridge_MD5);
  bool found = false;
  std::ifstream ss("md5.txt");
  std::string item;
  while (!found && std::getline(ss, item)) {
    if (!item.compare(0, md5.size(), md5)) {
      const std::string rom_candidate = item.substr(md5.size() + 1);
      found = true;
    }
  }
  if (!found) {
    // If the md5 doesn't match our master list, warn the user. 
    Logger::Warning << std::endl;
    Logger::Warning << "WARNING: Possibly unsupported ROM: mismatched MD5." << std::endl;
    Logger::Warning << "Cartridge_MD5: " << md5 << std::endl;
    const std::string name = properties.get(Cartridge_Name);
    Logger::Warning << "Cartridge_name: " << name << std::endl;
    Logger::Warning << std::endl;
  }
}

void ALEInterface::loadSettings(const std::string& romfile,
                                std::unique_ptr<OSystem> &theOSystem) {
  // Load the configuration from a config file (passed on the command
  //  line), if provided
  std::string configFile = theOSystem->settings().getString("config", false);

  if (!configFile.empty()) {
    theOSystem->settings().loadConfig(configFile.c_str());
  }

  theOSystem->settings().validate();
  theOSystem->create();

  // Attempt to load the ROM
  if (romfile == "") {
    Logger::Error << "No ROM File specified." << std::endl;
    exit(1);
  } else if (!FilesystemNode::fileExists(romfile)) {
    Logger::Error << "ROM file " << romfile << " not found." << std::endl;
    exit(1);
  } else if (theOSystem->createConsole(romfile))  {
    checkForUnsupportedRom(theOSystem);
    Logger::Info << "Running ROM file..." << std::endl;
    theOSystem->settings().setString("rom_file", romfile);
  } else {
    Logger::Error << "Unable to create console for " << romfile << std::endl;
    exit(1);
  }

// Must force the resetting of the OSystem's random seed, which is set before we change
// choose our random seed.
  Logger::Info << "Random seed is "
      << theOSystem->settings().getInt("random_seed") << std::endl;
  theOSystem->resetRNGSeed();

  std::string currentDisplayFormat = theOSystem->console().getFormat();
  theOSystem->colourPalette().setPalette("standard", currentDisplayFormat);
}

ALEInterface::ALEInterface() {
  disableBufferedIO();
  Logger::Info << welcomeMessage() << std::endl;
  createOSystem(theOSystem, theSettings);
}

ALEInterface::ALEInterface(bool display_screen) {
  disableBufferedIO();
  Logger::Info << welcomeMessage() << std::endl;
  createOSystem(theOSystem, theSettings);
  this->setBool("display_screen", display_screen);
}

ALEInterface::~ALEInterface() {
}

// Loads and initializes a game. After this call the game should be
// ready to play. Resets the OSystem/Console/Environment/etc. This is
// necessary after changing a setting. Optionally specify a new rom to
// load.
void ALEInterface::loadROM(std::string rom_file = "") {
  assert(theOSystem.get());
  if (rom_file.empty()) {
    rom_file = theOSystem->romFile();
  }
  loadSettings(rom_file, theOSystem);
  romSettings.reset(buildRomRLWrapper(rom_file));
  environment.reset(new StellaEnvironment(theOSystem.get(), romSettings.get()));
  max_num_frames = theOSystem->settings().getInt("max_num_frames_per_episode");
  environment->reset();
#ifndef __USE_SDL
  if (theOSystem->p_display_screen != NULL) {
    Logger::Error
        << "Screen display requires directive __USE_SDL to be defined." << std::endl;
    Logger::Error << "Please recompile this code with flag '-D__USE_SDL'."
        << std::endl;
    Logger::Error
        << "Also ensure ALE has been compiled with USE_SDL active (see ALE makefile)."
        << std::endl;
    exit(1);
  }
#endif
}

// Get the value of a setting.
std::string ALEInterface::getString(const std::string& key) {
  assert(theSettings.get());
  return theSettings->getString(key);
}
int ALEInterface::getInt(const std::string& key) {
  assert(theSettings.get());
  return theSettings->getInt(key);
}
bool ALEInterface::getBool(const std::string& key) {
  assert(theSettings.get());
  return theSettings->getBool(key);
}
float ALEInterface::getFloat(const std::string& key) {
  assert(theSettings.get());
  return theSettings->getFloat(key);
}

// Set the value of a setting.
void ALEInterface::setString(const std::string& key, const std::string& value) {
  assert(theSettings.get());
  assert(theOSystem.get());
  theSettings->setString(key, value);
  theSettings->validate();
}
void ALEInterface::setInt(const std::string& key, const int value) {
  assert(theSettings.get());
  assert(theOSystem.get());
  theSettings->setInt(key, value);
  theSettings->validate();
}
void ALEInterface::setBool(const std::string& key, const bool value) {
  assert(theSettings.get());
  assert(theOSystem.get());
  theSettings->setBool(key, value);
  theSettings->validate();
}
void ALEInterface::setFloat(const std::string& key, const float value) {
  assert(theSettings.get());
  assert(theOSystem.get());
  theSettings->setFloat(key, value);
  theSettings->validate();
}

// Resets the game, but not the full system.
void ALEInterface::reset_game() {
  environment->reset();
}

// Indicates if the game has ended.
bool ALEInterface::game_over() const {
  return environment->isTerminal();
}

// The remaining number of lives.
int ALEInterface::lives() {
  if (!romSettings.get()) {
    throw std::runtime_error("ROM not set");
  }
  return romSettings->lives();
}

// Applies an action to the game and returns the reward. It is the
// user's responsibility to check if the game has ended and reset
// when necessary - this method will keep pressing buttons on the
// game over screen.
reward_t ALEInterface::act(Action action) {
  reward_t reward = environment->act(action, PLAYER_B_NOOP);
  if (theOSystem->p_display_screen != NULL) {
    theOSystem->p_display_screen->display_screen();
    while (theOSystem->p_display_screen->manual_control_engaged()) {
      Action user_action = theOSystem->p_display_screen->getUserAction();
      reward += environment->act(user_action, PLAYER_B_NOOP);
      theOSystem->p_display_screen->display_screen();
    }
  }
  return reward;
}

// Returns the vector of modes available for the current game.
// This should be called only after the rom is loaded.
ModeVect ALEInterface::getAvailableModes() {
  return romSettings->getAvailableModes();
}

// Sets the mode of the game.
// The mode must be an available mode.
// This should be called only after the rom is loaded.
void ALEInterface::setMode(game_mode_t m) {
  //We first need to make sure m is an available mode
  ModeVect available = romSettings->getAvailableModes();
  if(find(available.begin(), available.end(), m) != available.end()) {
    environment->setMode(m);
  } else {
    throw std::runtime_error("Invalid game mode requested");
  }
}

//Returns the vector of difficulties available for the current game.
//This should be called only after the rom is loaded.
DifficultyVect ALEInterface::getAvailableDifficulties() {
  return romSettings->getAvailableDifficulties();
}

// Sets the difficulty of the game.
// The difficulty must be an available mode.
// This should be called only after the rom is loaded.
void ALEInterface::setDifficulty(difficulty_t m) {
  DifficultyVect available = romSettings->getAvailableDifficulties();
  if(find(available.begin(), available.end(), m) != available.end()) {
    environment->setDifficulty(m);
  } else {
    throw std::runtime_error("Invalid difficulty requested");
  }
}

// Returns the vector of legal actions. This should be called only
// after the rom is loaded.
ActionVect ALEInterface::getLegalActionSet() {
  if (!romSettings.get()) {
    throw std::runtime_error("ROM not set");
  }
  return romSettings->getAllActions();
}

// Returns the vector of the minimal set of actions needed to play
// the game.
ActionVect ALEInterface::getMinimalActionSet() {
  if (!romSettings.get()) {
    throw std::runtime_error("ROM not set");
  }
  return romSettings->getMinimalActionSet();
}

// Returns the frame number since the loading of the ROM
int ALEInterface::getFrameNumber() {
  return environment->getFrameNumber();
}

// Returns the frame number since the start of the current episode
int ALEInterface::getEpisodeFrameNumber() const {
  return environment->getEpisodeFrameNumber();
}

// Returns the current game screen
const ALEScreen& ALEInterface::getScreen() {
  return environment->getScreen();
}

//This method should receive an empty vector to fill it with
//the grayscale colours
void ALEInterface::getScreenGrayscale(
    std::vector<unsigned char>& grayscale_output_buffer) {
  size_t w = environment->getScreen().width();
  size_t h = environment->getScreen().height();
  size_t screen_size = w * h;

  pixel_t *ale_screen_data = environment->getScreen().getArray();
  theOSystem->colourPalette().applyPaletteGrayscale(grayscale_output_buffer,
      ale_screen_data, screen_size);
}

//This method should receive a vector to fill it with
//the RGB colours. The first positions contain the red colours,
//followed by the green colours and then the blue colours
void ALEInterface::getScreenRGB(std::vector<unsigned char>& output_rgb_buffer) {
  size_t w = environment->getScreen().width();
  size_t h = environment->getScreen().height();
  size_t screen_size = w * h;

  pixel_t *ale_screen_data = environment->getScreen().getArray();

  theOSystem->colourPalette().applyPaletteRGB(output_rgb_buffer, ale_screen_data, screen_size);
}

// Returns the current RAM content
const ALERAM& ALEInterface::getRAM() {
  return environment->getRAM();
}

// Saves the state of the system
void ALEInterface::saveState() {
  environment->save();
}

// Loads the state of the system
void ALEInterface::loadState() {
  environment->load();
}

ALEState ALEInterface::cloneState() {
  return environment->cloneState();
}

void ALEInterface::restoreState(const ALEState& state) {
  return environment->restoreState(state);
}

ALEState ALEInterface::cloneSystemState() {
  return environment->cloneSystemState();
}

void ALEInterface::restoreSystemState(const ALEState& state) {
  return environment->restoreSystemState(state);
}

void ALEInterface::saveScreenPNG(const std::string& filename) {
  ScreenExporter exporter(theOSystem->colourPalette());
  exporter.save(environment->getScreen(), filename);
}

ScreenExporter *ALEInterface::createScreenExporter(
    const std::string &filename) const {
  return new ScreenExporter(theOSystem->colourPalette(), filename);
}
