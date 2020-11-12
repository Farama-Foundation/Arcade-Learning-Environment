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

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstddef>
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

namespace ale {

// Display ALE welcome message
std::string ALEInterface::welcomeMessage() {
  std::ostringstream oss;
  oss << "A.L.E: Arcade Learning Environment (version " << ALE_VERSION_STR << ")\n"
      << "[Powered by Stella]\n"
      << "Use -help for help screen.";
  return oss.str();
}

void ALEInterface::createOSystem(std::unique_ptr<OSystem>& theOSystem,
                                 std::unique_ptr<Settings>& theSettings) {
#if (defined(WIN32) || defined(__MINGW32__))
  theOSystem.reset(new OSystemWin32());
  theSettings.reset(new SettingsWin32(theOSystem.get()));
#else
  theOSystem.reset(new OSystemUNIX());
  theSettings.reset(new SettingsUNIX(theOSystem.get()));
#endif

  theOSystem->settings().loadConfig();
}

bool ALEInterface::isSupportedRom() {
  const Properties properties = theOSystem->console().properties();
  const std::string md5 = properties.get(Cartridge_MD5);
  bool found = false;
  std::string item(romSettings->md5());
  if (!item.compare(0, md5.size(), md5)) {
    return true;
  }

  return false;
}

void ALEInterface::loadSettings(const std::string& romfile,
                                std::unique_ptr<OSystem>& theOSystem) {
  // Load the configuration from a config file (passed on the command
  //  line), if provided
  std::string configFile = theOSystem->settings().getString("config", false);

  if (!configFile.empty()) {
    theOSystem->settings().loadConfig(configFile.c_str());
  }

  theOSystem->settings().validate();
  theOSystem->create();

  // Attempt to load the ROM
  if (romfile.empty()) {
    Logger::Error << "No ROM File specified." << std::endl;
    std::exit(1);
  } else if (!FilesystemNode::fileExists(romfile)) {
    Logger::Error << "ROM file " << romfile << " not found." << std::endl;
    std::exit(1);
  } else if (theOSystem->createConsole(romfile)) {
    if (!isSupportedRom()) {
      const Properties properties = theOSystem->console().properties();
      const std::string md5 = properties.get(Cartridge_MD5);
      const std::string name = properties.get(Cartridge_Name);

      // If the md5 doesn't match our master list, warn the user.
      Logger::Warning << std::endl;
      Logger::Warning << "WARNING: Possibly unsupported ROM: mismatched MD5."
                      << std::endl;
      Logger::Warning << "Cartridge_MD5: " << md5 << std::endl;
      Logger::Warning << "Cartridge_name: " << name << std::endl;
      Logger::Warning << "Expected_MD5: " << romSettings->md5() << std::endl;
      Logger::Warning << std::endl;
    }
    Logger::Info << "Running ROM file..." << std::endl;
    theOSystem->settings().setString("rom_file", romfile);
  } else {
    Logger::Error << "Unable to create console for " << romfile << std::endl;
    std::exit(1);
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
  Logger::Info << welcomeMessage() << std::endl;
  createOSystem(theOSystem, theSettings);
}

ALEInterface::ALEInterface(bool display_screen) {
  Logger::Info << welcomeMessage() << std::endl;
  createOSystem(theOSystem, theSettings);
  this->setBool("display_screen", display_screen);
}

ALEInterface::~ALEInterface() {}

// Loads and initializes a game. After this call the game should be
// ready to play. Resets the OSystem/Console/Environment/etc. This is
// necessary after changing a setting.
void ALEInterface::loadROM(std::string rom_file) {
  assert(theOSystem.get());
  if (rom_file.empty()) {
    rom_file = theOSystem->romFile();
  }

  RomSettings* wrapper = buildRomRLWrapper(rom_file);
  if (wrapper == NULL) {
    Logger::Error << std::endl
      << "Attempt to wrap ROM " << rom_file << " failed." << std::endl;

    // We need to load the settings now to make the isSupportedRom check work;
    // the check needs the console (and its properties) to be initialized.
    loadSettings(rom_file, theOSystem);

    if (isSupportedRom()) {
      Logger::Error << "It seems the ROM is supported." << std::endl;
    } else {
      Logger::Error
        << "This ROM may not be supported." << std::endl
        << "For a list of supported ROMs see "
        << "https://github.com/mgbellemare/Arcade-Learning-Environment"
        << std::endl;
    }

    Logger::Error
      << "Perhaps the filename isn't what we expected." << std::endl
      << "ROM files should be named using snake case, "
      << "e.g., space_invaders.bin" << std::endl;

    std::exit(1);
  }

  romSettings.reset(wrapper);

  // Custom environment settings required for a specific ROM must be added
  // before the StellaEnvironment is constructed.
  romSettings->modifyEnvironmentSettings(theOSystem->settings());

  // Load all settings corresponding to the ROM file and create a new game
  // console, with attached devices, capable of emulating the ROM.
  loadSettings(rom_file, theOSystem);

  environment.reset(new StellaEnvironment(theOSystem.get(), romSettings.get()));
  max_num_frames = theOSystem->settings().getInt("max_num_frames_per_episode");

  if(romSettings->getAvailableModes().size()){
    setMode(romSettings->getAvailableModes()[0]);
  }
  else if(romSettings->get2PlayerModes().size()){
    setMode(romSettings->get2PlayerModes()[0]);
  }
  else if(romSettings->get4PlayerModes().size()){
    setMode(romSettings->get4PlayerModes()[0]);
  }
  else{
    assert(false && "game has no avaliable modes!!!");
  }

  environment->reset();


#ifndef __USE_SDL
  if (theOSystem->p_display_screen != NULL) {
    Logger::Error
        << "Screen display requires directive __USE_SDL to be defined."
        << std::endl;
    Logger::Error << "Please recompile this code with flag '-D__USE_SDL'."
                  << std::endl;
    Logger::Error << "Also ensure ALE has been compiled with USE_SDL active "
                     "(see ALE makefile)."
                  << std::endl;
    std::exit(1);
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
bool ALEInterface::game_over() const { return environment->isTerminal(); }

bool ALEInterface::supportsNumPlayers(int num_players){
  if (romSettings == nullptr) {
    throw std::runtime_error("ROM not set");
  } else {
    if(num_players == 1){
      return romSettings->getAvailableModes().size() != 0;
    }
    else if(num_players == 2){
      return romSettings->get2PlayerModes().size() != 0;
    }
    else if(num_players == 4){
      return romSettings->get4PlayerModes().size() != 0;
    }
    else{
      return false;
    }
  }
}
int ALEInterface::lives(){
  if (romSettings == nullptr) {
    throw std::runtime_error("ROM not set");
  }
  else{
    if(numPlayersActive() == 1){
      return romSettings->lives();
    }
    else{
      throw std::runtime_error("called `lives` in a multiplayer mode. Call allLives() instead.");
    }
  }
}

// The remaining number of lives.
std::vector<int> ALEInterface::allLives(){
  if (romSettings == nullptr) {
    throw std::runtime_error("ROM not set");
  } else {
    int num_players = this->numPlayersActive();
    if(num_players == 1){
      return {romSettings->lives()};
    }
    else if(num_players == 2){
      return {romSettings->lives(),romSettings->livesP2()};
    }
    else if(num_players == 4){
      return {romSettings->lives(),
        romSettings->livesP2(),
        romSettings->livesP3(),
        romSettings->livesP4()
      };
    }
    else{
      assert(false);
    }
  }
}

// Applies an action to the game and returns the reward. It is the
// user's responsibility to check if the game has ended and reset
// when necessary - this method will keep pressing buttons on the
// game over screen.
reward_t ALEInterface::act(Action action) {
  if(numPlayersActive() != 1){
    throw std::runtime_error("Single player action when in multi player mode");
  }
  reward_t reward = environment->act({action})[0];
  if (theOSystem->p_display_screen != NULL) {
    theOSystem->p_display_screen->display_screen();
    while (theOSystem->p_display_screen->manual_control_engaged()) {
      Action user_action = theOSystem->p_display_screen->getUserAction();
      reward += environment->act({user_action})[0];
      theOSystem->p_display_screen->display_screen();
    }
  }
  return reward;
}
std::vector<reward_t> ALEInterface::act(std::vector<Action> action){
  if (romSettings == nullptr) {
    throw std::runtime_error("ROM not set");
  }
  if(action.size() != numPlayersActive()){
    throw std::runtime_error("number of players active in the mode is not equal to the action size given to act");
  }

  return environment->act(action);
}


// Returns the vector of modes available for the current game.
// This should be called only after the rom is loaded.
ModeVect ALEInterface::getAvailableModes(int num_players) {
  if(num_players == 1){
    return romSettings->getAvailableModes();
  }
  else if(num_players == 2){
    return romSettings->get2PlayerModes();
  }
  else if(num_players == 3){
    return ModeVect{};
  }
  else if(num_players == 4){
    return romSettings->get4PlayerModes();
  }
  else {
    throw std::runtime_error(std::to_string(num_players)+" is not a valid number of players, only 1-4 players allowed.");
  }
}

// Sets the mode of the game.
// The mode must be an available mode.
// This should be called only after the rom is loaded.
void ALEInterface::setMode(game_mode_t m) {
  //We first need to make sure m is an available mode
  ModeVect available = romSettings->getAvailableModes();
  ModeVect available2P = romSettings->get2PlayerModes();
  ModeVect available4P = romSettings->get4PlayerModes();
  available.insert(available.end(),available2P.begin(),available2P.end());
  available.insert(available.end(),available4P.begin(),available4P.end());
  if (std::find(available.begin(), available.end(), m) != available.end()) {
    environment->setMode(m);
  } else {
    throw std::runtime_error("Invalid game mode requested");
  }
}

bool in_modes(const ModeVect & modes,game_mode_t m){
  return std::find(modes.begin(), modes.end(), m) != modes.end();
}

int ALEInterface::numPlayersActive(){
  game_mode_t m = this->getMode();
  if(in_modes(romSettings->get4PlayerModes(),m)){
    return 4;
  }
  else if (in_modes(romSettings->get2PlayerModes(),m)) {
    return 2;
  }
  else if(in_modes(romSettings->getAvailableModes(),m)){
    return 1;
  }
  else{
    assert(false && "bad mode");
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
  if (std::find(available.begin(), available.end(), m) != available.end()) {
    environment->setDifficulty(m);
  } else {
    throw std::runtime_error("Invalid difficulty requested");
  }
}

// Returns the vector of legal actions. This should be called only
// after the rom is loaded.
ActionVect ALEInterface::getLegalActionSet() {
  if (romSettings == nullptr) {
    throw std::runtime_error("ROM not set");
  } else {
    return romSettings->getAllActions();
  }
}

// Returns the vector of the minimal set of actions needed to play
// the game.
ActionVect ALEInterface::getMinimalActionSet() {
  if (romSettings == nullptr) {
    throw std::runtime_error("ROM not set");
  } else {
    return romSettings->getMinimalActionSet();
  }
}

// Returns the frame number since the loading of the ROM
int ALEInterface::getFrameNumber() { return environment->getFrameNumber(); }

// Returns the frame number since the start of the current episode
int ALEInterface::getEpisodeFrameNumber() const {
  return environment->getEpisodeFrameNumber();
}

// Returns the current game screen
const ALEScreen& ALEInterface::getScreen() { return environment->getScreen(); }

//This method should receive an empty vector to fill it with
//the grayscale colours
void ALEInterface::getScreenGrayscale(
    std::vector<unsigned char>& grayscale_output_buffer) {
  size_t w = environment->getScreen().width();
  size_t h = environment->getScreen().height();
  size_t screen_size = w * h;

  pixel_t* ale_screen_data = environment->getScreen().getArray();
  theOSystem->colourPalette().applyPaletteGrayscale(
      grayscale_output_buffer, ale_screen_data, screen_size);
}

//This method should receive a vector to fill it with
//the RGB colours. The first positions contain the red colours,
//followed by the green colours and then the blue colours
void ALEInterface::getScreenRGB(std::vector<unsigned char>& output_rgb_buffer) {
  size_t w = environment->getScreen().width();
  size_t h = environment->getScreen().height();
  size_t screen_size = w * h;

  pixel_t* ale_screen_data = environment->getScreen().getArray();

  theOSystem->colourPalette().applyPaletteRGB(output_rgb_buffer,
                                              ale_screen_data, screen_size);
}

// Returns the current RAM content
const ALERAM& ALEInterface::getRAM() { return environment->getRAM(); }

void ALEInterface::setRAM(size_t memory_index, byte_t value) {
  return environment->setRAM(memory_index, value);
}

// Saves the state of the system
void ALEInterface::saveState() { environment->save(); }

// Loads the state of the system
void ALEInterface::loadState() { environment->load(); }

ALEState ALEInterface::cloneState() { return environment->cloneState(); }

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

ScreenExporter*
ALEInterface::createScreenExporter(const std::string& filename) const {
  return new ScreenExporter(theOSystem->colourPalette(), filename);
}

}  // namespace ale
