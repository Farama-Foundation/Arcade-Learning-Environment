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
#include <stdexcept>
#include <ctime>

using namespace std;
using namespace ale;
// Display ALE welcome message
std::string ALEInterface::welcomeMessage() {
  std::ostringstream oss;
  oss << "A.L.E: Arcade Learning Environment (version "
      << Version << ")\n"
      << "[Powered by Stella]\n"
      << "Use -help for help screen.";
  return oss.str();
}

void ALEInterface::disableBufferedIO() {
  setvbuf(stdout, NULL, _IONBF, 0);
  setvbuf(stdin, NULL, _IONBF, 0);
  cin.rdbuf()->pubsetbuf(0,0);
  cout.rdbuf()->pubsetbuf(0,0);
  cin.sync_with_stdio();
  cout.sync_with_stdio();
}

void ALEInterface::createOSystem(std::auto_ptr<OSystem> &theOSystem,
                          std::auto_ptr<Settings> &theSettings) {
#if (defined(WIN32) || defined(__MINGW32__))
  theOSystem.reset(new OSystemWin32());
  theSettings.reset(new SettingsWin32(theOSystem.get()));
#else
  theOSystem.reset(new OSystemUNIX());
  theSettings.reset(new SettingsUNIX(theOSystem.get()));
#endif

  setDefaultSettings(theOSystem->settings());

  theOSystem->settings().loadConfig();
}

void ALEInterface::loadSettings(const string& romfile,
                                std::auto_ptr<OSystem> &theOSystem) {
  // Load the configuration from a config file (passed on the command
  //  line), if provided
  string configFile = theOSystem->settings().getString("config", false);

  if (!configFile.empty())
    theOSystem->settings().loadConfig(configFile.c_str());

  theOSystem->settings().validate();
  theOSystem->create();

  // Attempt to load the ROM
  if (romfile == "" || !FilesystemNode::fileExists(romfile)) {
    Logger::Error << "No ROM File specified or the ROM file was not found."
              << std::endl;
    exit(1);
  } else if (theOSystem->createConsole(romfile))  {
    Logger::Info << "Running ROM file..." << std::endl;
    theOSystem->settings().setString("rom_file", romfile);
  } else {
    exit(1);
  }

  // Seed random number generator
  if (theOSystem->settings().getString("random_seed") == "time") {
    Logger::Info << "Random Seed: Time" << endl;
    Random::seed((uInt32)time(NULL));
  } else {
    int seed = theOSystem->settings().getInt("random_seed");
    assert(seed >= 0);
    Logger::Info << "Random Seed: " << seed << endl;
    Random::seed((uInt32)seed);
  }

  string currentDisplayFormat = theOSystem->console().getFormat();
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

ALEInterface::~ALEInterface() {}

// Loads and initializes a game. After this call the game should be
// ready to play. Resets the OSystem/Console/Environment/etc. This is
// necessary after changing a setting. Optionally specify a new rom to
// load.
void ALEInterface::loadROM(string rom_file = "") {
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
    Logger::Error << "Screen display requires directive __USE_SDL to be defined." << endl;
    Logger::Error << "Please recompile this code with flag '-D__USE_SDL'." << endl;
    Logger::Error << "Also ensure ALE has been compiled with USE_SDL active (see ALE makefile)." << endl;
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
void ALEInterface::setString(const string& key, const string& value) {
  assert(theSettings.get());
  assert(theOSystem.get());
  theSettings->setString(key, value);
  theSettings->validate();
}
void ALEInterface::setInt(const string& key, const int value) {
  assert(theSettings.get());
  assert(theOSystem.get());
  theSettings->setInt(key, value);
  theSettings->validate();
}
void ALEInterface::setBool(const string& key, const bool value) {
  assert(theSettings.get());
  assert(theOSystem.get());
  theSettings->setBool(key, value);
  theSettings->validate();
}
void ALEInterface::setFloat(const string& key, const float value) {
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
bool ALEInterface::game_over() {
  return (environment->isTerminal() ||
          (max_num_frames > 0 && getEpisodeFrameNumber() >= max_num_frames));
}

// The remaining number of lives.
const int ALEInterface::lives() {
  if (!romSettings.get()){
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

// Returns the vector of legal actions. This should be called only
// after the rom is loaded.
ActionVect ALEInterface::getLegalActionSet() {
  if (!romSettings.get()){
    throw std::runtime_error("ROM not set");
  }
  return romSettings->getAllActions();
}

// Returns the vector of the minimal set of actions needed to play
// the game.
ActionVect ALEInterface::getMinimalActionSet() {
  if (!romSettings.get()){
    throw std::runtime_error("ROM not set");
  }
  return romSettings->getMinimalActionSet();
}

// Returns the frame number since the loading of the ROM
int ALEInterface::getFrameNumber() {
  return environment->getFrameNumber();
}

// Returns the frame number since the start of the current episode
int ALEInterface::getEpisodeFrameNumber() {
  return environment->getEpisodeFrameNumber();
}

// Returns the current game screen
const ALEScreen& ALEInterface::getScreen() {
  return environment->getScreen();
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

void ALEInterface::saveScreenPNG(const string& filename) {
  
  ScreenExporter exporter(theOSystem->colourPalette());
  exporter.save(environment->getScreen(), filename);
}

ScreenExporter *ALEInterface::createScreenExporter(const std::string &filename) const {
    return new ScreenExporter(theOSystem->colourPalette(), filename); 
}
