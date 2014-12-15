/* *****************************************************************************
 * A.L.E (Arcade Learning Environment)
 * Copyright (c) 2009-2013 by Yavar Naddaf, Joel Veness, Marc G. Bellemare,
 *  Matthew Hausknecht, and the Reinforcement Learning and Artificial Intelligence 
 *  Laboratory
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
#include "common/Defaults.hpp"
#include "common/display_screen.h"
#include "environment/stella_environment.hpp"

static const std::string Version = "0.4.4";

// Display ALE welcome message
static std::string welcomeMessage() {
  std::ostringstream oss;
  oss << "A.L.E: Arcade Learning Environment (version "
      << Version << ")\n"
      << "[Powered by Stella]\n"
      << "Use -help for help screen.";
  return oss.str();
}

static void disableBufferedIO() {
  setvbuf(stdout, NULL, _IONBF, 0);
  setvbuf(stdin, NULL, _IONBF, 0);
  cin.rdbuf()->pubsetbuf(0,0);
  cout.rdbuf()->pubsetbuf(0,0);
  cin.sync_with_stdio();
  cout.sync_with_stdio();
}

static void createOSystem(std::auto_ptr<OSystem> &theOSystem,
                          std::auto_ptr<Settings> &theSettings) {
#ifdef WIN32
  theOSystem.reset(new OSystemWin32());
  theSettings.reset(new SettingsWin32(theOSystem.get()));
#else
  theOSystem.reset(new OSystemUNIX());
  theSettings.reset(new SettingsUNIX(theOSystem.get()));
#endif

  setDefaultSettings(theOSystem->settings());

  theOSystem->settings().loadConfig();
}

static void loadSettings(const string& romfile,
                         std::auto_ptr<OSystem> &theOSystem) {
  // Load the configuration from a config file (passed on the command
  //  line), if provided
  string configFile = theOSystem->settings().getString("config", false);

  if (!configFile.empty())
    theOSystem->settings().loadConfig(configFile.c_str());

  theOSystem->settings().validate();
  theOSystem->create();

  string outputFile = theOSystem->settings().getString("output_file", false);
  if (!outputFile.empty()) {
    cerr << "Redirecting ... " << outputFile << endl;
    FILE* fp = freopen(outputFile.c_str(), "w", stdout);
  }

  // Attempt to load the ROM
  if (romfile == "" || !FilesystemNode::fileExists(romfile)) {
    std::cerr << "No ROM File specified or the ROM file was not found."
              << std::endl;
    exit(1);
  } else if (theOSystem->createConsole(romfile))  {
    std::cerr << "Running ROM file..." << std::endl;
    theOSystem->settings().setString("rom_file", romfile);
  } else {
    exit(1);
  }

  // Seed random number generator
  if (theOSystem->settings().getString("random_seed") == "time") {
    cerr << "Random Seed: Time" << endl;
    srand((unsigned)time(0));
    //srand48((unsigned)time(0));
  } else {
    int seed = theOSystem->settings().getInt("random_seed");
    assert(seed >= 0);
    cerr << "Random Seed: " << seed << endl;
    srand((unsigned)seed);
    //srand48((unsigned)seed);
  }

  theOSystem->console().setPalette("standard");
}

/**
   This class interfaces ALE with external code for controlling agents.
 */
class ALEInterface {
public:
  std::auto_ptr<OSystem> theOSystem;
  std::auto_ptr<Settings> theSettings;
  std::auto_ptr<RomSettings> romSettings;
  std::auto_ptr<StellaEnvironment> environment;
  int max_num_frames; // Maximum number of frames for each episode

  ALEInterface() {
    disableBufferedIO();
    std::cerr << welcomeMessage() << std::endl;
    createOSystem(theOSystem, theSettings);
  }
  ~ALEInterface() {}

  // Resets the OSystem/Console/Environment/etc. This is necessary
  // after changing a setting. Optionally specify a new rom to load.
  void resetSystem(string rom_file = "") {
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
      cerr << "Screen display requires directive __USE_SDL to be defined." << endl;
      cerr << "Please recompile this code with flag '-D__USE_SDL'." << endl;
      cerr << "Also ensure ALE has been compiled with USE_SDL active (see ALE makefile)." << endl;
      exit(1);
    }
#endif
  }

  // Loads and initializes a game. After this call the game should be
  // ready to play.
  void loadROM(string rom_file) {
    resetSystem(rom_file);
  }

  // Get the value of a setting.
  string get(const string& key) {
    assert(theSettings.get());
    return theSettings->getString(key);
  }

  // Set the value of a string setting.
  void set(const string& key, const string& value) {
    assert(theSettings.get());
    assert(theOSystem.get());
    theSettings->setString(key, value);
    theSettings->validate();
  }

  // Set the value of a int setting.
  void set(const string& key, const int& value) {
    assert(theSettings.get());
    assert(theOSystem.get());
    theSettings->setInt(key, value);
    theSettings->validate();
  }

  // Set the value of a bool setting.
  void set(const string& key, const bool& value) {
    assert(theSettings.get());
    assert(theOSystem.get());
    theSettings->setBool(key, value);
    theSettings->validate();
  }

  // Resets the game, but not the full system.
  void reset_game() {
    environment->reset();
  }

  // Indicates if the game has ended.
  bool game_over() {
    return (environment->isTerminal() ||
            (max_num_frames > 0 && getEpisodeFrameNumber() >= max_num_frames));
  }

  // Applies an action to the game and returns the reward. It is the
  // user's responsibility to check if the game has ended and reset
  // when necessary - this method will keep pressing buttons on the
  // game over screen.
  reward_t act(Action action) {
    reward_t reward = environment->act(action, PLAYER_B_NOOP);
    if (theOSystem->p_display_screen != NULL) {
      theOSystem->p_display_screen->display_screen(
          theOSystem->console().mediaSource());
    }
    return reward;
  }

  // Returns the vector of legal actions. This should be called only
  // after the rom is loaded.
  ActionVect getLegalActionSet() {
    return romSettings->getAllActions();
  }

  // Returns the vector of the minimal set of actions needed to play
  // the game.
  ActionVect getMinimalActionSet() {
    return romSettings->getMinimalActionSet();
  }

  // Returns the frame number since the loading of the ROM
  int getFrameNumber() {
    return environment->getFrameNumber();
  }

  // Returns the frame number since the start of the current episode
  int getEpisodeFrameNumber() {
    return environment->getEpisodeFrameNumber();
  }

  // Returns the current game screen
  const ALEScreen &getScreen() {
    return environment->getScreen();
  }

  // Returns the current RAM content
  const ALERAM &getRAM() {
    return environment->getRAM();
  }

  // Saves the state of the system
  void saveState() {
    environment->save();
  }

  // Loads the state of the system
  void loadState() {
    environment->load();
  }

  ALEState cloneState() {
    return environment->cloneState();
  }

  void restoreState(const ALEState& state) {
    return environment->restoreState(state);
  }
};

#endif
