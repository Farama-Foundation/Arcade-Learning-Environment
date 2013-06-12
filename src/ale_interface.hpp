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

// @todo 
static const std::string Version = "0.4";

// Display welcome message 
static std::string welcomeMessage() {
    // ALE welcome message
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

static void createOSystem(int argc, char* argv[],
                          std::auto_ptr<OSystem> &theOSystem,
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

    // process commandline arguments, which over-ride all possible config file settings
    string romfile = theOSystem->settings().loadCommandLine(argc, argv);

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
        freopen(outputFile.c_str(), "w", stdout);
    }

    // attempt to load the ROM
    if (argc == 1 || romfile == "" || !FilesystemNode::fileExists(romfile)) {
		
        std::cerr << "No ROM File specified or the ROM file was not found." << std::endl;
        exit(1); 

    } else if (theOSystem->createConsole(romfile))  {
        
        std::cerr << "Running ROM file..." << std::endl;
        theOSystem->settings().setString("rom_file", romfile);

    } else {
        exit(1);
    }

    // seed random number generator
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
class ALEInterface
{
public:
    std::auto_ptr<OSystem> theOSystem;
    std::auto_ptr<Settings> theSettings;
    std::auto_ptr<RomSettings> settings;
    std::auto_ptr<StellaEnvironment> environment;

protected:
    reward_t episode_score; // Score accumulated throughout the course of an episode
    bool display_active;    // Should the screen be displayed or not
    int max_num_frames;     // Maximum number of frames for each episode

public:
    ALEInterface(bool display_screen=false): episode_score(0), display_active(display_screen) {
#ifndef __USE_SDL
        if (display_active) {
            cout << "Screen display requires directive __USE_SDL to be defined." << endl;
            cout << "Please recompile this code with flag '-D__USE_SDL'." << endl;
            cout << "Also ensure ALE has been compiled with USE_SDL active (see ALE makefile)." << endl;
            exit(0);
        }
#endif
        disableBufferedIO();
        std::cerr << welcomeMessage() << endl;
    }

    ~ALEInterface() {}

    // Loads and initializes a game. After this call the game should be ready to play.
    void loadROM(string rom_file) {
        int argc = 6;
        char** argv = new char*[argc];
        for (int i=0; i<=argc; i++) {
            argv[i] = new char[200];
        }
        strcpy(argv[0],"./ale");
        strcpy(argv[1],"-player_agent");
        strcpy(argv[2],"random_agent");
        strcpy(argv[3],"-display_screen");
        if (display_active) strcpy(argv[4],"true");
        else                strcpy(argv[4],"false");
        strcpy(argv[5],rom_file.c_str());  

        createOSystem(argc, argv, theOSystem, theSettings);
        settings.reset(buildRomRLWrapper(rom_file));
        environment.reset(new StellaEnvironment(theOSystem.get(), settings.get()));
        max_num_frames = theOSystem->settings().getInt("max_num_frames_per_episode");
        reset_game();
    }

    // Resets the game
    void reset_game() {
        environment->reset();
    }

    // Indicates if the game has ended
    bool game_over() {
        return (environment->isTerminal() || (max_num_frames > 0 && getEpisodeFrameNumber() >= max_num_frames));
    }

    // Applies an action to the game and returns the reward. It is the user's responsibility
    // to check if the game has ended and reset when necessary - this method will keep pressing
    // buttons on the game over screen.
    reward_t act(Action action) {
        environment->act(action, PLAYER_B_NOOP);
        reward_t reward = settings->getReward();
        if (display_active)
            theOSystem->p_display_screen->display_screen(theOSystem->console().mediaSource());
        return reward;
    }

    // Returns the vector of legal actions. This should be called only after the rom is loaded.
    ActionVect getLegalActionSet() {
        return settings->getAllActions();
    }

    // Returns the vector of the minimal set of actions needed to play the game.
    ActionVect getMinimalActionSet() {
        return settings->getMinimalActionSet();
    }

    // Returns the frame number since the loading of the ROM
    int getFrameNumber() {
        return environment->getFrameNumber();
    }

    // Returns the frame number since the start of the current episode
    int getEpisodeFrameNumber() {
        return environment->getEpisodeFrameNumber();
    }

    // Sets the episodic frame limit
    void setMaxNumFrames(int newMax) {
        max_num_frames = newMax;
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
