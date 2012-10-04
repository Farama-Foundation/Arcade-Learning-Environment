/* *****************************************************************************
 * A.L.E (Arcade Learning Environment)
 * Copyright (c) 2009-2012 by Yavar Naddaf, Joel Veness, Marc G. Bellemare and 
 *   the Reinforcement Learning and Artificial Intelligence Laboratory
 * Released under the GNU General Public License; see License.txt for details. 
 *
 * Based on: Stella  --  "An Atari 2600 VCS Emulator"
 * Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
 *
 * *****************************************************************************
 */
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <memory>

#include "emucore/m6502/src/bspf/src/bspf.hxx"
#include "emucore/Console.hxx"
#include "emucore/Event.hxx"
#include "emucore/PropsSet.hxx"
#include "emucore/Settings.hxx"
#include "emucore/FSNode.hxx"
#include "emucore/OSystem.hxx"

#include "common/Defaults.hpp"

#ifdef WIN32
#   include "os_dependent/SettingsWin32.hxx"
#   include "os_dependent/OSystemWin32.hxx"
#else
#   include "os_dependent/SettingsUNIX.hxx"
#   include "os_dependent/OSystemUNIX.hxx"
#endif

#include "control/fifo_controller.h"
#include "control/internal_controller.h"
#include "common/Constants.h"

// ALE Version number
static const std::string Version = "0.3";


static std::auto_ptr<OSystem> theOSystem(NULL);
static std::auto_ptr<GameController> p_game_controller(NULL);


/* display welcome message */
static std::string welcomeMessage() {

    // ALE welcome message
    std::ostringstream oss;

    oss << "A.L.E: Arcade Learning Environment (version "
        << Version << ")\n" 
        << "[Powered by Stella]\n"
        << "Use -help for help screen.";

    return oss.str();
}


/* disable buffered IO */
static void disableBufferedIO() {

    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stdin, NULL, _IONBF, 0);
    cin.rdbuf()->pubsetbuf(0,0);
    cout.rdbuf()->pubsetbuf(0,0);
    cin.sync_with_stdio();
    cout.sync_with_stdio();
}

std::streambuf * redirected_buffer;
std::ofstream * os;

void redirectOutput(string & outputFile) {
  cerr << "Redirecting ... " << outputFile << endl;

  os = new std::ofstream(outputFile.c_str(), ios_base::out | ios_base::app);
  redirected_buffer = std::cout.rdbuf(os->rdbuf());
}

/* application entry point */
int main(int argc, char* argv[]) {

    disableBufferedIO();

	std::cerr << welcomeMessage() << endl;
    
#ifdef WIN32
    theOSystem.reset(new OSystemWin32());
    SettingsWin32 settings(theOSystem.get());
#else
    theOSystem.reset(new OSystemUNIX());
    SettingsUNIX settings(theOSystem.get());
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
    if (!outputFile.empty())
      redirectOutput(outputFile);
   
    // attempt to load the ROM
    if (argc == 1 || romfile == "" || !FilesystemNode::fileExists(romfile)) {
		
		std::cerr << "No ROM File specified or the ROM file was not found." << std::endl;
        return -1;

    } else if (theOSystem->createConsole(romfile))  {
        
		std::cerr << "Running ROM file..." << std::endl;
        theOSystem->settings().setString("rom_file", romfile);

    } else {

        return -1;
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

    // create game controller
    if (theOSystem->settings().getString("game_controller") == "fifo") {
        if (!outputFile.empty()) {
          cerr << "Cannot redirect stdout when using FIFO." << endl;
          return -1;
        }

        p_game_controller.reset(new FIFOController(theOSystem.get()));
        theOSystem->setGameController(p_game_controller.get());
        cerr << "Game will be controlled through FIFO pipes." << endl;

    } else if (theOSystem->settings().getString("game_controller") == "fifo_named") {

        p_game_controller.reset(new FIFOController(theOSystem.get(), true));
        theOSystem->setGameController(p_game_controller.get());
        std::cerr << "Game will be controlled through FIFO pipes." << std::endl;
    } else if (theOSystem->settings().getString("game_controller") == "internal") {
        p_game_controller.reset(new InternalController(theOSystem.get()));
        theOSystem->setGameController(p_game_controller.get());
        std::cerr << "Game will be controlled internally." << std::endl;
    }

    theOSystem->console().setPalette("standard");
    theOSystem->mainLoop();

    // If we redirected stdout, restore it
    if (!outputFile.empty()) {
      std::cout.rdbuf(redirected_buffer);
      delete os;
    }

    // MUST delete theOSystem to avoid a segfault (theOSystem relies on Settings
    //  still being a valid construct)
    theOSystem.reset(NULL);

    return 0;
}

