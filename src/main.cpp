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

#include "controllers/ale_controller.hpp"
#include "controllers/fifo_controller.hpp"
#include "controllers/internal_controller.hpp"
#include "common/Constants.h"

// ALE Version number
static const std::string Version = "0.4";


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
std::string redirected_file;

void redirectOutput(string & outputFile) {
  cerr << "Redirecting ... " << outputFile << endl;

  redirected_file = outputFile;

  os = new std::ofstream(outputFile.c_str(), ios_base::out | ios_base::app);
  redirected_buffer = std::cout.rdbuf(os->rdbuf());
}

static std::auto_ptr<OSystem> theOSystem(NULL);
#ifdef WIN32
static std::auto_ptr<SettingsWin32> theSettings(NULL);
#else
static std::auto_ptr<SettingsUNIX> theSettings(NULL);
#endif

void createOSystem(int argc, char* argv[]) {
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
    if (!outputFile.empty())
      redirectOutput(outputFile);
   
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

ALEController* createController(OSystem* osystem, std::string type) {
  if (type == "fifo") {
    std::cerr << "Game will be controlled through FIFO pipes." << std::endl;
    return new FIFOController(osystem, false);
  } 
  else if (type == "fifo_named") {
    std::cerr << "Game will be controlled through named FIFO pipes." << std::endl;
    return new FIFOController(osystem, true);
  }
  else if (type == "internal") {
    std::cerr << "Game will be controlled by an internal agent." << std::endl;
    return new InternalController(osystem); 
  }
  else {
    std::cerr << "Invalid controller type: " << type << " " << std::endl;
    exit(1);
  }
}


/* application entry point */
int main(int argc, char* argv[]) {

  disableBufferedIO();

	std::cerr << welcomeMessage() << endl;
    
  createOSystem(argc, argv);

  // Create the game controller 
  std::string controller_type = theOSystem->settings().getString("game_controller");
  std::auto_ptr<ALEController> controller(createController(theOSystem.get(), controller_type));

  controller->run();

  // If we redirected stdout, restore it
  if (!redirected_file.empty()) {
    std::cout.rdbuf(redirected_buffer);
    delete os;
  }

  // MUST delete theOSystem to avoid a segfault (theOSystem relies on Settings
  //  still being a valid construct)
  theOSystem.reset(NULL);

  return 0;
}

