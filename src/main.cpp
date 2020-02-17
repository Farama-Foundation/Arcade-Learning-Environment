/* *****************************************************************************
 * A.L.E (Arcade Learning Environment)
 * Copyright (c) 2009-2013 by Yavar Naddaf, Joel Veness, Marc G. Bellemare and
 *   the Reinforcement Learning and Artificial Intelligence Laboratory
 * Released under the GNU General Public License; see License.txt for details.
 *
 * Based on: Stella  --  "An Atari 2600 VCS Emulator"
 * Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
 *
 * *****************************************************************************
 */

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <sstream>

#include "emucore/m6502/src/bspf/src/bspf.hxx"
#include "emucore/Console.hxx"
#include "emucore/Event.hxx"
#include "emucore/PropsSet.hxx"
#include "emucore/Settings.hxx"
#include "emucore/FSNode.hxx"
#include "emucore/OSystem.hxx"

#if (defined(WIN32) || defined(__MINGW32__))
#include "os_dependent/SettingsWin32.hxx"
#include "os_dependent/OSystemWin32.hxx"
#else
#include "os_dependent/SettingsUNIX.hxx"
#include "os_dependent/OSystemUNIX.hxx"
#endif

#include "controllers/ale_controller.hpp"
#include "controllers/fifo_controller.hpp"
#include "controllers/rlglue_controller.hpp"
#include "common/Constants.h"
#include "ale_interface.hpp"

namespace ale {
namespace {

std::unique_ptr<OSystem> theOSystem;
std::unique_ptr<Settings> theSettings;

ALEController* createController(OSystem* osystem, std::string type) {
  if (type.empty()) {
    std::cerr << "You must specify a controller type (via -game_controller)."
              << std::endl;
    exit(1);
  } else if (type == "fifo") {
    std::cerr << "Game will be controlled through FIFO pipes." << std::endl;
    return new FIFOController(osystem, false);
  } else if (type == "fifo_named") {
    std::cerr << "Game will be controlled through named FIFO pipes."
              << std::endl;
    return new FIFOController(osystem, true);
  } else if (type == "rlglue") {
    std::cerr << "Game will be controlled through RL-Glue." << std::endl;
    return new RLGlueController(osystem);
  } else {
    std::cerr << "Invalid controller type: " << type << " " << std::endl;
    exit(1);
  }
}

}  // namespace
}  // namespace ale

/* application entry point */
int main(int argc, char* argv[]) {
  // Disable buffered I/O.
  setvbuf(stdout, NULL, _IONBF, 0);
  setvbuf(stdin, NULL, _IONBF, 0);
  std::cin.rdbuf()->pubsetbuf(0, 0);
  std::cout.rdbuf()->pubsetbuf(0, 0);
  std::cin.sync_with_stdio();
  std::cout.sync_with_stdio();

  std::cerr << ale::ALEInterface::welcomeMessage() << std::endl;

  ale::ALEInterface::createOSystem(ale::theOSystem, ale::theSettings);
  // Process commandline arguments, which over-ride all possible
  // config file settings
  std::string romfile = ale::theOSystem->settings().loadCommandLine(argc, argv);
  ale::ALEInterface::loadSettings(romfile, ale::theOSystem);

  // Create the game controller
  std::string controller_type =
      ale::theOSystem->settings().getString("game_controller");
  std::unique_ptr<ale::ALEController> controller(
      ale::createController(ale::theOSystem.get(), controller_type));

  controller->run();

  // MUST delete theOSystem to avoid a segfault (theOSystem relies on Settings
  //  still being a valid construct)
  ale::theOSystem.reset(NULL);

  return 0;
}
