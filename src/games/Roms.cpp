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

#include "Roms.hpp"
#include "RomUtils.hpp"

// include the game implementations
#include "supported/Backgammon.hpp"
#include "supported/Boxing.hpp"
#include "supported/Combat.hpp"
#include "supported/DoubleDunk.hpp"
#include "supported/Entombed.hpp"
#include "supported/FishingDerby.hpp"
#include "supported/FlagCapture.hpp"
#include "supported/IceHockey.hpp"
#include "supported/Joust.hpp"
#include "supported/MarioBros.hpp"
#include "supported/MazeCraze.hpp"
#include "supported/Othello.hpp"
#include "supported/Pong.hpp"
#include "supported/SpaceInvaders.hpp"
#include "supported/SpaceWar.hpp"
#include "supported/Surround.hpp"
#include "supported/Tennis.hpp"
#include "supported/VideoCheckers.hpp"
#include "supported/WizardOfWor.hpp"

namespace ale {

/* list of supported games */
static const RomSettings* roms[] = {
    new BackgammonSettings(),
    new BoxingSettings(),
    new CombatSettings(),
    new DoubleDunkSettings(),
    new EntombedSettings(),
    new FishingDerbySettings(),
    new FlagCaptureSettings(),
    new IceHockeySettings(),
    new JoustSettings(),
    new MarioBrosSettings(),
    new MazeCrazeSettings(),
    new OthelloSettings(),
    new PongSettings(),
    new SpaceInvadersSettings(),
    new SpaceWarSettings(),
    new SurroundSettings(),
    new TennisSettings(),
    new VideoCheckersSettings(),
    new WizardOfWorSettings(),
};

/* looks for the RL wrapper corresponding to a particular rom title */
RomSettings* buildRomRLWrapper(const std::string& rom) {
  size_t slash_ind = rom.find_last_of("/\\");
  std::string rom_str = rom.substr(slash_ind + 1);
  size_t dot_idx = rom_str.find_first_of(".");
  rom_str = rom_str.substr(0, dot_idx);
  std::transform(rom_str.begin(), rom_str.end(), rom_str.begin(), ::tolower);

  for (size_t i = 0; i < sizeof(roms) / sizeof(roms[0]); i++) {
    if (rom_str == roms[i]->rom())
      return roms[i]->clone();
  }

  return NULL;
}

}  // namespace ale
