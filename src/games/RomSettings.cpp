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
 *
 * RomSettings.cpp
 *
 * The interface to describe games as RL environments. It provides terminal and
 *  reward information.
 * *****************************************************************************
 */

#include "RomSettings.hpp"

#include <algorithm>
#include <cassert>

namespace ale {

RomSettings::RomSettings() {}

bool RomSettings::isLegal(const Action& a) const {
  return true;
}
int RomSettings::lives() {
  return 0;
}

ActionVect RomSettings::getMinimalActionSet() {
  ActionVect actions;
  for (int a = 0; a < PLAYER_B_NOOP; a++) {
    if (isMinimal((Action)a) && isLegal((Action)a)) {
      actions.push_back((Action)a);
    }
  }
  return actions;
}

ActionVect RomSettings::getAllActions() {
  ActionVect actions;
  for (int a = 0; a < PLAYER_B_NOOP; a++) {
    if (isLegal((Action)a)) {
      actions.push_back((Action)a);
    }
  }
  return actions;
}

ActionVect RomSettings::getStartingActions() {
  return ActionVect();
}

ModeVect RomSettings::getAvailableModes() {
  return ModeVect(1, 0);
}

void RomSettings::setMode(game_mode_t m, System&, std::unique_ptr<StellaEnvironmentWrapper>) {
  // By default, 0 is the only available mode
  if (m != 0) {
    throw std::runtime_error("This mode is not currently available for this game");
  }
}

DifficultyVect RomSettings::getAvailableDifficulties() {
  return DifficultyVect(1, 0);
}

bool RomSettings::isModeSupported(game_mode_t m, int players) {
  auto modes = (players == 1) ? getAvailableModes() : ((players == 2) ? get2PlayerModes() : get4PlayerModes()) ;
  return std::find(modes.begin(), modes.end(), m) != modes.end();
}
#define xstr(a) str(a)
#define str(a) #a
#define two_player_fail() throw std::runtime_error("2 player method used for 1 player game in line: " xstr(__LINE__));
#define four_player_fail() throw std::runtime_error("4 player method used for 1 or two player game in line: " xstr(__LINE__));

reward_t RomSettings::getRewardP2() const { return -1; }
int RomSettings::livesP2() { two_player_fail(); }
ModeVect RomSettings::get2PlayerModes() { return ModeVect{}; }

reward_t RomSettings::getRewardP3() const { four_player_fail(); }
reward_t RomSettings::getRewardP4() const { four_player_fail(); };
int RomSettings::livesP3() { four_player_fail(); };
int RomSettings::livesP4() { four_player_fail(); };
ModeVect RomSettings::get3PlayerModes() { return ModeVect{}; };
ModeVect RomSettings::get4PlayerModes() { return ModeVect{}; };
}  // namespace ale
