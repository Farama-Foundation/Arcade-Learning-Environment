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

RomSettings::RomSettings() {}

bool RomSettings::isLegal(const Action& a) const {
  return true;
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
};

void RomSettings::setMode(game_mode_t m, System&, std::unique_ptr<StellaEnvironmentWrapper>) {
  //By default, 0 is the only available mode
  if (m != 0) {
    throw std::runtime_error("This mode is not currently available for this game");
  }
}

DifficultyVect RomSettings::getAvailableDifficulties() {
  return DifficultyVect(1, 0);
};
