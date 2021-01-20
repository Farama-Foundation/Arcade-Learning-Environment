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

#include "games/RomSettings.hpp"

#include <algorithm>

namespace ale {

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
}

void RomSettings::setMode(game_mode_t m, System&, std::unique_ptr<StellaEnvironmentWrapper>) {
  // By default, 0 is the only available mode
  if (m != 0) {
    throw std::runtime_error("This mode is not currently available for this game");
  }
}

game_mode_t RomSettings::getDefaultMode() {
  // By default, return the first available mode, or 0 if none are listed
  ModeVect available_modes = getAvailableModes();
  if (available_modes.empty()) {
    available_modes = get2PlayerModes();
    if (available_modes.empty()) {
      available_modes = get4PlayerModes();
      if (available_modes.empty()) {
        return 0;
      }
    }
  }
  return available_modes[0];
}

DifficultyVect RomSettings::getAvailableDifficulties() {
  return DifficultyVect(1, 0);
}

bool RomSettings::isModeSupported(game_mode_t m) {
  auto modes = getAvailableModes();
  return std::find(modes.begin(), modes.end(), m) != modes.end();
}

reward_t RomSettings::getRewardP2() const {
  return 0;
}

int RomSettings::livesP2() {
  throw std::logic_error("2 player method used for 1 player game");
}

ModeVect RomSettings::get2PlayerModes() {
  // return no modes for 2 players by default
  return ModeVect{};
}

reward_t RomSettings::getRewardP3() const {
  throw std::logic_error("4 player method used game that does not support 4 players");
}

reward_t RomSettings::getRewardP4() const {
  throw std::logic_error("4 player method used game that does not support 4 players");
}

int RomSettings::livesP3() {
  throw std::logic_error("4 player method used game that does not support 4 players");
}

int RomSettings::livesP4() {
  throw std::logic_error("4 player method used game that does not support 4 players");
}

ModeVect RomSettings::get3PlayerModes() {
  return ModeVect{};
}

ModeVect RomSettings::get4PlayerModes() {
  return ModeVect{};
}

}  // namespace ale
