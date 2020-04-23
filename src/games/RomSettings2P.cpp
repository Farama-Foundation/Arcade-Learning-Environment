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

#include "RomSettings2P.hpp"

#include <algorithm>
#include <unordered_set>

namespace ale {

RomSettings2P::RomSettings2P() {}

reward_t RomSettings2P::getRewardP2() const {
  return -getReward();
}
bool RomSettings2P::supportsTwoPlayers() const {
  return true;
}
bool RomSettings2P::isLegalP2(const Action& a)const {
  return true;
}
bool RomSettings2P::isMinimalP2(const Action& a) const{
  return isMinimal((Action)(a - PLAYER_B_NOOP));
}
ActionVect RomSettings2P::getMinimalActionSetP2() {
  ActionVect actions;
  for(Action a : getMinimalActionSet()){
    actions.push_back((Action)(a + PLAYER_B_NOOP));
  }
  return actions;
}

ActionVect RomSettings2P::getAllActionsP2() {
  return getAllActions();
}
ModeVect RomSettings2P::oppositeModes(int num_modes) {
  ModeVect single_p_ms = getAvailableModes();
  std::unordered_set<int> single_p_modes(single_p_ms.begin(),single_p_ms.end());
  ModeVect other_modes;
  for(int mode = 0; mode < num_modes; mode++){
    if(!single_p_modes.count(mode)){
      other_modes.push_back(mode);
    }
  }
  return other_modes;
}
// void RomSettings::setMode(game_mode_t m, System&, std::unique_ptr<StellaEnvironmentWrapper>) {
//   // By default, 0 is the only available mode
//   if (m != 0) {
//     throw std::runtime_error("This mode is not currently available for this game");
//   }
// }

}  // namespace ale
