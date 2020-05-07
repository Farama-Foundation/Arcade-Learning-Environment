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

#include "RomSettings4P.hpp"

#include <algorithm>
#include <unordered_set>

namespace ale {

RomSettings4P::RomSettings4P() {}

int RomSettings4P::livesP3() {
  return 0;
}
int RomSettings4P::livesP4() {
  return 0;
}

// bool RomSettings4P::in_two_player_list(int mode){
//   auto modes = get2PlayerModes();
//   return std::find(modes.begin(), modes.end(), m) != modes.end();
// }

}  // namespace ale
