/* *****************************************************************************
 * A.L.E (Atari 2600 Learning Environment)
 * Copyright (c) 2009-2013 by Yavar Naddaf
 * Released under the GNU General Public License; see License.txt for details.
 *
 * Based on: Stella  --  "An Atari 2600 VCS Emulator"
 * Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
 *
 * *****************************************************************************
 *  Constants.cpp
 *
 *  Defines a set of constants used by various parts of the player agent code
 *
 **************************************************************************** */

#include "ale/common/Constants.h"

#include <cassert>
#include <string>

namespace ale {

std::string action_to_string(Action a) {
  static std::string tmp_action_to_string[] = {
      "NOOP",
      "FIRE",
      "UP",
      "RIGHT",
      "LEFT",
      "DOWN",
      "UPRIGHT",
      "UPLEFT",
      "DOWNRIGHT",
      "DOWNLEFT",
      "UPFIRE",
      "RIGHTFIRE",
      "LEFTFIRE",
      "DOWNFIRE",
      "UPRIGHTFIRE",
      "UPLEFTFIRE",
      "DOWNRIGHTFIRE",
      "DOWNLEFTFIRE",
      "RESET",        // 18
      "UNDEFINED",    // 19
      "RANDOM",       // 20
      "__invalid__",  // 21
      "__invalid__",  // 22
      "__invalid__",  // 23
      "__invalid__",  // 24
  };
  assert(a >= 0 && a <= 24);
  return tmp_action_to_string[a];
}

}  // namespace ale
