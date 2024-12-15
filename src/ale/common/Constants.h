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
 *  common_constants.h
 *
 *  Defines a set of constants used by various parts of the player agent code
 *
 **************************************************************************** */

#ifndef __CONSTANTS_H__
#define __CONSTANTS_H__

#include <string>
#include <vector>

namespace ale {

// Define actions
enum Action {
  NOOP                   = 0,
  FIRE                   = 1,
  UP                     = 2,
  RIGHT                  = 3,
  LEFT                   = 4,
  DOWN                   = 5,
  UPRIGHT                = 6,
  UPLEFT                 = 7,
  DOWNRIGHT              = 8,
  DOWNLEFT               = 9,
  UPFIRE                 = 10,
  RIGHTFIRE              = 11,
  LEFTFIRE               = 12,
  DOWNFIRE               = 13,
  UPRIGHTFIRE            = 14,
  UPLEFTFIRE             = 15,
  DOWNRIGHTFIRE          = 16,
  DOWNLEFTFIRE           = 17,
  RESET                  = 18, // MGB: Use SYSTEM_RESET to reset the environment.
  UNDEFINED              = 19,
  RANDOM                 = 20,
  SAVE_STATE             = 21,
  LOAD_STATE             = 22,
  SYSTEM_RESET           = 23,
  LAST_ACTION_INDEX      = 24
};

#define ACTION_MAX (18)

std::string action_to_string(Action a);

//  Define datatypes
typedef std::vector<Action> ActionVect;

// mode type
typedef unsigned game_mode_t;
typedef std::vector<game_mode_t> ModeVect;

// difficulty type
typedef unsigned difficulty_t;
typedef std::vector<difficulty_t> DifficultyVect;

// reward type for RL interface
typedef int reward_t;

// Other constant values
#define RAM_LENGTH 128

}  // namespace ale

#endif  // __CONSTANTS_H__
