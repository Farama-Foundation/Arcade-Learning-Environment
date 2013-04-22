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
 *  RandomAgent.cpp
 *
 * The implementation of the RandomAgent class.
 **************************************************************************** */

#include "RandomAgent.hpp"
#include "random_tools.h"

RandomAgent::RandomAgent(OSystem* _osystem, RomSettings* _settings) : 
    PlayerAgent(_osystem, _settings) {
}

Action RandomAgent::act() {
  return choice(&available_actions);
}

