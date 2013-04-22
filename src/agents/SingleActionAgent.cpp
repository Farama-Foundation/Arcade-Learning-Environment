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
 *  SingleActionAgent.cpp
 *
 * The implementation of the SingleActionAgent class. With probability epsilon
 *  it takes a random action; otherwise it takes the action specified by the 
 *  configuration under 'agent_action'.
 **************************************************************************** */

#include "SingleActionAgent.hpp"
#include "random_tools.h"

SingleActionAgent::SingleActionAgent(OSystem* _osystem, RomSettings* _settings) : PlayerAgent(_osystem, _settings) {
    epsilon = _osystem->settings().getFloat("agent_epsilon", true);
    agent_action = (Action)_osystem->settings().getInt("agent_action", true);
}

Action SingleActionAgent::act() {
  if (drand48() < epsilon)
    return choice(&available_actions);
  else
    return agent_action; 
}

