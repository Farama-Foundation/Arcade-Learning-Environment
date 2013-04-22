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
 *  SingleActionAgent.hpp
 *
 * The implementation of the SingleActionAgent class. With probability epsilon
 *  it takes a random action; otherwise it takes the action specified by the 
 *  configuration under 'agent_action'.
 **************************************************************************** */

#ifndef __SINGLE_ACTION_AGENT_HPP__
#define __SINGLE_ACTION_AGENT_HPP__

#include "../common/Constants.h"
#include "PlayerAgent.hpp"
#include "../emucore/OSystem.hxx"

class SingleActionAgent : public PlayerAgent {
    public:
        SingleActionAgent(OSystem * _osystem, RomSettings * _settings);
		
	protected:
        /* *********************************************************************
            Returns the best action from the set of possible actions
         ******************************************************************** */
        virtual Action act();

  protected:
      double epsilon;
      Action agent_action;
};

#endif // __SINGLE_ACTION_AGENT_HPP__
 

