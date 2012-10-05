/* *****************************************************************************
 * A.L.E (Arcade Learning Environment)
 * Copyright (c) 2009-2012 by Yavar Naddaf, Joel Veness, Marc G. Bellemare and 
 *   the Reinforcement Learning and Artificial Intelligence Laboratory
 * Released under the GNU General Public License; see License.txt for details. 
 *
 * Based on: Stella  --  "An Atari 2600 VCS Emulator"
 * Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
 *
 *
 * *****************************************************************************
 *  SDLKeyboardAgent.hpp
 *
 * The implementation of a keyboard-controllable agent. 
 **************************************************************************** */

#ifndef __SDL_KEYBOARD_AGENT_HPP__
#define __SDL_KEYBOARD_AGENT_HPP__

#include "../common/Constants.h"
#include "PlayerAgent.hpp"
#include "../emucore/OSystem.hxx"

class SDLKeyboardAgent : public PlayerAgent {
    public:
        SDLKeyboardAgent(OSystem * _osystem, RomSettings * _settings);
		
	protected:
        /* *********************************************************************
            Returns the best action from the set of possible actions
         ******************************************************************** */
        virtual Action act();
};

#endif // __SDL_KEYBOARD_AGENT_HPP__
