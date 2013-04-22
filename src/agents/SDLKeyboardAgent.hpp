/* *****************************************************************************
 * A.L.E (Arcade Learning Environment)
 * Copyright (c) 2009-2013 by Yavar Naddaf, Joel Veness, Marc G. Bellemare and 
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
#include "../common/display_screen.h"

#ifdef __USE_SDL
class SDLKeyboardAgent : public PlayerAgent, SDLEventHandler {
  public:
    SDLKeyboardAgent(OSystem * _osystem, RomSettings * _settings);
		
    /* *********************************************************************
       Handles SDL events such as allowing the player to control the screen
       using the keyboard.
       * ****************************************************************** */
    bool handleSDLEvent(const SDL_Event& event);
    void display_screen(IntMatrix& screen_matrix, int screen_width, int screen_height) {};
    void usage();

	protected:
    /* *********************************************************************
       Captures the users keypresses. Used for manually controlling the game.
       * ****************************************************************** */
    Action waitForKeypress();

    /* *********************************************************************
        Returns the best action from the set of possible actions
     ******************************************************************** */
    virtual Action act();

  private:
    bool returnToPause;               // This is used after manual control is released to set paused state
    bool manual_control;              // Is the game being controlled manually?
};
#else
/** A dummy class for when SDL is not used. */
class SDLKeyboardAgent : public PlayerAgent {
  public:
    SDLKeyboardAgent(OSystem * _osystem, RomSettings * _settings):PlayerAgent(_osystem, _settings) {}
    
    virtual Action act() { return (Action)0; }
};
#endif

#endif // __SDL_KEYBOARD_AGENT_HPP__
