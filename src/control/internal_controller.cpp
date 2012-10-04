/* *****************************************************************************
 * A.L.E (Arcade Learning Environment)
 * Copyright (c) 2009-2012 by Yavar Naddaf, Joel Veness, Marc G. Bellemare and 
 *   the Reinforcement Learning and Artificial Intelligence Laboratory
 * Released under the GNU General Public License; see License.txt for details. 
 *
 * Based on: Stella  --  "An Atari 2600 VCS Emulator"
 * Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
 *
 * *****************************************************************************
 *  internal_controller.cpp
 *
 *  The internal controller acts as an environment wrapper for a C++ agent.
 **************************************************************************** */

#include <sstream>
#include "internal_controller.h"
#include "game_controller.h"
#include "RomSettings.hpp"
#include "Roms.hpp"
#include "PlayerAgent.hpp"
#include "Settings.hxx"

#include "random_tools.h"
#include "misc_tools.h"

/* *********************************************************************
    Constructor
 ******************************************************************** */
InternalController::InternalController(OSystem* _osystem)  : 
    GameController(_osystem), first_step(true) {
    p_player_agent_right = NULL; // Change this if you want a right player
                                 // Note that current agents only produce action 
                                 // for the left player, and need to be fixed
    p_player_agent_left = PlayerAgent::generate_agent_instance(p_osystem, 
      m_rom_settings);
}
    
/* *********************************************************************
    Destructor 
 ******************************************************************** */
InternalController::~InternalController() {
    if (p_player_agent_right) {
        delete p_player_agent_right;
        p_player_agent_right = NULL;
    }
    if (p_player_agent_left) {
        delete p_player_agent_left;
        p_player_agent_left = NULL;
    }
}

// @dbg
int episodeNumber = 1;
reward_t episodeScore = 0;

/* *********************************************************************
    This is called on every iteration of the main loop. It is responsible 
    passing the framebuffer and the RAM content to whatever AI module we 
    are using, and applying the returned actions.
 * ****************************************************************** */
void InternalController::update() {
	Action player_a_action, player_b_action;

  // Bookkeeping
  state.incrementFrameNumber();
  m_rom_settings->step(*p_emulator_system);
  bool isTerminal = m_rom_settings->isTerminal();
  episodeScore += m_rom_settings->getReward();

  // @dbg
  if (m_rom_settings->getReward() != 0) {
    cerr << "Reward " << m_rom_settings->getReward() << " Score " << episodeScore << endl;
  }

  if (first_step) {
    if (p_player_agent_left) player_a_action = p_player_agent_left->episode_start();
    else player_a_action = PLAYER_A_NOOP; 
    if (p_player_agent_right) player_b_action = p_player_agent_right->episode_start();
    else player_b_action = PLAYER_B_NOOP;
    
    first_step = false;
  }
  else {
    if (p_player_agent_left) {
      player_a_action = p_player_agent_left->agent_step();
    } else {
      player_a_action = PLAYER_A_NOOP;
    }
    if (p_player_agent_right) {
      player_b_action = p_player_agent_right->agent_step();
    } else {
      player_b_action = PLAYER_B_NOOP;
    }
  }

  bool resetRequested = false;
  
  if (player_a_action == RESET) {
    std::cerr << "Reset requested... terminating episode" << endl;
    resetRequested = true;
  }
  else if (has_terminated()) {
    p_osystem->quit();
    return;
  }

  // If we were in a non-terminal state, take the next action
  if (!resetRequested && !isTerminal) {
    e_previous_a_action = player_a_action;
    e_previous_b_action = player_b_action;

    state.apply_action(player_a_action, player_b_action);
  }
  // Otherwise, reset the system now
  else {
    systemReset();
    p_osystem->skipEmulation();

    if (p_player_agent_left) p_player_agent_left->episode_end();
    if (p_player_agent_right) p_player_agent_right->episode_end();

    first_step = true;

    // @dbg - move to PlayerAgent?
    std::cout << "EPISODE " << episodeNumber << " " << episodeScore << std::endl;
    episodeNumber++;
    episodeScore = 0;
  }
}

bool InternalController::has_terminated() {
  if (p_player_agent_left == NULL && p_player_agent_right == NULL)
    return true;
  else if (p_player_agent_left != NULL && p_player_agent_left->has_terminated())
    return true;
  else if (p_player_agent_right != NULL && 
      p_player_agent_right->has_terminated())
    return true;
  else
    return false;
}
