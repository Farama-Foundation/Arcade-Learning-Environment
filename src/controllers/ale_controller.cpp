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
 *  controller.hpp 
 *
 *  Superclass defining a variety of controllers -- main loops interfacing with
 *   an agent in a particular way. This superclass handles work common to all 
 *   controllers, e.g. loading ROM settings and constructing the environment
 *   wrapper.
 **************************************************************************** */

#include "ale_controller.hpp"
#include "../games/Roms.hpp"

#include "../common/display_screen.h"
#include "../common/Log.hpp"

ALEController::ALEController(OSystem* osystem):
  m_osystem(osystem),
  m_settings(buildRomRLWrapper(m_osystem->settings().getString("rom_file"))),
  m_environment(m_osystem, m_settings.get()) {

  if (m_settings.get() == NULL) {
    ale::Logger::Warning << "Unsupported ROM file: " << std::endl;
    exit(1);
  }
  else {
    m_environment.reset();
  }
}

void ALEController::display() {
  // Display the screen if applicable
  DisplayScreen* display = m_osystem->p_display_screen;
  if (display) {
    display->display_screen();
    while (display->manual_control_engaged()) {
      Action user_action = display->getUserAction();
      applyActions(user_action, PLAYER_B_NOOP);
      display->display_screen();
    }
  }
}

reward_t ALEController::applyActions(Action player_a, Action player_b) {
  reward_t sum_rewards = 0;
  // Perform different operations based on the first player's action 
  switch (player_a) {
    case LOAD_STATE: // Load system state
      // Note - this does not reset the game screen; so that the subsequent screen
      //  is incorrect (in fact, two screens, due to colour averaging)
      m_environment.load();
      break;
    case SAVE_STATE: // Save system state
      m_environment.save();
      break;
    case SYSTEM_RESET:
      m_environment.reset();
      break;
    default:
      // Pass action to emulator!
      sum_rewards = m_environment.act(player_a, player_b);
      break;
  }
  return sum_rewards;
}

