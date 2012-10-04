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
 *  game_controller.cpp
 *
 *  The implementation of the GameController class, which is the superclass for
 *  other controller classes. A controller class sits between Stella, and 
 *  whatever mechanism we are using to control Stella, i.e. FIFO pipes and
 *  external code, or internal player_agent classes.
 **************************************************************************** */

#include "game_controller.h"
#include "Roms.hpp"


/* *********************************************************************
    Constructor
 ******************************************************************** */
GameController::GameController(OSystem* _osystem): state(_osystem) {
    p_osystem = _osystem;
    p_global_event_obj = p_osystem->event();
    p_console = &(p_osystem->console());
    MediaSource& mediasrc = p_console->mediaSource();
    p_emulator_system = &(p_console->system());
    i_screen_width  = mediasrc.width();
    i_screen_height = mediasrc.height();
    b_send_screen_matrix    = true;
    b_send_console_ram      = true;
    i_skip_frames_num       = 0;
    i_skip_frames_counter   = 0;
    e_previous_a_action     = PLAYER_A_NOOP;
    e_previous_b_action     = PLAYER_B_NOOP;
    
    // load the RL wrapper for the ROM
    string rom_file = p_osystem->settings().getString("rom_file");
    m_rom_settings = buildRomRLWrapper(rom_file);
    if (m_rom_settings == NULL) {
        std::cerr << "Unsupported ROM file" << std::endl;
        exit(1);
    }

    state.setSettings(m_rom_settings);
    // MGB
    p_num_system_reset_steps = atoi(_osystem->settings().getString("system_reset_steps").c_str());

    systemReset();

}
        
/* *********************************************************************
    Destructor
 ******************************************************************** */
GameController::~GameController() {
    if (m_rom_settings) {
        delete m_rom_settings;
        m_rom_settings = NULL;
    }
}

void GameController::saveState() {
  state.save();
}

void GameController::loadState() {
  state.load();
}

/* reset the interface and emulator */
void GameController::systemReset() {
  state.reset(p_num_system_reset_steps);
}

/* ***************************************************************************
 *  Function read_ram
 *  Reads a byte from console ram
 *  
 *  The code is mainly based on RamDebug.cxx
 * ***************************************************************************/
int GameController::read_ram(int offset) {
    offset &= 0x7f; // there are only 128 bytes
    return p_emulator_system->peek(offset + 0x80);
}

