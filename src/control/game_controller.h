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
 *  game_controller.h
 *
 *  The implementation of the GameController class, which is the superclass for
 *  other controller classes. A controller class sits between Stella, and 
 *  whatever mechanism we are using to control Stella, i.e. FIFO pipes and
 *  external code, or internal player_agent classes.
 **************************************************************************** */
#ifndef __GAME_CONTROLLER_H__
#define __GAME_CONTROLLER_H__

#include "../emucore/OSystem.hxx"
#include "../emucore/m6502/src/System.hxx"
#include "ALEState.hpp"
#include "../common/Constants.h"

#define PADDLE_DELTA 23000
// MGB Values taken from Paddles.cxx (Stella 3.3) - 1400000 * [5,235] / 255
#define PADDLE_MIN 27450 
#define PADDLE_MAX 1290196 
#define PADDLE_DEFAULT_VALUE (((PADDLE_MAX - PADDLE_MIN) / 2) + PADDLE_MIN)


class GameController {
    /* *************************************************************************
        This is the superclass for all controller classes.
    ************************************************************************* */
    public:
        /* *********************************************************************
            Constructor
         ******************************************************************** */
        GameController(OSystem* _osystem);

        /* *********************************************************************
            Destructor 
         ******************************************************************** */
        virtual ~GameController();

        /* *********************************************************************
            This is called on every iteration of the main loop. It is responsible
            passing the framebuffer and the RAM content to whatever AI module we
            are using, and applying the returned actions.
         * ****************************************************************** */
        virtual void update() = 0;

        /** Returns true if the agent is done playing the game. */ 
        virtual bool has_terminated() = 0;

        /* *********************************************************************
         *  Reads a byte from console ram
         * ********************************************************************/
        int read_ram(int offset);

        void saveState();

        void loadState();

        void systemReset();

        ALEState* getState() { return &state; };

        Action getPreviousActionA() { return e_previous_a_action; };
        Action getPreviousActionB() { return e_previous_b_action; };

    protected:
        OSystem* p_osystem;         // Pointer to Stella's OSystem object
        Event* p_global_event_obj;  // Pointer to the global event object

        int i_screen_width;         // Width of the screen
        int i_screen_height;        // Height of the screen
        Console* p_console;         // Pointer to the Console object
        System* p_emulator_system;  // Pointer to the emulator system  (used to
                                    // read the system RAM)
        bool b_send_rewards;        // When True, will send the rewards
        bool b_send_screen_matrix;  // When True, we will send the screen matrix
        bool b_send_console_ram;    // When True, we will send the console ram
        int i_skip_frames_num;      // We skip this number of frames after
                                    // sending a frame
        int i_skip_frames_counter;  // Counts how many frames we have skipped
        Action e_previous_a_action; // Action applied for player A/B during the
        Action e_previous_b_action; // last frame (used when skipping frames)

        ALEState state;
        RomSettings* m_rom_settings;		// Pointer to a GameSettings object

        // How many frames we want to send the reset action after a system reset
        int p_num_system_reset_steps;
};


#endif // __GAME_CONTROLLER_H__
