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
 *  stella_environment.hpp
 *
 *  A class that wraps around the Stella core to provide users with a typical
 *  reinforcement learning environment interface.
 *  
 **************************************************************************** */

#ifndef __STELLA_ENVIRONMENT_HPP__ 
#define __STELLA_ENVIRONMENT_HPP__

#include "ale_state.hpp"
#include "ale_screen.hpp"
#include "ale_ram.hpp"
#include "phosphor_blend.hpp"
#include "../emucore/OSystem.hxx"
#include "../emucore/Event.hxx"
#include "../games/RomSettings.hpp"

#include <stack>

// This defines the number of "random" environments
#define NUM_RANDOM_ENVIRONMENTS (500)

class StellaEnvironment {
  public:
    StellaEnvironment(OSystem * system, RomSettings * settings);

    /** Resets the system to its start state. */
    void reset();

    /** Save/restore the environment state onto the stack. */
    void save();
    void load();

    /** Returns a copy of the current emulator state. */
    ALEState cloneState();
    /** Restores a previously saved copy of the state. */
    void restoreState(const ALEState&);

    /** Applies the given actions (e.g. updating paddle positions when the paddle is used)
      *  and performs one simulation step in Stella. Returns the resultant reward. When 
      *  frame skip is set to > 1, up the corresponding number of simulation steps are performed.
      *  Note that the post-act() frame number might not correspond to the pre-act() frame
      *  number plus the frame skip.
      */
    reward_t act(Action player_a_action, Action player_b_action);

    /** Returns true once we reach a terminal state */
    bool isTerminal();

    /** Accessor methods for the environment state. */
    void setState(const ALEState & state);
    const ALEState &getState() const;

    /** Returns the current screen after processing (e.g. colour averaging) */
    const ALEScreen &getScreen() const { return m_screen; }
    const ALERAM &getRAM() const { return m_ram; }

    int getFrameNumber() const { return m_frame_number; }
    int getEpisodeFrameNumber() const { return m_episode_frame_number; }

  private:
    /** This applies an action exactly one time step. Helper function to act(). */
    reward_t oneStepAct(Action player_a_action, Action player_b_action);

    /** Actually emulates the emulator for a given number of steps. */
    void emulate(Action player_a_action, Action player_b_action, size_t num_steps = 1);

    /** Drops illegal actions, such as the fire button in skiing. Note that this is different
      *   from the minimal set of actions. */
    void noopIllegalActions(Action& player_a_action, Action& player_b_action);

    /** Processes the current emulator screen and saves it in m_screen */
    void processScreen();
    /** Processes the emulator RAM and saves it in m_ram */
    void processRAM();

  private:
    OSystem * m_osystem;
    RomSettings * m_settings;
    PhosphorBlend m_phosphor_blend; // For performing phosphor colour averaging, if so desired
    std::string m_cartridge_md5; // Necessary for saving and loading emulator state

    std::stack<ALEState> m_saved_states; // States are saved on a stack
    
    ALEState m_state; // Current environment state
    int m_frame_number; // How many frames since the start
    int m_episode_frame_number; // How many frames since the beginning of this episode
    ALEScreen m_screen; // The current ALE screen (possibly colour-averaged)
    ALERAM m_ram; // The current ALE RAM

    bool m_use_paddles;  // Whether this game uses paddles
    
    /** Parameters loaded from Settings. */
    bool m_use_starting_actions; // Whether we run a set of starting actions after reset 
    int m_num_reset_steps; // Number of RESET frames per reset
    bool m_colour_averaging; // Whether to average frames
    bool m_stochastic_start; // Whether to "draw" the environment from a random distribution
    int m_max_num_frames_per_episode; // Maxmimum number of frames per episode 
    size_t m_frame_skip; // How many frames to emulate per act()

    bool m_backward_compatible_save; // Enable the save/load mechanism from ALE 0.2 (no stack)
};

#endif // __STELLA_ENVIRONMENT_HPP__
