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
 *  ALEState.hpp
 *
 *  A class that stores a copy of the current ALE state. We use one to keep track
 *   of paddle resistance and in search trees.
 *  Currently this class plays a double role. It is used to store the state of 
 *   Stella via serialization. It is also used to store relevant state variables 
 *   that are not part of Stella: paddle resistance, frame number.
 *  The general contract of this class is that save() stores a snapshot of the 
 *   current emulator state, as well as paddle information. load() restores the 
 *   state. The catch is that the extra variables such as paddle resistance are NOT
 *   obtained from Stella. Consider two states, s1 and s2, assuming s1 contains
 *   relevant state data. The following code:
 * 
 *  s1.load()
 *   (affect paddles)
 *  s2.save()
 *  s2.load()
 *   
 *  will use the paddle state from s2 but the emulator state from s1. If you need
 *   to copy the extra variables from s1 to s2, consider using the copy constructor.
 *  
 **************************************************************************** */

#ifndef __ALESTATE_HPP__ 
#define __ALESTATE_HPP__

#include "../emucore/OSystem.hxx"
#include "../emucore/Event.hxx"
#include <string>
#include "../games/RomSettings.hpp"

#define PADDLE_DELTA 23000
// MGB Values taken from Paddles.cxx (Stella 3.3) - 1400000 * [5,235] / 255
#define PADDLE_MIN 27450 
#define PADDLE_MAX 1290196 
#define PADDLE_DEFAULT_VALUE (((PADDLE_MAX - PADDLE_MIN) / 2) + PADDLE_MIN)

class ALEState {
  protected:
    OSystem * m_osystem;
    RomSettings * m_settings;

    bool m_use_starting_actions;

    string serialized;
    string s_cartridge_md5;

  protected:
    static int left_paddle_curr_x;   // Current x value for the left-paddle
    static int right_paddle_curr_x;  // Current x value for the right-paddle

    // For debugging purposes, we store the frame number
    int frame_number;

    bool uses_paddles;

  public:
    /** When this constructor is called a copy of the given state is made. */
    ALEState(ALEState & _state);
    /** This constructor creates a default ALEState; it is not saved. */
    ALEState(OSystem * system);

    void setSettings(RomSettings *);

    /** Resets ALE (emulator and ROM settings) to the state described by
      * this object. */
    void load();

    /** Sets ALE (emulator and ROM settings) to the state described by
      * this object. */
    void save();

    /** Resets the system to its start state. numResetSteps 'RESET' actions are taken after the
      *  start. */
    void reset(int numResetSteps = 1);

    /** This applies the given actions to the emulator, but does not simulate. It assumes that 
      *  load() has been previously called, or that we want to move the current system state 
      *  forward. This method modifies this ALEState's paddle information, but NOT the actual 
      *  saved state. To generate the successor of ALEState s in a search tree, one would do:
      * 
      *  ALEState* successor = new ALEState(s);
      *  successor->load();
      *  successor->apply_action(act, act2);
      *  successor->simulate();
      *  successor->save();
      * 
      *  The apply_action method provides flexibility, e.g. allows us to move the state forward
      *  without requiring that we load/save at every step. Be warned that in this case, 
      *  the semantics of ALEState become murkier - only the paddle state is truly stored.
      * 
      */
    void apply_action(Action player_a_action, Action player_b_action);

    /** This calls the emulator for one step and updates the RL data in RomSettings.
      *  It is assumed that apply_action() has been called prior to this to set the
      *  action to be taken.
      */
    void simulate();

    RomSettings * getSettings() { return m_settings; }
    OSystem * getSystem() { return m_osystem; }

    int getFrameNumber() const { return frame_number; }
    /** Used to increment the frame counter when ALE directly simulates the 
      *  system, rather via than the simulate() method. See internal_agent.
      */
    void incrementFrameNumber() { frame_number++; }

    /** Returns true if the two states contain the same saved information */
    bool equals(ALEState &state);

  protected:
    /** Methods for updating the Event object (which contains joystick/paddle information) */
    void apply_action_paddles(Event * event_obj, int player_a_action, int player_b_action);
    void apply_action_joysticks(Event * event_obj, int player_a_action, int player_b_action);
    void reset_keys(Event * event_obj);

    /* *********************************************************************
     *  Calculates the Paddle resistance, based on the given x val
     * ********************************************************************/
    int calc_paddle_resistance(int x_val);

    /* *********************************************************************
     *  Updates the positions of the paddles, and sets an event for
     *  updating the corresponding paddle's resistance
     * ********************************************************************/
    void update_paddles_positions(int delta_left, int delta_right);

    void default_paddles();

    void set_paddles(int left, int right);
};

#endif // __ALESTATE_HPP__


