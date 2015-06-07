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
 *  rlglue_controller.hpp
 *
 *  The RLGlueController class implements an RL-Glue interface. It's based off
 *   the FIFOController, but the internals are different.
 **************************************************************************** */

#ifndef __RLGLUE_CONTROLLER_HPP__
#define __RLGLUE_CONTROLLER_HPP__

#include "ale_controller.hpp"
#include "../common/Constants.h"

#ifdef __USE_RLGLUE
// We namespace the whole RL-Glue business to avoid name conflicts
#include <rlglue/Environment_common.h>
#include <rlglue/network/RL_network.h>

class RLGlueController : public ALEController {
  public:
    RLGlueController(OSystem* osystem);
    virtual ~RLGlueController();

    virtual void run();

  private:
    /** Initializes the RL-Glue business */
    void initRLGlue(); 
    /** Closes the RL-Glue connection */ 
    void endRLGlue();
    /** Loops through the RL-Glue statement machine until termination. */ 
    void rlGlueLoop();

    bool isDone();

    /** RL-Glue interface methods. These methods' output mechanism is to fill the RL buffer 
      *  with data. */
     
    /** Initializes the environment; sends a 'task spec' */
    void envInit();
    /** Sends the first observation out -- beginning an episode */
    void envStart();
    /** Reads in an action, returns the next observation-reward-terminal tuple */ 
    void envStep();
    /** Performs some RL-Glue related cleanup */
    void envCleanup();
    /** RL-Glue custom messages */
    void envMessage();

    /** RL-Glue helper methods. */

    reward_observation_terminal_t constructRewardObservationTerminal(reward_t reward);
    /** Filters the action received by RL-Glue */
    void filterActions(Action& player_a_action, Action& player_b_action);

  private:
    /** RL-Glue variables */
    rlBuffer m_buffer;
    int m_connection;
    observation_t m_observation;
    action_t m_rlglue_action;

    int m_max_num_frames; // Maximum number of total frames before we stop

    ActionVect available_actions;
    bool m_send_rgb;
};
#else
class RLGlueController : public ALEController {
  public:
    RLGlueController(OSystem* osystem);
    virtual ~RLGlueController() {}

    /** This prints an error message and terminate. */
    virtual void run();
};
#endif // __USE_RLGLUE


#endif // __RLGLUE_CONTROLLER_HPP__
