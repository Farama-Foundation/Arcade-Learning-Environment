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
 *  PlayerAgent.hpp
 *
 * The implementation of the PlayerAgent abstract class
 **************************************************************************** */
#ifndef __PLAYER_AGENT_HPP__
#define __PLAYER_AGENT_HPP__

#include "../emucore/OSystem.hxx"
#include "../common/Constants.h"
#include "../games/RomSettings.hpp"

class PlayerAgent { 
public:
    PlayerAgent(OSystem* _osystem, RomSettings* _settings);
    virtual ~PlayerAgent();
        
    /** This methods handles the basic agent functionality: bookkeeping
     *  the number of episodes, frames, etc... It calls the method 
     *  act(), which will provide it with an action. act() which should 
     *  be overriden by subclasses.
     */
    virtual Action agent_step();
        
    /* *********************************************************************
       This method is called when the game ends. The superclass 
       implementation takes care of counting the number of episodes, and 
       saving the reward history. Any other update should be taken care of
       in the derived classes
       ******************************************************************** */
    virtual void episode_end(void);

    /* *********************************************************************
       This method is called at the beginning of each game
       ******************************************************************** */
    virtual Action episode_start(void);

    /* *********************************************************************
       Generates an instance of one of the PlayerAgent subclasses, based on
       "player_agent" value in the settings. 
       Note 1: If you add a new player_agent subclass, you need to also 
       add it here
       Note 2: The caller is resposible for deleting the returned pointer
       ******************************************************************** */
    static PlayerAgent* generate_agent_instance(OSystem* _osystem, 
                                                RomSettings * _settings);
   
    /** Returns true when the agent is done playing the game. */
    virtual bool has_terminated();


protected:
    virtual Action act() = 0;
    /** Completes this run */
    void end_game();
        
    void record_action(Action a);

protected:
    OSystem* p_osystem;               // Pointer to the stella's OSystem 
    RomSettings* p_rom_settings;      // An instance of the GameSettings class
		
    int i_max_num_episodes;           // We exit the program after this 
    int i_max_num_frames;	      // We exit the program after this 
    int i_max_num_frames_per_episode; // Episode ends after this many frames

    int frame_number;
    int episode_frame_number;
    int episode_number;
        
    ActionVect & available_actions;

    bool record_trajectory;
    ActionVect trajectory;

    bool m_has_terminated;
};



#endif // __PLAYER_AGENT_HPP__
