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
 */
#ifndef __AIRRAID_HPP__
#define __AIRRAID_HPP__

#include "../RomSettings.hpp"


/* RL wrapper for Air Raid settings */
class AirRaidSettings : public RomSettings {

    public:

        AirRaidSettings();

        // reset
        void reset(System& system, StellaEnvironment& environment);

        // is end of game
        bool isTerminal() const;

        // get the most recently observed reward
        reward_t getReward() const;

        // the rom-name
        const char* rom() const { return "air_raid"; }

        // create a new instance of the rom
        RomSettings* clone() const;

        // is an action part of the minimal set?
        bool isMinimal(const Action& a) const;

        // process the latest information from ALE
        void step(const System& system);

        // saves the state of the rom settings
        void saveState(Serializer & ser);
    
        // loads the state of the rom settings
        void loadState(Deserializer & ser);

        ActionVect getStartingActions();

        //Returns a list of mode that the game can be played in. In this game, there are 8 available modes.
        ModeVect getAvailableModes();

        //Set the mode of the game. The given mode must be one returned by the previous function. 
        void setMode(game_mode_t, System &system, StellaEnvironment& environment);

     private:

        bool m_terminal;
        reward_t m_reward;
        reward_t m_score;
};

#endif // __AIRRAID_HPP__

