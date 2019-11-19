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
#ifndef __MRDO_HPP__
#define __MRDO_HPP__

#include "../RomSettings.hpp"


/* RL wrapper for MrDo */
class MrDoSettings : public RomSettings {

    public:

        MrDoSettings();

        // reset
        void reset();

        // is end of game
        bool isTerminal() const;

        // get the most recently observed reward
        reward_t getReward() const;

        // the rom-name
        const char* rom() const { return "mr_do"; }

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

        // Mr. Do requires the fire action to start the game
        ActionVect getStartingActions();

        virtual int lives() { return isTerminal() ? 0 : m_lives; }

    private:

        bool m_terminal;
        reward_t m_reward;
        reward_t m_score;
        int m_lives;
};

#endif // __MRDO_HPP__

