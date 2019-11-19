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
#ifndef __LASERGATES_HPP__
#define __LASERGATES_HPP__

#include "../RomSettings.hpp"


/* RL wrapper for Laser Gates */
class LaserGatesSettings : public RomSettings {
    public:
        LaserGatesSettings();

        // reset
        void reset();

        // is end of game
        bool isTerminal() const;

        // get the most recently observed reward
        reward_t getReward() const;

        // the rom-name
		// MD5 1fa58679d4a39052bd9db059e8cda4ad
        const char* rom() const { return "laser_gates"; }

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

        // LaserGates requires the fire action to start the game
        ActionVect getStartingActions();

        virtual int lives() { return 0; }

    private:
        bool m_terminal;
        reward_t m_reward;
        reward_t m_score;
};

#endif // __LASERGATES_HPP__

