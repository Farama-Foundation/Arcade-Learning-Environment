/* *****************************************************************************
 * The method lives() is based on Xitari's code, from Google Inc.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License version 2
 * as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 * *****************************************************************************
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
#ifndef __DOUBLEDUNK_HPP__
#define __DOUBLEDUNK_HPP__

#include "../RomSettings.hpp"


/* RL wrapper for Double Dunk settings */
class DoubleDunkSettings : public RomSettings {

    public:

        DoubleDunkSettings();

        // reset
        void reset();

        // is end of game
        bool isTerminal() const;

        // get the most recently observed reward
        reward_t getReward() const;

        // the rom-name
        const char* rom() const { return "double_dunk"; }

        // get the available number of modes
        unsigned int getNumModes() const { return 16; }

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

        virtual int lives() { return 0; }

        // returns a list of mode that the game can be played in
        // in this game, there are 16 available modes
        ModeVect getAvailableModes();

        // set the mode of the game
        // the given mode must be one returned by the previous function
        void setMode(game_mode_t, System &system,
                     std::unique_ptr<StellaEnvironmentWrapper> environment); 


    private:

        bool m_terminal;
        reward_t m_reward;
        reward_t m_score;

        // this game has a menu that allows to define various yes/no options
        // this function goes to the next option in the menu
        void goDown(System &system,
            std::unique_ptr<StellaEnvironmentWrapper> &environment);

        // once we are at the proper option in the menu,
        // if we want to enable it all we have to do is to go right
        void activateOption(System &system, unsigned int bitOfInterest,
            std::unique_ptr<StellaEnvironmentWrapper> &environment);

        // once we are at the proper optio in the menu,
        // if we want to disable it all we have to do is to go left
        void deactivateOption(System &system, unsigned int bitOfInterest,
            std::unique_ptr<StellaEnvironmentWrapper> &environment);
};

#endif // __DOUBLEDUNK_HPP__

