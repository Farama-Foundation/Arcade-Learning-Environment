/* *****************************************************************************
 * The lines 91, 99 and 107 are based on Xitari's code, from Google Inc.
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
#include "Asterix.hpp"

#include "../RomUtils.hpp"


AsterixSettings::AsterixSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* AsterixSettings::clone() const { 
    
    RomSettings* rval = new AsterixSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void AsterixSettings::step(const System& system) {

    // update the reward
    reward_t score = getDecimalScore(0xE0, 0xDF, 0xDE, &system); 
    m_reward = score - m_score;
    m_score = score;

    // update terminal status
    m_lives = readRam(&system, 0xD3) & 0xF;
    int death_counter = readRam(&system, 0xC7);
    
    // we cannot wait for lives to be set to 0, because the agent has the
    // option of the restarting the game on the very last frame (when lives==1
    // and death_counter == 0x01) by holding 'fire'
    m_terminal = (death_counter == 0x01 && m_lives == 1);
}


/* is end of game */
bool AsterixSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t AsterixSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool AsterixSettings::isMinimal(const Action &a) const {

    switch (a) {
        case PLAYER_A_NOOP:
        case PLAYER_A_UP:
        case PLAYER_A_RIGHT:
        case PLAYER_A_LEFT:
        case PLAYER_A_DOWN:
        case PLAYER_A_UPRIGHT:
        case PLAYER_A_UPLEFT:
        case PLAYER_A_DOWNRIGHT:
        case PLAYER_A_DOWNLEFT:
            return true;
        default:
            return false;
    }   
}


/* reset the state of the game */
void AsterixSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
    m_lives    = 3;
}
        
/* saves the state of the rom settings */
void AsterixSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putInt(m_lives);
}

// loads the state of the rom settings
void AsterixSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_lives = ser.getInt();
}

ActionVect AsterixSettings::getStartingActions() {
    ActionVect startingActions;
    startingActions.push_back(PLAYER_A_FIRE);
    return startingActions;
}

