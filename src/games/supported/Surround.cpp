/* *****************************************************************************
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
#include "Surround.hpp"

#include "../RomUtils.hpp"


SurroundSettings::SurroundSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* SurroundSettings::clone() const { 
    
    RomSettings* rval = new SurroundSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void SurroundSettings::step(const System& system) {

    int oppt_score = getDecimalScore(0xF6, &system);
    int my_score = getDecimalScore(0xF7, &system);

    int delta_score = my_score - oppt_score;
    m_reward = delta_score - m_score;
    
    // "Score" really is the difference between raw scores
    m_score = delta_score;

    // Game ends when one of the players reaches 10.
    m_terminal = (oppt_score == 10 || my_score == 10);
}


/* is end of game */
bool SurroundSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t SurroundSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool SurroundSettings::isMinimal(const Action &a) const {

    switch (a) {
        case PLAYER_A_NOOP:
        case PLAYER_A_UP:
        case PLAYER_A_RIGHT:
        case PLAYER_A_LEFT:
        case PLAYER_A_DOWN:
            return true;
        default:
            return false;
    }   
}


/* reset the state of the game */
void SurroundSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
}
        
/* saves the state of the rom settings */
void SurroundSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void SurroundSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

ActionVect SurroundSettings::getStartingActions() {

    ActionVect vec;

    // We want to play vs the CPU.
    // TODO(bellemare): When modes/difficulties are brought in, this will need to be updated. 
    vec.push_back(SELECT);
    vec.push_back(RESET);

    return vec;
}
