/* *****************************************************************************
 * The lines 100, 110 and 118 are based on Xitari's code, from Google Inc.
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
#include "Atlantis.hpp"

#include "../RomUtils.hpp"


AtlantisSettings::AtlantisSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* AtlantisSettings::clone() const { 
    
    RomSettings* rval = new AtlantisSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void AtlantisSettings::step(const System& system) {

    // update the reward. Score in Atlantis is a bit funky: when you "roll" the score, it increments
    // the *lowest* digit. E.g., 999900 -> 000001. 
    reward_t score = getDecimalScore(0xA2, 0xA3, 0xA1, &system); 
    score *= 100;
    m_reward = score - m_score;
    reward_t old_score = m_score;
    m_score = score;

    // update terminal status
    m_lives = readRam(&system, 0xF1);
    m_terminal = (m_lives == 0xFF);

    //when the game terminates, some garbage gets written on a1, screwing up the score computation
    //since it is not possible to score on the very last frame, we can safely set the reward to 0.
    if(m_terminal){
        m_reward = 0;
        m_score = old_score;
    }
    
  
    // MGB: Also d4-da contain the 'building' status; 05 indicates a destroyed
    //      building, 00 a live building
}


/* is end of game */
bool AtlantisSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t AtlantisSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool AtlantisSettings::isMinimal(const Action &a) const {

    switch (a) {
        case PLAYER_A_NOOP:
        case PLAYER_A_FIRE:
        case PLAYER_A_RIGHTFIRE:
        case PLAYER_A_LEFTFIRE:
            return true;
        default:
            return false;
    }   
}


/* reset the state of the game */
void AtlantisSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
    m_lives    = 6;
}


        
/* saves the state of the rom settings */
void AtlantisSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putInt(m_lives);
}

// loads the state of the rom settings
void AtlantisSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_lives = ser.getInt();
}

