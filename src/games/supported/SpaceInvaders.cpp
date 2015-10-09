/* *****************************************************************************
 * The lines 54 - 59, 100, 109 and 117 are based on Xitari's code, from Google Inc.
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
#include "SpaceInvaders.hpp"

#include "../RomUtils.hpp"

ActionVect SpaceInvadersSettings::actions;

SpaceInvadersSettings::SpaceInvadersSettings() {
    reset();
}


/* create a new instance of the rom */
RomSettings* SpaceInvadersSettings::clone() const { 
    
    RomSettings* rval = new SpaceInvadersSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void SpaceInvadersSettings::step(const System& system) {

    // update the reward
    int score = getDecimalScore(0xE8, 0xE6, &system);
    // reward cannot get negative in this game. When it does, it means that the score has looped 
    // (overflow)
    m_reward = score - m_score;
    if(m_reward < 0) {
        // 10000 is the highest possible score
        const int maximumScore = 10000;
        m_reward = (maximumScore - m_score) + score; 
    }
    m_score = score;
    m_lives = readRam(&system, 0xC9);

    // update terminal status
    // If bit 0x80 is on, then game is over 
    int some_byte = readRam(&system, 0x98); 
    m_terminal = (some_byte & 0x80) || m_lives == 0;
}


/* is end of game */
bool SpaceInvadersSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t SpaceInvadersSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool SpaceInvadersSettings::isMinimal(const Action &a) const {

    switch (a) {
        case PLAYER_A_NOOP:
        case PLAYER_A_LEFT:
        case PLAYER_A_RIGHT:
        case PLAYER_A_FIRE:
        case PLAYER_A_LEFTFIRE:
        case PLAYER_A_RIGHTFIRE:
            return true;
        default:
            return false;
    }   
}


/* reset the state of the game */
void SpaceInvadersSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
    m_lives    = 3; 
}

        
/* saves the state of the rom settings */
void SpaceInvadersSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putInt(m_lives);
}

// loads the state of the rom settings
void SpaceInvadersSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_lives = ser.getInt();
}

