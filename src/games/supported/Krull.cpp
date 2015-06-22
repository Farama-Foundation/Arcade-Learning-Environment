/* *****************************************************************************
 * The lines 61, 114, 123 and 131 are based on Xitari's code, from Google Inc.
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
#include "Krull.hpp"

#include "../RomUtils.hpp"


KrullSettings::KrullSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* KrullSettings::clone() const { 
    
    RomSettings* rval = new KrullSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void KrullSettings::step(const System& system) {

    // update the reward
    int score = getDecimalScore(0x9E, 0x9D, 0x9C, &system); 
    int reward = score - m_score;
    m_reward = reward;
    m_score = score;

    // update terminal status
    int lives = readRam(&system, 0x9F);
    int byte1 = readRam(&system, 0xA2);
    int byte2 = readRam(&system, 0x80);
    m_terminal = lives == 0 && byte1 == 0x03 && byte2 == 0x80;
    m_lives = (lives & 0x7) + 1;
}


/* is end of game */
bool KrullSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t KrullSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool KrullSettings::isMinimal(const Action &a) const {

    switch (a) {
        case PLAYER_A_NOOP:
        case PLAYER_A_FIRE:
        case PLAYER_A_UP:
        case PLAYER_A_RIGHT:
        case PLAYER_A_LEFT:
        case PLAYER_A_DOWN:
        case PLAYER_A_UPRIGHT:
        case PLAYER_A_UPLEFT:
        case PLAYER_A_DOWNRIGHT:
        case PLAYER_A_DOWNLEFT:
        case PLAYER_A_UPFIRE:
        case PLAYER_A_RIGHTFIRE:
        case PLAYER_A_LEFTFIRE:
        case PLAYER_A_DOWNFIRE:
        case PLAYER_A_UPRIGHTFIRE:
        case PLAYER_A_UPLEFTFIRE:
        case PLAYER_A_DOWNRIGHTFIRE:
        case PLAYER_A_DOWNLEFTFIRE:
            return true;
        default:
            return false;
    }   
}


/* reset the state of the game */
void KrullSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
    m_lives    = 3;
}

        
/* saves the state of the rom settings */
void KrullSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putInt(m_lives);
}

// loads the state of the rom settings
void KrullSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_lives = ser.getInt();
}

