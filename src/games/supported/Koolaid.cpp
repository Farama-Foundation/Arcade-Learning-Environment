/* *****************************************************************************
 * The lines 61, 63, 116, 124 and 132 are based on Xitari's code, from Google Inc.
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
#include "Koolaid.hpp"

#include "../RomUtils.hpp"


KoolaidSettings::KoolaidSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* KoolaidSettings::clone() const { 
    
    RomSettings* rval = new KoolaidSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void KoolaidSettings::step(const System& system) {

    // update the reward
    int score = getDecimalScore(0x81, 0x80, &system);
    score *= 100;
    int reward = score - m_score;
    m_reward = reward;
    m_score = score;

    m_terminal = readRam(&system, 0xD1) == 0x80;
}


/* is end of game */
bool KoolaidSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t KoolaidSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool KoolaidSettings::isMinimal(const Action &a) const {

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
void KoolaidSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
}
        
/* saves the state of the rom settings */
void KoolaidSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void KoolaidSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

