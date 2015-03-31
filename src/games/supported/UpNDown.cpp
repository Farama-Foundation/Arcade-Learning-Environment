/* *****************************************************************************
 * The lines 61, 102, 110 and 118 are based on Xitari's code, from Google Inc.
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
#include "UpNDown.hpp"

#include "../RomUtils.hpp"


UpNDownSettings::UpNDownSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* UpNDownSettings::clone() const { 
    
    RomSettings* rval = new UpNDownSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void UpNDownSettings::step(const System& system) {

    // update the reward
    int score = getDecimalScore(0x82, 0x81, 0x80, &system);
    int reward = score - m_score;
    m_reward = reward;
    m_score = score;

    // update terminal status
    int lives_byte = readRam(&system, 0x86) & 0xF;
    int death_timer = readRam(&system, 0x94);
    m_terminal = death_timer > 0x40 && lives_byte == 0;

    m_lives = lives_byte + 1;
}


/* is end of game */
bool UpNDownSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t UpNDownSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool UpNDownSettings::isMinimal(const Action &a) const {

    switch (a) {
        case PLAYER_A_NOOP:
        case PLAYER_A_FIRE:
        case PLAYER_A_UP:
        case PLAYER_A_DOWN:
        case PLAYER_A_UPFIRE:
        case PLAYER_A_DOWNFIRE:
            return true;
        default:
            return false;
    }   
}


/* reset the state of the game */
void UpNDownSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
    m_lives    = 5;
}
        
/* saves the state of the rom settings */
void UpNDownSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putInt(m_lives);
}

// loads the state of the rom settings
void UpNDownSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_lives = ser.getInt();
}

ActionVect UpNDownSettings::getStartingActions() {
    ActionVect startingActions;
    startingActions.push_back(PLAYER_A_FIRE);
    return startingActions;
}


