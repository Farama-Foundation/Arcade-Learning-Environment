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
#include "Kingkong.hpp"

#include "../RomUtils.hpp"

KingkongSettings::KingkongSettings() {
    reset();
}

/* create a new instance of the rom */
RomSettings* KingkongSettings::clone() const { 
    RomSettings* rval = new KingkongSettings();
    *rval = *this;
    return rval;
}

/* process the latest information from ALE */
void KingkongSettings::step(const System& system) {
    // update the reward
    int score = getDecimalScore(0x83, 0x82, &system);
    int reward = score - m_score;
    m_reward = reward;
    m_score = score;

    // update terminal status
    m_lives = readRam(&system, 0xEE);
    m_terminal = (m_lives == 0);
}

/* is end of game */
bool KingkongSettings::isTerminal() const {
    return m_terminal;
};


/* get the most recently observed reward */
reward_t KingkongSettings::getReward() const { 
    return m_reward; 
}


/* is an action part of the minimal set? */
bool KingkongSettings::isMinimal(const Action &a) const {
    switch (a) {
        case PLAYER_A_NOOP:
        case PLAYER_A_FIRE:
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
void KingkongSettings::reset() {
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
    m_lives    = 3;
}
        
/* saves the state of the rom settings */
void KingkongSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putInt(m_lives);
}

// loads the state of the rom settings
void KingkongSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_lives = ser.getInt();
}

ActionVect KingkongSettings::getStartingActions() {
    ActionVect startingActions;
    startingActions.push_back(RESET);
    return startingActions;
}


