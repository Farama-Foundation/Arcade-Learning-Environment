/* *****************************************************************************
 * ALE support for Tetris https://github.com/udibr/tetris26
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
#include "Tetris.hpp"

#include "../RomUtils.hpp"


TetrisSettings::TetrisSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* TetrisSettings::clone() const {
    
    RomSettings* rval = new TetrisSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void TetrisSettings::step(const System& system) {

    // update the reward
    reward_t score = getDecimalScore(0x71, 0x72, &system);
    if (score > m_score) {
        m_reward = score - m_score;
    } else {
        m_reward = 0;
    }
    m_score = score;

    if (!m_started) {
        m_started = true;
    }

    int byte_val = readRam(&system, 0x73);
    m_terminal = m_started && (byte_val & 0x80);
    if (m_terminal) {
        m_score = 0;
        m_started = false;
    }
}


/* is end of game */
bool TetrisSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t TetrisSettings::getReward() const {

    return m_reward; 
}


/* is an action part of the minimal set? */
bool TetrisSettings::isMinimal(const Action &a) const {

    switch (a) {
        case PLAYER_A_NOOP:
        case PLAYER_A_FIRE:
        case PLAYER_A_RIGHT:
        case PLAYER_A_LEFT:
        case PLAYER_A_DOWN:
            return true;
        default:
            return false;
    }   
}


/* reset the state of the game */
void TetrisSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_lives    = 0;
    m_terminal = false;
    m_started  = false;
}

        
/* saves the state of the rom settings */
void TetrisSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putBool(m_started);
  ser.putInt(m_lives);
}

// loads the state of the rom settings
void TetrisSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_started = ser.getBool();
  m_lives = ser.getInt();
}

