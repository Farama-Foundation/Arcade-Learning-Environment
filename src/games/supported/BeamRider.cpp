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
#include "BeamRider.hpp"

#include "../RomUtils.hpp"


BeamRiderSettings::BeamRiderSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* BeamRiderSettings::clone() const { 
    
    RomSettings* rval = new BeamRiderSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void BeamRiderSettings::step(const System& system) {

    // update the reward
    int score = getDecimalScore(9, 10, 11, &system);
    m_reward = score - m_score;
    m_score = score;
    int new_lives = readRam(&system, 0x85) + 1;

    // Decrease lives *after* the death animation; this is necessary as the lives counter
    // blinks during death
    if (new_lives == m_lives - 1) {
        if (readRam(&system, 0x8C) == 0x01)
            m_lives = new_lives;
    }
    else
        m_lives = new_lives;

    // update terminal status
    int byte_val = readRam(&system, 5);
    m_terminal = byte_val == 255;
    byte_val = byte_val & 15;
    m_terminal = m_terminal || byte_val < 0;
}


/* is end of game */
bool BeamRiderSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t BeamRiderSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool BeamRiderSettings::isMinimal(const Action &a) const {

    switch (a) {
        case PLAYER_A_NOOP:
        case PLAYER_A_FIRE:
        case PLAYER_A_UP:
        case PLAYER_A_RIGHT:
        case PLAYER_A_LEFT:
        case PLAYER_A_UPRIGHT:
        case PLAYER_A_UPLEFT:
        case PLAYER_A_RIGHTFIRE:
        case PLAYER_A_LEFTFIRE:
            return true;
        default:
            return false;
    }   
}


/* reset the state of the game */
void BeamRiderSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
    m_lives    = 3;
}

        
/* saves the state of the rom settings */
void BeamRiderSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putInt(m_lives);
}

// loads the state of the rom settings
void BeamRiderSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_lives = ser.getInt();
}

ActionVect BeamRiderSettings::getStartingActions() {
    ActionVect startingActions;
    startingActions.push_back(PLAYER_A_RIGHT);
    return startingActions;
}

DifficultyVect BeamRiderSettings::getAvailableDifficulties() {
    DifficultyVect diff = {0, 1};
    return diff;
}

