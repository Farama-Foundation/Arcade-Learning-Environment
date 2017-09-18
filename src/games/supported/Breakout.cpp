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
#include "Breakout.hpp"

#include "../RomUtils.hpp"


BreakoutSettings::BreakoutSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* BreakoutSettings::clone() const { 
    
    RomSettings* rval = new BreakoutSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void BreakoutSettings::step(const System& system) {

    // update the reward
    int x = readRam(&system, 77);
    int y = readRam(&system, 76);
    reward_t score = 1 * (x & 0x000F) + 10 * ((x & 0x00F0) >> 4) + 100 * (y & 0x000F);
    m_reward = score - m_score;
    m_score = score;

    // update terminal status
    int byte_val = readRam(&system, 57);
    if (!m_started && byte_val == 5) m_started = true;
    m_terminal = m_started && byte_val == 0;
    m_lives = byte_val;
}


/* is end of game */
bool BreakoutSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t BreakoutSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool BreakoutSettings::isMinimal(const Action &a) const {

    switch (a) {
        case PLAYER_A_NOOP:
        case PLAYER_A_FIRE:
        case PLAYER_A_RIGHT:
        case PLAYER_A_LEFT:
            return true;
        default:
            return false;
    }   
}


/* reset the state of the game */
void BreakoutSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_lives    = 5;
    m_terminal = false;
    m_started  = false;
}

        
/* saves the state of the rom settings */
void BreakoutSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putBool(m_started);
  ser.putInt(m_lives);
}

// loads the state of the rom settings
void BreakoutSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_started = ser.getBool();
  m_lives = ser.getInt();
}

// returns a list of mode that the game can be played in
ModeVect BreakoutSettings::getAvailableModes() {
    ModeVect modes(getNumModes());
    for (unsigned int i = 0; i < modes.size(); i++) {
        modes[i] = i * 4;
    }
    return modes;
}

// set the mode of the game
// the given mode must be one returned by the previous function
void BreakoutSettings::setMode(game_mode_t m, System &system,
                              std::unique_ptr<StellaEnvironmentWrapper> environment) {

    if(m < getNumModes() * 4 && m % 4 == 0) {
        // read the mode we are currently in
        unsigned char mode = readRam(&system, 0xB2);
        // press select until the correct mode is reached
        while (mode != m) {
            environment->pressSelect();
            mode = readRam(&system, 0xB2);
        }
        //reset the environment to apply changes.
        environment->softReset();
    }
    else {
        throw std::runtime_error("This mode doesn't currently exist for this game");
    }
 }


DifficultyVect BreakoutSettings::getAvailableDifficulties() {
    DifficultyVect diff = {0, 1};
    return diff;
}

