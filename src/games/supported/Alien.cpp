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
#include "Alien.hpp"

#include "../RomUtils.hpp"


AlienSettings::AlienSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* AlienSettings::clone() const { 
    
    RomSettings* rval = new AlienSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void AlienSettings::step(const System& system) {

    // update the reward
    int b1 = getDigit(system, 0x8B);
    int b2 = getDigit(system, 0x89); 
    int b3 = getDigit(system, 0x87); 
    int b4 = getDigit(system, 0x85); 
    int b5 = getDigit(system, 0x83); 
    reward_t score = b1 + b2 * 10 + b3 * 100 + b4 * 1000 + b5 * 10000;
    score *= 10;
    m_reward = score - m_score;
    m_score = score;

    // update terminal status
    int byte = readRam(&system, 0xC0);
    byte = byte & 15;
    m_terminal = byte <= 0;
    m_lives = byte;
}


/* is end of game */
bool AlienSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t AlienSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool AlienSettings::isMinimal(const Action &a) const {

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
void AlienSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
    m_lives    = 3;
}


/* special code to read digits for Alien */
int AlienSettings::getDigit(const System& system, int address) const {
  
    int byte = readRam(&system, address);
    return byte == 0x80 ? 0 : byte >> 3;
}
        
/* saves the state of the rom settings */
void AlienSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putInt(m_lives);
}

// loads the state of the rom settings
void AlienSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_lives = ser.getInt();
}

// returns a list of mode that the game can be played in
ModeVect AlienSettings::getAvailableModes() {
    ModeVect modes(getNumModes());
    for (unsigned int i = 0; i < modes.size(); i++) {
        modes[i] = i;
    }
    return modes;
}

// set the mode of the game
// the given mode must be one returned by the previous function
void AlienSettings::setMode(game_mode_t m, System &system,
                              std::unique_ptr<StellaEnvironmentWrapper> environment) {

    if(m < getNumModes()) {
        // read the mode we are currently in
        unsigned char mode = readRam(&system, 0x81);
        // press select until the correct mode is reached
        while (mode != m) {
            environment->pressSelect();
            mode = readRam(&system, 0x81);
        }
        //update the number of lives
        int byte = readRam(&system, 0xC0);
        byte = byte & 15;
        m_lives = byte;
        //reset the environment to apply changes.
        environment->softReset();
    }
    else {
        throw std::runtime_error("This mode doesn't currently exist for this game");
    }
 }

DifficultyVect AlienSettings::getAvailableDifficulties() {
    DifficultyVect diff = {0, 1, 2, 3};
    return diff;
}
