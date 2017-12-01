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
#include "BankHeist.hpp"

#include "../RomUtils.hpp"


BankHeistSettings::BankHeistSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* BankHeistSettings::clone() const { 
    
    RomSettings* rval = new BankHeistSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void BankHeistSettings::step(const System& system) {

    // update the reward
    reward_t score = getDecimalScore(0xDA, 0xD9, 0xD8, &system);
    m_reward = score - m_score;
    m_score = score;

    // update terminal status
    int death_timer = readRam(&system, 0xCE);
    m_lives = readRam(&system, 0xD5);
        
    m_terminal = (death_timer == 0x01 && m_lives == 0x00);
}


/* is end of game */
bool BankHeistSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t BankHeistSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool BankHeistSettings::isMinimal(const Action &a) const {

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
void BankHeistSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
    m_lives    = 5;
}


        
/* saves the state of the rom settings */
void BankHeistSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putInt(m_lives);
}

// loads the state of the rom settings
void BankHeistSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_lives = ser.getInt();
}

// returns a list of mode that the game can be played in
ModeVect BankHeistSettings::getAvailableModes() {
    ModeVect modes(getNumModes());
    for (unsigned int i = 0; i < modes.size(); i++) {
        modes[i] = i * 4;
    }
    return modes;
}

// set the mode of the game
// the given mode must be one returned by the previous function
void BankHeistSettings::setMode(game_mode_t m, System &system,
                              std::unique_ptr<StellaEnvironmentWrapper> environment) {

    if(m <= 28 && m % 4 == 0) {
        // read the mode we are currently in
        unsigned char mode = readRam(&system, 0x80);
        // press select until the correct mode is reached
        while (mode != m) {
            // hold select button for 10 frames
            environment->pressSelect();
            mode = readRam(&system, 0x80);
        }
        //reset the environment to apply changes.
        environment->softReset();
    }
    else {
        throw std::runtime_error("This mode doesn't currently exist for this game");
    }
 }

DifficultyVect BankHeistSettings::getAvailableDifficulties() {
    DifficultyVect diff = {0, 1, 2, 3};
    return diff;
}
