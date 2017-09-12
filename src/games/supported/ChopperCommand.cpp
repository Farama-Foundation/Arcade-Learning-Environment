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
#include "ChopperCommand.hpp"

#include "../RomUtils.hpp"


ChopperCommandSettings::ChopperCommandSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* ChopperCommandSettings::clone() const { 
    
    RomSettings* rval = new ChopperCommandSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void ChopperCommandSettings::step(const System& system) {

    // update the reward
    reward_t score = getDecimalScore(0xEE, 0xEC, &system);
    score *= 100;
    m_reward = score - m_score;
    m_score = score;

    // update terminal status
    m_lives = readRam(&system, 0xE4) & 0xF; 
    m_terminal = (m_lives == 0);
    m_is_started = (readRam(&system, 0xC2) == 1);
}


/* is end of game */
bool ChopperCommandSettings::isTerminal() const {

    return (m_is_started && m_terminal);
};


/* get the most recently observed reward */
reward_t ChopperCommandSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool ChopperCommandSettings::isMinimal(const Action &a) const {

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
void ChopperCommandSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
    m_lives    = 3;
    m_is_started = false;
}


        
/* saves the state of the rom settings */
void ChopperCommandSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putInt(m_lives);
}

// loads the state of the rom settings
void ChopperCommandSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_lives = ser.getInt();
}


// returns a list of mode that the game can be played in
ModeVect ChopperCommandSettings::getAvailableModes() {
    ModeVect modes = {0, 2};
    return modes;
}

// set the mode of the game
// the given mode must be one returned by the previous function
void ChopperCommandSettings::setMode(game_mode_t m, System &system,
                              std::unique_ptr<StellaEnvironmentWrapper> environment) {

    if(m == 0 || m == 2) {
        // read the mode we are currently in
        unsigned char mode = readRam(&system, 0xE0);
        // press select until the correct mode is reached
        while (mode != m) {
            environment->pressSelect(2);
            mode = readRam(&system, 0xE0);
        }
        //reset the environment to apply changes.
        environment->softReset();
    }
    else {
        throw std::runtime_error("This mode doesn't currently exist for this game");
    }
 }

DifficultyVect ChopperCommandSettings::getAvailableDifficulties() {
    DifficultyVect diff = {0, 1};
    return diff;
}

