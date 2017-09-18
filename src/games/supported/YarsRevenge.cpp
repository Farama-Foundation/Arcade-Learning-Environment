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
#include "YarsRevenge.hpp"

#include "../RomUtils.hpp"


YarsRevengeSettings::YarsRevengeSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* YarsRevengeSettings::clone() const { 
    
    RomSettings* rval = new YarsRevengeSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void YarsRevengeSettings::step(const System& system) {

    // update the reward
    int score = getDecimalScore(0xE2, 0xE1, 0xE0, &system);
    int reward = score - m_score;
    m_reward = reward;
    m_score = score;

    // update terminal status
    int lives_byte = readRam(&system, 0x9E) >> 4;
    m_terminal = (lives_byte == 0);

    m_lives = lives_byte;
}


/* is end of game */
bool YarsRevengeSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t YarsRevengeSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool YarsRevengeSettings::isMinimal(const Action &a) const {

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
void YarsRevengeSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
    m_lives    = 4;
}
        
/* saves the state of the rom settings */
void YarsRevengeSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putInt(m_lives);
}

// loads the state of the rom settings
void YarsRevengeSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_lives = ser.getInt();
}

ActionVect YarsRevengeSettings::getStartingActions() {
    ActionVect startingActions;
    startingActions.push_back(PLAYER_A_FIRE);
    return startingActions;
}

// returns a list of mode that the game can be played in
ModeVect YarsRevengeSettings::getAvailableModes() {
    ModeVect modes = {0, 0x20, 0x40, 0x60};
    return modes;
}

// set the mode of the game
// the given mode must be one returned by the previous function
void YarsRevengeSettings::setMode(game_mode_t m, System &system,
                              std::unique_ptr<StellaEnvironmentWrapper> environment) {

    if(m == 0 || m == 0x20 || m == 0x40 || m == 0x60) {
        // enter in mode selection screen
        environment->pressSelect(2);
        // read the mode we are currently in
        unsigned char mode = readRam(&system, 0xE3);
        // press select until the correct mode is reached
        while (mode != m) {
            environment->pressSelect();
            mode = readRam(&system, 0xE3);
        }
        //reset the environment to apply changes.
        environment->softReset();
    }
    else {
        throw std::runtime_error("This mode doesn't currently exist for this game");
    }
 }

DifficultyVect YarsRevengeSettings::getAvailableDifficulties() {
    DifficultyVect diff = {0, 1};
    return diff;
}