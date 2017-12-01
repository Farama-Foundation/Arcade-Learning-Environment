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
#include "NameThisGame.hpp"

#include "../RomUtils.hpp"


NameThisGameSettings::NameThisGameSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* NameThisGameSettings::clone() const { 
    
    RomSettings* rval = new NameThisGameSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void NameThisGameSettings::step(const System& system) {

    // update the reward
    int score = getDecimalScore(0xC6, 0xC5, 0xC4, &system);
    int reward = score - m_score;
    m_reward = reward;
    m_score = score;

    // update terminal status
    m_lives = (readRam(&system, 0xC7) & 0x7); 
    m_terminal = (m_lives == 0);
}


/* is end of game */
bool NameThisGameSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t NameThisGameSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool NameThisGameSettings::isMinimal(const Action &a) const {

    switch (a) {
        case PLAYER_A_NOOP:
        case PLAYER_A_FIRE:
        case PLAYER_A_RIGHT:
        case PLAYER_A_LEFT:
        case PLAYER_A_RIGHTFIRE:
        case PLAYER_A_LEFTFIRE:
            return true;
        default:
            return false;
    }   
}


/* reset the state of the game */
void NameThisGameSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
    m_lives    = 3;
}
        
/* saves the state of the rom settings */
void NameThisGameSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putInt(m_lives);
}

// loads the state of the rom settings
void NameThisGameSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_lives = ser.getInt();
}

// returns a list of mode that the game can be played in
ModeVect NameThisGameSettings::getAvailableModes() {
    ModeVect modes = {0x08, 0x18, 0x28};
    return modes;
}

// set the mode of the game
// the given mode must be one returned by the previous function
void NameThisGameSettings::setMode(game_mode_t m, System &system,
                              std::unique_ptr<StellaEnvironmentWrapper> environment) {

    if(m == 0) {
      m = 0x08; // the default mode is not valid here
    }
    if(m == 0x08 || m == 0x18 || m == 0x28) {
        // read the mode we are currently in
        unsigned char mode = readRam(&system, 0xDE);
        // press select until the correct mode is reached
        while (mode != m) {
            environment->pressSelect(2);
            mode = readRam(&system, 0xDE);
        }
        //reset the environment to apply changes.
        environment->softReset();
    }
    else {
        throw std::runtime_error("This mode doesn't currently exist for this game");
    }
 }

DifficultyVect NameThisGameSettings::getAvailableDifficulties() {
    DifficultyVect diff = {0, 1};
    return diff;
}

