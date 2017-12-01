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
#include "Gravitar.hpp"

#include "../RomUtils.hpp"


GravitarSettings::GravitarSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* GravitarSettings::clone() const { 
    
    RomSettings* rval = new GravitarSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void GravitarSettings::step(const System& system) {

    // update the reward
    int score = getDecimalScore(9, 8, 7, &system);
    int reward = score - m_score;
    m_reward = reward;
    m_score = score;

    // Byte 0x81 contains information about the current screen
    int screen_byte = readRam(&system, 0x81);

    // update terminal status
    m_terminal = screen_byte == 0x01;

    // On the starting screen, we set our lives total to 6; otherwise read it from data 
    m_lives = screen_byte == 0x0? 6 : (readRam(&system, 0x84) + 1);
}


/* is end of game */
bool GravitarSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t GravitarSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool GravitarSettings::isMinimal(const Action &a) const {

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
void GravitarSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
    m_lives    = 6;
}

        
/* saves the state of the rom settings */
void GravitarSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putInt(m_lives);
}

// loads the state of the rom settings
void GravitarSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_lives = ser.getInt();
}

ActionVect GravitarSettings::getStartingActions() {
    ActionVect startingActions;
    for (int i=0; i<16; i++) 
        startingActions.push_back(PLAYER_A_FIRE);
    return startingActions;
}

// returns a list of mode that the game can be played in
ModeVect GravitarSettings::getAvailableModes() {
    ModeVect modes(getNumModes());
    for (unsigned int i = 0; i < modes.size(); i++) {
        modes[i] = i;
    }
    return modes;
}

// set the mode of the game
// the given mode must be one returned by the previous function
void GravitarSettings::setMode(game_mode_t m, System &system,
                              std::unique_ptr<StellaEnvironmentWrapper> environment) {

    if(m < getNumModes()) {
        // read the mode we are currently in
        unsigned char mode = readRam(&system, 0x80);
        // press select until the correct mode is reached
        while (mode != m) {
            // hold select button for 10 frames
            environment->pressSelect(10);
            mode = readRam(&system, 0x80);
        }

        //update the number of lives
        switch(m){
            case 0:
            case 2:
                m_lives = 6;
                break;
            case 1:
                m_lives = 15;
                break;
            case 3:
                m_lives = 100;
            case 4:
                m_lives = 25;
        }
        //reset the environment to apply changes.
        environment->softReset();
    }
    else {
        throw std::runtime_error("This mode doesn't currently exist for this game");
    }
 }
