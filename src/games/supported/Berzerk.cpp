/* *****************************************************************************
 * The lines 44, 97, 107 and 115 are based on Xitari's code, from Google Inc.
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
#include "Berzerk.hpp"

#include "../RomUtils.hpp"


BerzerkSettings::BerzerkSettings() {

    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
    m_lives    = 3;
    m_mode = 1;
}


/* create a new instance of the rom */
RomSettings* BerzerkSettings::clone() const { 
    
    RomSettings* rval = new BerzerkSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void BerzerkSettings::step(const System& system) {

    // update the reward
    reward_t score = getDecimalScore(95, 94, 93, &system);
    m_reward = score - m_score;
    m_score = score;

    // update terminal status
    int livesByte = readRam(&system, 0xDA);
    
    m_terminal = (livesByte == 0xFF);
    m_lives = livesByte + 1;
}


/* is end of game */
bool BerzerkSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t BerzerkSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool BerzerkSettings::isMinimal(const Action &a) const {

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
void BerzerkSettings::reset(System& system, StellaEnvironment& environment) {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
    m_lives    = 3;
    setMode(m_mode, system, environment);
}


        
/* saves the state of the rom settings */
void BerzerkSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putInt(m_lives);
}

// loads the state of the rom settings
void BerzerkSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_lives = ser.getInt();
}

// returns a list of mode that the game can be played in.
ModeVect BerzerkSettings::getAvailableModes(){
    ModeVect modes(9);
    for(unsigned i = 0; i < 9; i++){
        modes[i] = i+1;
    }
    modes.push_back(16);
    modes.push_back(17);
    modes.push_back(18);
    return modes;
}

// set the mode of the game. The given mode must be one returned by the previous function. 
void BerzerkSettings::setMode(game_mode_t m, System &system, StellaEnvironment& environment){
    if(m >= 1 && (m <= 9 || m == 16 || m == 17 || m == 18)){
        m_mode = m;

        // we wait that the game is ready to change mode
        for(unsigned i = 0; i < 20; i++)
            environment.act(PLAYER_A_NOOP, PLAYER_B_NOOP);
        // read the mode we are currently in
        unsigned char mode = readRam(&system, 0);
        // press select until the correct mode is reached
        while(mode != m_mode){
            environment.pressSelect(2);
            mode = readRam(&system,0);
        }
        //reset the environment to apply changes.
        environment.soft_reset();
    } else{
        throw std::runtime_error("This mode doesn't currently exist for this game");
    }
}

ActionVect BerzerkSettings::getStartingActions() {
    ActionVect startingActions;
    // startingActions.push_back(PLAYER_A_NOOP);
    return startingActions;
}
