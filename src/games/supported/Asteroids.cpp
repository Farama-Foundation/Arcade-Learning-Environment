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
#include "Asteroids.hpp"

#include "../RomUtils.hpp"


AsteroidsSettings::AsteroidsSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* AsteroidsSettings::clone() const { 
    
    RomSettings* rval = new AsteroidsSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void AsteroidsSettings::step(const System& system) {

    // update the reward
    reward_t score = getDecimalScore(0xBE, 0xBD, &system);
    score *= 10;
    m_reward = score - m_score;
    // Deal with score wrapping. In truth this should be done for all games and in a more
    // uniform fashion.
    if (m_reward < 0) {
        const int WRAP_SCORE = 100000;
        m_reward += WRAP_SCORE; 
    }
    m_score = score;

    // update terminal status
    int byte = readRam(&system, 0xBC);
    m_lives = (byte - (byte & 15)) >> 4;
    m_terminal = (m_lives == 0);
}


/* is end of game */
bool AsteroidsSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t AsteroidsSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool AsteroidsSettings::isMinimal(const Action &a) const {
    switch (a) {
        case PLAYER_A_NOOP:
        case PLAYER_A_FIRE:
        case PLAYER_A_UP:
        case PLAYER_A_RIGHT:
        case PLAYER_A_LEFT:
        case PLAYER_A_DOWN:
        case PLAYER_A_UPRIGHT:
        case PLAYER_A_UPLEFT:
        case PLAYER_A_UPFIRE:
        case PLAYER_A_RIGHTFIRE:
        case PLAYER_A_LEFTFIRE:
        case PLAYER_A_DOWNFIRE:
        case PLAYER_A_UPRIGHTFIRE:
        case PLAYER_A_UPLEFTFIRE:
            return true;
        default:
            return false;
    }
}


/* reset the state of the game */
void AsteroidsSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
    m_lives    = 4;
}


        
/* saves the state of the rom settings */
void AsteroidsSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putInt(m_lives);
}

// loads the state of the rom settings
void AsteroidsSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_lives = ser.getInt();
}

// returns a list of mode that the game can be played in
ModeVect AsteroidsSettings::getAvailableModes() {
    ModeVect modes(getNumModes() - 1);
    for (unsigned int i = 0; i < modes.size(); i++) {
        modes[i] = i;
    }
    modes.push_back(0x80); //this is the "kids" mode
    return modes;
}

// set the mode of the game
// the given mode must be one returned by the previous function
void AsteroidsSettings::setMode(game_mode_t m, System &system,
                              std::unique_ptr<StellaEnvironmentWrapper> environment) {

    if(m < 32 || m == 0x80) {
        // read the mode we are currently in
        unsigned char mode = readRam(&system, 0x80);
        // press select until the correct mode is reached
        while (mode != m) {
            environment->pressSelect(2);
            mode = readRam(&system, 0x80);
        }
        //reset the environment to apply changes.
        environment->softReset();
    }
    else {
        throw std::runtime_error("This mode doesn't currently exist for this game");
    }
 }

DifficultyVect AsteroidsSettings::getAvailableDifficulties() {
    DifficultyVect diff = {0, 3};
    return diff;
}

