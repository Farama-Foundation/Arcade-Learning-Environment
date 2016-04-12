/* *****************************************************************************
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
#include "BreakoutSuper.hpp"

#include "../RomUtils.hpp"


BreakoutSuperSettings::BreakoutSuperSettings() {

    m_reward   = 0;
    m_score    = 0;
    m_lives    = 0;
    m_terminal = false;
    m_started  = false;
    m_mode     = 1;
}


/* create a new instance of the rom */
RomSettings* BreakoutSuperSettings::clone() const {

    RomSettings* rval = new BreakoutSuperSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void BreakoutSuperSettings::step(const System& system) {
    // update the reward
    int x = readRam(&system, 93);
    int y = readRam(&system, 92);
    reward_t score =   1 * ( x & 0x000F) +
                      10 * ((x & 0x00F0) >> 4) +
                     100 * ( y & 0x000F);
    m_reward = score - m_score;
    m_score = score;

    std::cout << "reward" << m_reward << std::endl;

    // update terminal status
    int byte_val = readRam(&system, 97);
    if (!m_started && byte_val == 1) m_started = true;
    int byte_end = readRam(&system, 121);
    m_terminal = byte_end == 247;
    m_lives = 6 - byte_val;
}


/* is end of game */
bool BreakoutSuperSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t BreakoutSuperSettings::getReward() const {

    return m_reward;
}


/* is an action part of the minimal set? */
bool BreakoutSuperSettings::isMinimal(const Action &a) const {

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
void BreakoutSuperSettings::reset(System& system, StellaEnvironment& environment) {

    m_reward   = 0;
    m_score    = 0;
    m_lives    = 0;
    m_terminal = false;
    m_started  = false;
    setMode(m_mode, system, environment);
}


/* saves the state of the rom settings */
void BreakoutSuperSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putBool(m_started);
  ser.putInt(m_lives);
}

/* loads the state of the rom settings */
void BreakoutSuperSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_started = ser.getBool();
  m_lives = ser.getInt();
}

// list of available game modes
ModeVect BreakoutSuperSettings::getAvailableModes() {
    ModeVect modes(5);
    int modes_tmp[] = {1, 3, 5, 7, 8};
    for (int i = 0; i < 5; ++i)
    {
      modes[i] = modes_tmp[i];
    }
    return modes;
}

// set game mode. mode must be within the  outputs of the previous function
void BreakoutSuperSettings::setMode(game_mode_t m, System &system, StellaEnvironment& environment) {
    if(m >= 1 && m <= 9){
        m_mode = m;
        // open the mode selection panel
        environment.pressSelect(10);
        environment.pressSelect(10);
        // read the mode we are currently in
        int mode = readRam(&system, 64);
        // press select until the correct mode is reached
        while(mode != m_mode) {
            environment.pressSelect(10);
            mode = readRam(&system, 64);
        }
        // reset the environment to apply changes.
        environment.soft_reset();
    } else{
        throw std::runtime_error("This mode doesn't currently exist for this game");
    }
}
