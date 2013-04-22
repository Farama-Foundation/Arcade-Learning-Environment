/* *****************************************************************************
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
#include "RiverRaid.hpp"

#include "../RomUtils.hpp"


RiverRaidSettings::RiverRaidSettings() {

    m_ram_vals_to_digits[0]  = 0;
    m_ram_vals_to_digits[8]  = 1;
    m_ram_vals_to_digits[16] = 2;
    m_ram_vals_to_digits[24] = 3;
    m_ram_vals_to_digits[32] = 4;
    m_ram_vals_to_digits[40] = 5;
    m_ram_vals_to_digits[48] = 6;
    m_ram_vals_to_digits[56] = 7;
    m_ram_vals_to_digits[64] = 8;
    m_ram_vals_to_digits[72] = 9;

    reset();
}


/* create a new instance of the rom */
RomSettings* RiverRaidSettings::clone() const { 
    
    RomSettings* rval = new RiverRaidSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void RiverRaidSettings::step(const System& system) {

    // update the reward
    int score = 0;
    int digit = m_ram_vals_to_digits[readRam(&system, 87)];
    score += digit;
    digit = m_ram_vals_to_digits[readRam(&system, 85)];
    score += 10 * digit;
    digit = m_ram_vals_to_digits[readRam(&system, 83)];
    score += 100 * digit;
    digit = m_ram_vals_to_digits[readRam(&system, 81)];
    score += 1000 * digit;
    digit = m_ram_vals_to_digits[readRam(&system, 79)];
    score += 10000 * digit;
    digit = m_ram_vals_to_digits[readRam(&system, 77)];
    score += 100000 * digit;
    m_reward = score - m_score;
    m_score = score;

    // update terminal status
    int byte_val = readRam(&system, 64);
    m_terminal = byte_val == 88 && m_lives == 89;
    m_lives = byte_val;
}


/* is end of game */
bool RiverRaidSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t RiverRaidSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool RiverRaidSettings::isMinimal(const Action &a) const {

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
void RiverRaidSettings::reset() {
    
    m_lives = 0;
    m_reward     = 0;
    m_score      = 0;
    m_terminal   = false;
}

        
/* saves the state of the rom settings */
void RiverRaidSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putInt(m_lives);
}

// loads the state of the rom settings
void RiverRaidSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_lives = ser.getInt();
}
