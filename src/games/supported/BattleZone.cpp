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
#include "BattleZone.hpp"

#include "../RomUtils.hpp"


BattleZoneSettings::BattleZoneSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* BattleZoneSettings::clone() const { 
    
    RomSettings* rval = new BattleZoneSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void BattleZoneSettings::step(const System& system) {

    // update the reward
    int first_val = readRam(&system, 0x9D);
    int first_right_digit = first_val & 15;
    int first_left_digit = (first_val - first_right_digit) >> 4;
    if (first_left_digit == 10) first_left_digit = 0;

    int second_val = readRam(&system, 0x9E);
    int second_right_digit = second_val & 15;
    int second_left_digit = (second_val - second_right_digit) >> 4;
    if (second_right_digit == 10) second_right_digit = 0;
    if (second_left_digit == 10) second_left_digit = 0;
    
    reward_t score = 0;
    score += first_left_digit;
    score += 10 * second_right_digit;
    score += 100 * second_left_digit;
    score *= 1000;
    m_reward = score - m_score;
    m_score = score;

    // update terminal status
    int lives = readRam(&system, 0xBA) & 0xF;
    m_terminal = lives == 0;
}


/* is end of game */
bool BattleZoneSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t BattleZoneSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool BattleZoneSettings::isMinimal(const Action &a) const {

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
void BattleZoneSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
}


        
/* saves the state of the rom settings */
void BattleZoneSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void BattleZoneSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

