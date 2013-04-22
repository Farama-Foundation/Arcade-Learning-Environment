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
#include "DemonAttack.hpp"

#include "../RomUtils.hpp"


DemonAttackSettings::DemonAttackSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* DemonAttackSettings::clone() const { 
    
    RomSettings* rval = new DemonAttackSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void DemonAttackSettings::step(const System& system) {

    // update the reward
    int score = getDecimalScore(0x85, 0x83, 0x81, &system);
    // MGB: something funny with the RAM; it is not initialized to 0?
    if (readRam(&system, 0x81) == 0xAB && 
        readRam(&system, 0x83) == 0xCD &&
        readRam(&system, 0x85) == 0xEA) score = 0;
    m_reward = score - m_score;
    m_score = score;

    // update terminal status
    int lives        = readRam(&system, 0xF2);
    int display_flag = readRam(&system, 0xF1);
    m_terminal = lives == 0 && display_flag == 0xBD;
}


/* is end of game */
bool DemonAttackSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t DemonAttackSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool DemonAttackSettings::isMinimal(const Action &a) const {

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
void DemonAttackSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
}

        
/* saves the state of the rom settings */
void DemonAttackSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void DemonAttackSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

