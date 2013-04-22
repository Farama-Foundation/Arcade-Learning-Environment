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
#include "VideoPinball.hpp"

#include "../RomUtils.hpp"


VideoPinballSettings::VideoPinballSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* VideoPinballSettings::clone() const { 
    
    RomSettings* rval = new VideoPinballSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void VideoPinballSettings::step(const System& system) {

    // update the reward
    int score = getDecimalScore(0xB0, 0xB2, 0xB4, &system);
    int reward = score - m_score;
    m_reward = reward;
    m_score = score;

    // update terminal status
    int flag = readRam(&system, 0xAF) & 0x1;
    m_terminal = flag != 0;
}


/* is end of game */
bool VideoPinballSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t VideoPinballSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool VideoPinballSettings::isMinimal(const Action &a) const {
    
    switch (a) {
        case PLAYER_A_NOOP:
        case PLAYER_A_FIRE:
        case PLAYER_A_UP:
        case PLAYER_A_RIGHT:
        case PLAYER_A_LEFT:
        case PLAYER_A_DOWN:
        case PLAYER_A_UPFIRE:
        case PLAYER_A_RIGHTFIRE:
        case PLAYER_A_LEFTFIRE:
            return true;
        default:
            return false;
    }   
}


/* reset the state of the game */
void VideoPinballSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
}
        
/* saves the state of the rom settings */
void VideoPinballSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void VideoPinballSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

