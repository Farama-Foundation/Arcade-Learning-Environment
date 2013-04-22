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
#include "Freeway.hpp"

#include "../RomUtils.hpp"


FreewaySettings::FreewaySettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* FreewaySettings::clone() const { 
    
    RomSettings* rval = new FreewaySettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void FreewaySettings::step(const System& system) {

    // update the reward
    int score = getDecimalScore(103, -1, &system);
    int reward = score - m_score;
    if (reward < 0) reward = 0;      
    if (reward > 1) reward = 1;
    m_reward = reward;
    m_score = score;

    // update terminal status
    m_terminal = readRam(&system, 22) == 1;
}


/* is end of game */
bool FreewaySettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t FreewaySettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool FreewaySettings::isMinimal(const Action &a) const {

    switch (a) {
        case PLAYER_A_NOOP:
        case PLAYER_A_UP:
        case PLAYER_A_DOWN:
            return true;
        default:
            return false;
    }   
}


/* reset the state of the game */
void FreewaySettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
}
        
/* saves the state of the rom settings */
void FreewaySettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void FreewaySettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

