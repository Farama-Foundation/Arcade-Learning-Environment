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
#include "Atlantis.hpp"

#include "../RomUtils.hpp"


AtlantisSettings::AtlantisSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* AtlantisSettings::clone() const { 
    
    RomSettings* rval = new AtlantisSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void AtlantisSettings::step(const System& system) {

    // update the reward
    reward_t score = getDecimalScore(34, 35, &system); 
    score *= 100;
    m_reward = score - m_score;
    m_score = score;

    // update terminal status
    int lives = readRam(&system, 0xF1);
    m_terminal = lives == 0xFF;
  
    // MGB: Also d4-da contain the 'building' status; 05 indicates a destroyed
    //      building, 00 a live building
}


/* is end of game */
bool AtlantisSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t AtlantisSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool AtlantisSettings::isMinimal(const Action &a) const {

    switch (a) {
        case PLAYER_A_NOOP:
        case PLAYER_A_FIRE:
        case PLAYER_A_RIGHTFIRE:
        case PLAYER_A_LEFTFIRE:
            return true;
        default:
            return false;
    }   
}


/* reset the state of the game */
void AtlantisSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
}


        
/* saves the state of the rom settings */
void AtlantisSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void AtlantisSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

