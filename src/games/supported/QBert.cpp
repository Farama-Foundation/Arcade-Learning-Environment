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
#include "QBert.hpp"

#include "../RomUtils.hpp"


QBertSettings::QBertSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* QBertSettings::clone() const { 
    
    RomSettings* rval = new QBertSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void QBertSettings::step(const System& system) {
    // update terminal status
    int lives_value = readRam(&system, 0x88);
    // Lives start at 2 (4 lives, 3 displayed) and go down to 0xFE (death)
    // Alternatively we can die and reset within one frame; we catch this case
    m_terminal = (lives_value == 0xFE) ||
      (lives_value == 0x02 && m_last_lives == 0xFF);
    
    m_last_lives = lives_value;

    // update the reward
    // Ignore reward if reset the game via the fire button; otherwise the agent 
    //  gets a big negative reward on its last step 
    if (!m_terminal) {
      int score = getDecimalScore(0xDB, 0xDA, 0xD9, &system);
      int reward = score - m_score;
      m_reward = reward;
      m_score = score;
    }
    else {
      m_reward = 0;
    }
}


/* is end of game */
bool QBertSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t QBertSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool QBertSettings::isMinimal(const Action &a) const {

    switch (a) {
        case PLAYER_A_NOOP:
        case PLAYER_A_FIRE:
        case PLAYER_A_UP:
        case PLAYER_A_RIGHT:
        case PLAYER_A_LEFT:
        case PLAYER_A_DOWN:
            return true;
        default:
            return false;
    }   
}


/* reset the state of the game */
void QBertSettings::reset() {
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
    // Anything non-0xFF
    m_last_lives = 2;
}
        
/* saves the state of the rom settings */
void QBertSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putInt(m_last_lives);
}

// loads the state of the rom settings
void QBertSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_last_lives = ser.getInt();
}

