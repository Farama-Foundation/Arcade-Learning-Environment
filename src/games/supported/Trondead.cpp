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
#include "Trondead.hpp"

#include "../RomUtils.hpp"


TrondeadSettings::TrondeadSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* TrondeadSettings::clone() const { 
    
    RomSettings* rval = new TrondeadSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void TrondeadSettings::step(const System& system) {

    // update the reward
    int score = getDecimalScore(0xBF, 0xBE, 0xBD, &system);
    int reward = score - m_score;
    m_reward = reward;
    m_score = score;

    // update terminal status
    int hit_count = readRam(&system, 0xC8);
    m_terminal = (hit_count == 5);	// Five times hit
    m_lives = 5 - hit_count;
}


/* is end of game */
bool TrondeadSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t TrondeadSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool TrondeadSettings::isMinimal(const Action &a) const {

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
void TrondeadSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
    m_lives    = 5;
}
        
/* saves the state of the rom settings */
void TrondeadSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putInt(m_lives);
}

// loads the state of the rom settings
void TrondeadSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_lives = ser.getInt();
}

