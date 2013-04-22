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
#include "JamesBond.hpp"

#include "../RomUtils.hpp"


JamesBondSettings::JamesBondSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* JamesBondSettings::clone() const { 
    
    RomSettings* rval = new JamesBondSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void JamesBondSettings::step(const System& system) {

    // update the reward
    int score = getDecimalScore(0xDC, 0xDD, 0xDE, &system);
    int reward = score - m_score;
    m_reward = reward;
    m_score = score;

    // update terminal status
    int lives = readRam(&system, 0x86) & 0xF;
    int screen_byte = readRam(&system, 0x8C);

    // byte 0x8C is 0x68 when we die; it does not remain so forever, as
    // the system loops back to start state after a while (where fire will
    // start a new game)
    m_terminal = lives == 0 && screen_byte == 0x68;
}


/* is end of game */
bool JamesBondSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t JamesBondSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool JamesBondSettings::isMinimal(const Action &a) const {

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
void JamesBondSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
}

        
/* saves the state of the rom settings */
void JamesBondSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void JamesBondSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

