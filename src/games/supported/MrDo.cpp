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
#include "MrDo.hpp"

#include "../RomUtils.hpp"


MrDoSettings::MrDoSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* MrDoSettings::clone() const { 
    
    RomSettings* rval = new MrDoSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void MrDoSettings::step(const System& system) {

    // update the reward
    int score = getDecimalScore(0x82, 0x83, &system);
    score *= 10;
    int reward = score - m_score;
    m_reward = reward;
    m_score = score;

    // update terminal status
    m_lives = readRam(&system, 0xDB);
    m_terminal = readRam(&system, 0xDA) == 0x40;
}


/* is end of game */
bool MrDoSettings::isTerminal() const {
    return m_terminal;
};


/* get the most recently observed reward */
reward_t MrDoSettings::getReward() const { 
    return m_reward; 
}


/* is an action part of the minimal set? */
bool MrDoSettings::isMinimal(const Action &a) const {

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
        case PLAYER_A_DOWNFIRE:
            return true;
        default:
            return false;
    }   
}


/* reset the state of the game */
void MrDoSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
    m_lives    = 4;
}
        
/* saves the state of the rom settings */
void MrDoSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putInt(m_lives);
}

// loads the state of the rom settings
void MrDoSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_lives = ser.getInt();
}

ActionVect MrDoSettings::getStartingActions() {
    ActionVect startingActions;
    startingActions.push_back(PLAYER_A_FIRE);
    return startingActions;
}

