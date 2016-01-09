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
#include "SirLancelot.hpp"

#include "../RomUtils.hpp"


SirLancelotSettings::SirLancelotSettings() {
    reset();
}


/* create a new instance of the rom */
RomSettings* SirLancelotSettings::clone() const { 
    RomSettings* rval = new SirLancelotSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void SirLancelotSettings::step(const System& system) {
    // update the reward
    int score = getDecimalScore(0xA0, 0x9F, 0x9E, &system);
    int reward = score - m_score;
    m_reward = reward;
    m_score = score;

    // update terminal status
	m_lives = readRam(&system, 0xA9);
    m_terminal = (m_lives == 0) && readRam(&system, 0xA7) == 0xA0;
}


/* is end of game */
bool SirLancelotSettings::isTerminal() const {
    return m_terminal;
};


/* get the most recently observed reward */
reward_t SirLancelotSettings::getReward() const { 
    return m_reward; 
}


/* is an action part of the minimal set? */
bool SirLancelotSettings::isMinimal(const Action &a) const {
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
void SirLancelotSettings::reset() {
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
    m_lives    = 3;
}
        
/* saves the state of the rom settings */
void SirLancelotSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putInt(m_lives);
}

// loads the state of the rom settings
void SirLancelotSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_lives = ser.getInt();
}

ActionVect SirLancelotSettings::getStartingActions() {
    ActionVect startingActions;
    startingActions.push_back(RESET);
    startingActions.push_back(PLAYER_A_LEFT);
    return startingActions;
}


