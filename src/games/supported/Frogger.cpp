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
#include "Frogger.hpp"

#include "../RomUtils.hpp"

FroggerSettings::FroggerSettings() {
    reset();
}

/* create a new instance of the rom */
RomSettings* FroggerSettings::clone() const { 
    RomSettings* rval = new FroggerSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void FroggerSettings::step(const System& system) {
    // update the reward
    int score = getDecimalScore(0xCE, 0xCC, &system);
    int reward = score - m_score;
    m_reward = reward;
    m_score = score;

    // update terminal status
	m_lives = readRam(&system, 0xD0);
    m_terminal = readRam(&system, 0xD0) == 0xFF;
}


/* is end of game */
bool FroggerSettings::isTerminal() const {
    return m_terminal;
};


/* get the most recently observed reward */
reward_t FroggerSettings::getReward() const { 
    return m_reward; 
}

/* is an action part of the minimal set? */
bool FroggerSettings::isMinimal(const Action &a) const {
    switch (a) {
        case PLAYER_A_NOOP:
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
void FroggerSettings::reset() {
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
    m_lives    = 4;
}
        
/* saves the state of the rom settings */
void FroggerSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putInt(m_lives);
}

// loads the state of the rom settings
void FroggerSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_lives = ser.getInt();
}

ActionVect FroggerSettings::getStartingActions() {
    ActionVect startingActions;
    startingActions.push_back(RESET);
    return startingActions;
}
