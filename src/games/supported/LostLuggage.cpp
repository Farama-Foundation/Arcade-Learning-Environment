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
#include "LostLuggage.hpp"

#include "../RomUtils.hpp"

LostLuggageSettings::LostLuggageSettings() {
    reset();
}

/* create a new instance of the rom */
RomSettings* LostLuggageSettings::clone() const { 
    RomSettings* rval = new LostLuggageSettings();
    *rval = *this;
    return rval;
}

/* process the latest information from ALE */
void LostLuggageSettings::step(const System& system) {
    // update the reward
    int score = getDecimalScore(0x96, 0x95, 0x94, &system);
    int reward = score - m_score;
    m_reward = reward;
    m_score = score;

    // update terminal status
    m_lives = readRam(&system, 0xCA);
    m_terminal = (m_lives == 0)
        && readRam(&system, 0xC8) == 0x0A
        && readRam(&system, 0xA5) == 0x00
        && readRam(&system, 0xA9) == 0x00;
}

/* is end of game */
bool LostLuggageSettings::isTerminal() const {
    return m_terminal;
};


/* get the most recently observed reward */
reward_t LostLuggageSettings::getReward() const { 
    return m_reward; 
}


/* is an action part of the minimal set? */
bool LostLuggageSettings::isMinimal(const Action &a) const {
    switch (a) {
        case PLAYER_A_NOOP:
        case PLAYER_A_UP:
        case PLAYER_A_RIGHT:
        case PLAYER_A_LEFT:
        case PLAYER_A_DOWN:
        case PLAYER_A_UPRIGHT:
        case PLAYER_A_UPLEFT:
        case PLAYER_A_DOWNRIGHT:
        case PLAYER_A_DOWNLEFT:
            return true;
        default:
            return false;
    }   
}

bool LostLuggageSettings::isLegal(const Action &a) const {
  switch (a) {
    // Don't allow pressing 'fire'
    case PLAYER_A_FIRE:
    case PLAYER_A_UPFIRE:
    case PLAYER_A_DOWNFIRE:
    case PLAYER_A_LEFTFIRE:
    case PLAYER_A_RIGHTFIRE:
    case PLAYER_A_UPLEFTFIRE:
    case PLAYER_A_UPRIGHTFIRE:
    case PLAYER_A_DOWNLEFTFIRE:
    case PLAYER_A_DOWNRIGHTFIRE:
      return false;
    default:
      return true;
  }
}

/* reset the state of the game */
void LostLuggageSettings::reset() {
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
    m_lives    = 3;
}
        
/* saves the state of the rom settings */
void LostLuggageSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putInt(m_lives);
}

// loads the state of the rom settings
void LostLuggageSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_lives = ser.getInt();
}

ActionVect LostLuggageSettings::getStartingActions() {
    ActionVect startingActions;
    startingActions.push_back(RESET);
    return startingActions;
}

