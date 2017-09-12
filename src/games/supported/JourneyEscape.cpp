/* *****************************************************************************
 * The method lives() is based on Xitari's code, from Google Inc.
 *
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
#include "JourneyEscape.hpp"

#include "../RomUtils.hpp"


JourneyEscapeSettings::JourneyEscapeSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* JourneyEscapeSettings::clone() const { 
    
    RomSettings* rval = new JourneyEscapeSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void JourneyEscapeSettings::step(const System& system) {

    // update the reward
    int score = getDecimalScore(0x92, 0x91, 0x90, &system);
    int reward = score - m_score;
    if (reward == 50000) reward = 0;  // HACK: ignoring starting cash
    m_reward = reward;
    m_score = score;

    // update terminal status
    int minutes = readRam(&system, 0x95);
    int seconds = readRam(&system, 0x96);
    m_terminal = minutes == 0 && seconds == 0;
}


/* is end of game */
bool JourneyEscapeSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t JourneyEscapeSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool JourneyEscapeSettings::isMinimal(const Action &a) const {

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
void JourneyEscapeSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
}

        
/* saves the state of the rom settings */
void JourneyEscapeSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void JourneyEscapeSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

ActionVect JourneyEscapeSettings::getStartingActions() {
    ActionVect startingActions;
    startingActions.push_back(PLAYER_A_FIRE);
    return startingActions;
}

DifficultyVect JourneyEscapeSettings::getAvailableDifficulties() {
    DifficultyVect diff = {0, 1};
    return diff;
}

