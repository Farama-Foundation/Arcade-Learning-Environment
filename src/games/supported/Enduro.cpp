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
#include "Enduro.hpp"

#include "../RomUtils.hpp"


EnduroSettings::EnduroSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* EnduroSettings::clone() const { 
    
    RomSettings* rval = new EnduroSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void EnduroSettings::step(const System& system) {

    // update the reward
    int score = 0;
    int level = readRam(&system, 0xAD);
    if (level != 0) {
        int cars_passed = getDecimalScore(0xAB, 0xAC, &system);
        if (level == 1) cars_passed = 200 - cars_passed;
        else if (level >= 2) cars_passed = 300 - cars_passed;
        else assert(false);

        // First level has 200 cars
        if (level >= 2) {
            score = 200;
            // For every level after the first, 300 cars
            score += (level - 2) * 300;
        }
        score += cars_passed;
    }

    m_reward = score - m_score;
    m_score = score;

    // update terminal status
    //int timeLeft = readRam(&system, 0xB1);
    int deathFlag = readRam(&system, 0xAF);
    m_terminal = deathFlag == 0xFF;
}


/* is end of game */
bool EnduroSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t EnduroSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool EnduroSettings::isMinimal(const Action &a) const {

    switch (a) {
        case PLAYER_A_NOOP:
        case PLAYER_A_FIRE:
        case PLAYER_A_RIGHT:
        case PLAYER_A_LEFT:
        case PLAYER_A_DOWN:
        case PLAYER_A_DOWNRIGHT:
        case PLAYER_A_DOWNLEFT:
        case PLAYER_A_RIGHTFIRE:
        case PLAYER_A_LEFTFIRE:
            return true;
        default:
            return false;
    }   
}


/* reset the state of the game */
void EnduroSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
}

        
/* saves the state of the rom settings */
void EnduroSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void EnduroSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

ActionVect EnduroSettings::getStartingActions() {
    ActionVect startingActions;
    startingActions.push_back(PLAYER_A_FIRE);
    return startingActions;
}
