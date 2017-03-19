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
#include "Kaboom.hpp"

#include "../RomUtils.hpp"
#include <iostream>

KaboomSettings::KaboomSettings() {
    reset();
}


/* create a new instance of the rom */
RomSettings* KaboomSettings::clone() const {
    
    RomSettings* rval = new KaboomSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void KaboomSettings::step(const System& system) {

    // update the reward
    reward_t score = getDecimalScore(0xA5, 0xA4, 0xA3, &system);
    m_reward = score - m_score;
    m_score = score;

    // update terminal status
    int lives = readRam(&system, 0xA1);
    m_terminal = lives == 0x0 || m_score == 999999;
}


/* is end of game */
bool KaboomSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t KaboomSettings::getReward() const {
     return m_reward;
}


/* is an action part of the minimal set? */
bool KaboomSettings::isMinimal(const Action &a) const {

    switch (a) {
        case PLAYER_A_NOOP:
        case PLAYER_A_FIRE:
        case PLAYER_A_RIGHT:
        case PLAYER_A_LEFT:
             return true;
        default:
            return false;
    }   
}


/* reset the state of the game */
void KaboomSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
}
        
/* saves the state of the rom settings */
void KaboomSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void KaboomSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

ActionVect KaboomSettings::getStartingActions() {
    ActionVect startingActions;
    startingActions.push_back(PLAYER_A_FIRE);
    return startingActions;
}
