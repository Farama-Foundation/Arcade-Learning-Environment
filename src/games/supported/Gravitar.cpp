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
#include "Gravitar.hpp"

#include "../RomUtils.hpp"


GravitarSettings::GravitarSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* GravitarSettings::clone() const { 
    
    RomSettings* rval = new GravitarSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void GravitarSettings::step(const System& system) {

    // update the reward
    int score = getDecimalScore(9, 8, 7, &system);
    int reward = score - m_score;
    m_reward = reward;
    m_score = score;

    // update terminal status
    int death_byte = readRam(&system, 0x81);
    m_terminal = death_byte == 0x01;
}


/* is end of game */
bool GravitarSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t GravitarSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool GravitarSettings::isMinimal(const Action &a) const {

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
void GravitarSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
}

        
/* saves the state of the rom settings */
void GravitarSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void GravitarSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

ActionVect GravitarSettings::getStartingActions() {
    ActionVect startingActions;
    for (int i=0; i<16; i++) 
        startingActions.push_back(PLAYER_A_FIRE);
    return startingActions;
}
