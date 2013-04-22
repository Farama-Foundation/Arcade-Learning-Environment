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
#include "BeamRider.hpp"

#include "../RomUtils.hpp"


BeamRiderSettings::BeamRiderSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* BeamRiderSettings::clone() const { 
    
    RomSettings* rval = new BeamRiderSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void BeamRiderSettings::step(const System& system) {

    // update the reward
    int score = getDecimalScore(9, 10, 11, &system);
    m_reward = score - m_score;
    m_score = score;

    // update terminal status
    int byte_val = readRam(&system, 5);
    m_terminal = byte_val == 255;
    byte_val = byte_val & 15;
    m_terminal = m_terminal || byte_val < 0;
}


/* is end of game */
bool BeamRiderSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t BeamRiderSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool BeamRiderSettings::isMinimal(const Action &a) const {

    switch (a) {
        case PLAYER_A_NOOP:
        case PLAYER_A_FIRE:
        case PLAYER_A_UP:
        case PLAYER_A_RIGHT:
        case PLAYER_A_LEFT:
        case PLAYER_A_UPRIGHT:
        case PLAYER_A_UPLEFT:
        case PLAYER_A_RIGHTFIRE:
        case PLAYER_A_LEFTFIRE:
            return true;
        default:
            return false;
    }   
}


/* reset the state of the game */
void BeamRiderSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
}

        
/* saves the state of the rom settings */
void BeamRiderSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void BeamRiderSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

ActionVect BeamRiderSettings::getStartingActions() {
    ActionVect startingActions;
    startingActions.push_back(PLAYER_A_RIGHT);
    return startingActions;
}

