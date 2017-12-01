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
#include "Boxing.hpp"

#include "../RomUtils.hpp"


BoxingSettings::BoxingSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* BoxingSettings::clone() const { 
    
    RomSettings* rval = new BoxingSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void BoxingSettings::step(const System& system) {

    // update the reward
    int my_score   = getDecimalScore(0x92, &system);
    int oppt_score = getDecimalScore(0x93, &system);

    // handle KO
    if (readRam(&system, 0x92) == 0xC0) my_score   = 100;
    if (readRam(&system, 0x93) == 0xC0) oppt_score = 100;
    reward_t score = my_score - oppt_score;
    m_reward = score - m_score;
    m_score = score;

    // update terminal status
    // if either is KO, the game is over
    if (my_score == 100 || oppt_score == 100) {
        m_terminal = true;
    } else {  // otherwise check to see if out of time
        int minutes = readRam(&system, 0x90) >> 4;
        int seconds = (readRam(&system, 0x91) & 0xF) + 
                      (readRam(&system, 0x91) >> 4) * 10;
        m_terminal = minutes == 0 && seconds == 0;
    }
}


/* is end of game */
bool BoxingSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t BoxingSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool BoxingSettings::isMinimal(const Action &a) const {

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
void BoxingSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
}


        
/* saves the state of the rom settings */
void BoxingSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void BoxingSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

DifficultyVect BoxingSettings::getAvailableDifficulties() {
    DifficultyVect diff = {0, 1, 2, 3};
    return diff;
}
