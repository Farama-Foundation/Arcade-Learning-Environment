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
#include "FishingDerby.hpp"

#include "../RomUtils.hpp"
using namespace std;


FishingDerbySettings::FishingDerbySettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* FishingDerbySettings::clone() const { 
    
    RomSettings* rval = new FishingDerbySettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void FishingDerbySettings::step(const System& system) {

    // update the reward
    int my_score   = max(getDecimalScore(0xBD, &system), 0);
    int oppt_score = max(getDecimalScore(0xBE, &system), 0);
    int score = my_score - oppt_score;
    m_reward = score - m_score;
    m_score = score;

    // update terminal status
    int my_score_byte      = readRam(&system, 0xBD);
    int my_oppt_score_byte = readRam(&system, 0xBE);

    m_terminal = my_score_byte == 0x99 || my_oppt_score_byte == 0x99;
}


/* is end of game */
bool FishingDerbySettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t FishingDerbySettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool FishingDerbySettings::isMinimal(const Action &a) const {

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
void FishingDerbySettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
}

        
/* saves the state of the rom settings */
void FishingDerbySettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void FishingDerbySettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

DifficultyVect FishingDerbySettings::getAvailableDifficulties() {
    DifficultyVect diff = {0, 1, 2, 3};
    return diff;
}


