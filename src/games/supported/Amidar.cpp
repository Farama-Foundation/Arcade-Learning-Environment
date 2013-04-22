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
#include "Amidar.hpp"

#include "../RomUtils.hpp"


AmidarSettings::AmidarSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* AmidarSettings::clone() const { 
    
    RomSettings* rval = new AmidarSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void AmidarSettings::step(const System& system) {

    // update the reward
    reward_t score = getDecimalScore(0xD9, 0xDA, 0xDB, &system);
    m_reward = score - m_score;
    m_score = score;

    // update terminal status
    int game_over = readRam(&system, 0xD6);
  
    // MGB it takes one step for the system to reset; this assumes we've 
    //  reset
    m_terminal = game_over == 0x80;
}


/* is end of game */
bool AmidarSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t AmidarSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool AmidarSettings::isMinimal(const Action &a) const {

    switch (a) {
        case PLAYER_A_NOOP:
        case PLAYER_A_FIRE:
        case PLAYER_A_UP:
        case PLAYER_A_RIGHT:
        case PLAYER_A_LEFT:
        case PLAYER_A_DOWN:
        case PLAYER_A_UPFIRE:
        case PLAYER_A_RIGHTFIRE:
        case PLAYER_A_LEFTFIRE:
        case PLAYER_A_DOWNFIRE:
            return true;
        default:
            return false;
    }   
}


/* reset the state of the game */
void AmidarSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
}

        
/* saves the state of the rom settings */
void AmidarSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void AmidarSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

