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
#include "Gopher.hpp"

#include "../RomUtils.hpp"


GopherSettings::GopherSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* GopherSettings::clone() const { 
    
    RomSettings* rval = new GopherSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void GopherSettings::step(const System& system) {

    // update the reward
    int score = getDecimalScore(0xB2, 0xB1, 0xB0, &system); 
    int reward = score - m_score;
    m_reward = reward;
    m_score = score;

    // update terminal status
    int carrot_bits = readRam(&system, 0xB4) & 0x7;
    m_terminal = carrot_bits == 0;
}


/* is end of game */
bool GopherSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t GopherSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool GopherSettings::isMinimal(const Action &a) const {

    switch (a) {
        case PLAYER_A_NOOP:
        case PLAYER_A_FIRE:
        case PLAYER_A_UP:
        case PLAYER_A_RIGHT:
        case PLAYER_A_LEFT:
        case PLAYER_A_UPFIRE:
        case PLAYER_A_RIGHTFIRE:
        case PLAYER_A_LEFTFIRE:
            return true;
        default:
            return false;
    }   
}


/* reset the state of the game */
void GopherSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
}

        
/* saves the state of the rom settings */
void GopherSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void GopherSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

ActionVect GopherSettings::getStartingActions() {
    ActionVect startingActions;
    startingActions.push_back(PLAYER_A_FIRE);
    return startingActions;
}
