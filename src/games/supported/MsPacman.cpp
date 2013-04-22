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
#include "MsPacman.hpp"

#include "../RomUtils.hpp"


MsPacmanSettings::MsPacmanSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* MsPacmanSettings::clone() const { 
    
    RomSettings* rval = new MsPacmanSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void MsPacmanSettings::step(const System& system) {

    // update the reward
    int score = getDecimalScore(0xF8, 0xF9, 0xFA, &system);
    int reward = score - m_score;
    m_reward = reward;
    m_score = score;

    // update terminal status
    int lives = readRam(&system, 0xFB) & 0xF;
    // MGB Did not work int black_screen_byte = readRam(&system, 0x94);
    int death_timer = readRam(&system, 0xA7);
    m_terminal = lives == 0 && death_timer == 0x53;
}


/* is end of game */
bool MsPacmanSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t MsPacmanSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool MsPacmanSettings::isMinimal(const Action &a) const {

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


/* reset the state of the game */
void MsPacmanSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
}


        
/* saves the state of the rom settings */
void MsPacmanSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void MsPacmanSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

