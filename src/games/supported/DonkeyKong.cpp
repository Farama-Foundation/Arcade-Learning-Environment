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
#include "DonkeyKong.hpp"

#include "../RomUtils.hpp"


DonkeyKongSettings::DonkeyKongSettings() {
    reset();
}


/* create a new instance of the rom */
RomSettings* DonkeyKongSettings::clone() const { 
    RomSettings* rval = new DonkeyKongSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void DonkeyKongSettings::step(const System& system) {
    // update the reward
    int score = getDecimalScore(0x88, 0x87, &system);
    score *= 100;
    int reward = score - m_score;
    m_reward = reward;
    m_score = score;

    // update lives and terminal status
    m_lives = readRam(&system, 0xA3);
    m_terminal = (m_lives == 0)
		&& readRam(&system, 0x8F) == 0x03
		&& readRam(&system, 0x8B) == 0x1F;
}

/* is end of game */
bool DonkeyKongSettings::isTerminal() const {
    return m_terminal;
};


/* get the most recently observed reward */
reward_t DonkeyKongSettings::getReward() const { 
    return m_reward; 
}


/* is an action part of the minimal set? */
bool DonkeyKongSettings::isMinimal(const Action &a) const {
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
void DonkeyKongSettings::reset() {
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
    m_lives    = 2;
}
        
/* saves the state of the rom settings */
void DonkeyKongSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putInt(m_lives);
}

// loads the state of the rom settings
void DonkeyKongSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_lives = ser.getInt();
}

