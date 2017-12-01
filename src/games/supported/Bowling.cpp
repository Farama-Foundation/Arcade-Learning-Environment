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
#include "Bowling.hpp"

#include "../RomUtils.hpp"


BowlingSettings::BowlingSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* BowlingSettings::clone() const { 
    
    RomSettings* rval = new BowlingSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void BowlingSettings::step(const System& system) {

    // update the reward
    reward_t score = getDecimalScore(0xA1, 0xA6, &system);
    m_reward = score - m_score;
    m_score = score;

    // update terminal status
    int round = readRam(&system, 0xA4);
    m_terminal = round > 0x10;
}


/* is end of game */
bool BowlingSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t BowlingSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool BowlingSettings::isMinimal(const Action &a) const {

    switch (a) {
        case PLAYER_A_NOOP:
        case PLAYER_A_FIRE:
        case PLAYER_A_UP:
        case PLAYER_A_DOWN:
        case PLAYER_A_UPFIRE:
        case PLAYER_A_DOWNFIRE:
            return true;
        default:
            return false;
    }   
}


/* reset the state of the game */
void BowlingSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
}


        
/* saves the state of the rom settings */
void BowlingSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void BowlingSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

// returns a list of mode that the game can be played in
ModeVect BowlingSettings::getAvailableModes() {
    ModeVect modes = {0, 2, 4};
    return modes;
}

// set the mode of the game
// the given mode must be one returned by the previous function
void BowlingSettings::setMode(game_mode_t m, System &system,
                              std::unique_ptr<StellaEnvironmentWrapper> environment) {

    if(m == 0 || m == 2 || m == 4) {
        // read the mode we are currently in
        unsigned char mode = readRam(&system, 2);
        // press select until the correct mode is reached
        while (mode != m) {
            environment->pressSelect(2);
            mode = readRam(&system, 2);
        }
        //reset the environment to apply changes.
        environment->softReset();
    }
    else {
        throw std::runtime_error("This mode doesn't currently exist for this game");
    }
 }

DifficultyVect BowlingSettings::getAvailableDifficulties() {
    DifficultyVect diff = {0, 1};
    return diff;
}

