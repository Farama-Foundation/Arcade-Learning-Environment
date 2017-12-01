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
#include "Pong.hpp"

#include "../RomUtils.hpp"


PongSettings::PongSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* PongSettings::clone() const { 
    
    RomSettings* rval = new PongSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void PongSettings::step(const System& system) {

    // update the reward
    int x = readRam(&system, 13); // cpu score
    int y = readRam(&system, 14); // player score
    reward_t score = y - x;
    m_reward = score - m_score;
    m_score = score;

    // update terminal status
    // (game over when a player reaches 21)
    m_terminal = x == 21 || y == 21;
}


/* is end of game */
bool PongSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t PongSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool PongSettings::isMinimal(const Action &a) const {

    switch (a) {
        case PLAYER_A_NOOP:
        case PLAYER_A_FIRE:
        case PLAYER_A_RIGHT:
        case PLAYER_A_LEFT:
        case PLAYER_A_RIGHTFIRE:
        case PLAYER_A_LEFTFIRE:
            return true;
        default:
            return false;
    }   
}


/* reset the state of the game */
void PongSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
}

        
/* saves the state of the rom settings */
void PongSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void PongSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

// returns a list of mode that the game can be played in
ModeVect PongSettings::getAvailableModes() {
    ModeVect modes(getNumModes());
    for (unsigned int i = 0; i < modes.size(); i++) {
        modes[i] = i;
    }
    return modes;
}

// set the mode of the game
// the given mode must be one returned by the previous function
void PongSettings::setMode(game_mode_t m, System &system,
                              std::unique_ptr<StellaEnvironmentWrapper> environment) {

    if(m < getNumModes()) {
        // read the mode we are currently in
        unsigned char mode = readRam(&system, 0x96);
        // press select until the correct mode is reached
        while (mode != m) {
            environment->pressSelect(2);
            mode = readRam(&system, 0x96);
        }
        //reset the environment to apply changes.
        environment->softReset();
    }
    else {
        throw std::runtime_error("This mode doesn't currently exist for this game");
    }
 }

DifficultyVect PongSettings::getAvailableDifficulties() {
    DifficultyVect diff = {0, 1};
    return diff;
}

