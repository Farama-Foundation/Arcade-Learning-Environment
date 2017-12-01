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
#include "Tennis.hpp"

#include "../RomUtils.hpp"


TennisSettings::TennisSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* TennisSettings::clone() const { 
    
    RomSettings* rval = new TennisSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void TennisSettings::step(const System& system) {

    // update the reward
    int my_score     = readRam(&system, 0xC5); 
    int oppt_score   = readRam(&system, 0xC6);
    int my_points    = readRam(&system, 0xC7);
    int oppt_points  = readRam(&system, 0xC8);
    int delta_score  = my_score - oppt_score;
    int delta_points = my_points - oppt_points;

    // a reward for the game 
    if (m_prev_delta_points != delta_points)
        m_reward = delta_points - m_prev_delta_points; 
    // a reward for each point
    else if (m_prev_delta_score != delta_score)
        m_reward = delta_score - m_prev_delta_score; 
    else
      m_reward = 0;

    m_prev_delta_points = delta_points;
    m_prev_delta_score = delta_score;

    // update terminal status
    m_terminal = (my_points >= 6 && delta_points >= 2)    || 
                 (oppt_points >= 6 && -delta_points >= 2) || 
                 (my_points == 7 || oppt_points == 7);
}


/* is end of game */
bool TennisSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t TennisSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool TennisSettings::isMinimal(const Action &a) const {

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
void TennisSettings::reset() {
    
    m_reward               = 0;
    m_prev_delta_points    = 0;
    m_prev_delta_score     = 0;
    m_terminal             = false;
}
        
/* saves the state of the rom settings */
void TennisSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putBool(m_terminal);
  
  ser.putInt(m_prev_delta_points);
  ser.putInt(m_prev_delta_score);
}

// loads the state of the rom settings
void TennisSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_terminal = ser.getBool();
  
  m_prev_delta_points = ser.getInt();
  m_prev_delta_score = ser.getInt();
}

// returns a list of mode that the game can be played in
ModeVect TennisSettings::getAvailableModes() {
    ModeVect modes = {0, 2};
    return modes;
}

// set the mode of the game
// the given mode must be one returned by the previous function
void TennisSettings::setMode(game_mode_t m, System &system,
                              std::unique_ptr<StellaEnvironmentWrapper> environment) {

    if(m == 0 || m == 2) {
        // read the mode we are currently in
        unsigned char mode = readRam(&system, 0x80);
        // press select until the correct mode is reached
        while (mode != m) {
            environment->pressSelect(2);
            mode = readRam(&system, 0x80);
        }
        //reset the environment to apply changes.
        environment->softReset();
    }
    else {
        throw std::runtime_error("This mode doesn't currently exist for this game");
    }
 }

DifficultyVect TennisSettings::getAvailableDifficulties() {
    DifficultyVect diff = {0, 1, 2, 3};
    return diff;
}

