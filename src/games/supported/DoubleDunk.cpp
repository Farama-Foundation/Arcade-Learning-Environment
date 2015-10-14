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
#include "DoubleDunk.hpp"
#include "../RomUtils.hpp"

using namespace std;

DoubleDunkSettings::DoubleDunkSettings() {

    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
}

/* create a new instance of the rom */
RomSettings* DoubleDunkSettings::clone() const { 
    
    RomSettings* rval = new DoubleDunkSettings();
    *rval = *this;
    return rval;
}


/* process the latest information from ALE */
void DoubleDunkSettings::step(const System& system) {

    // update the reward
    int my_score = getDecimalScore(0xF6, &system);
    int oppt_score = getDecimalScore(0xF7, &system);
    int score = my_score - oppt_score;
    m_reward = score - m_score;
    m_score = score;

    // update terminal status
    int some_value = readRam(&system, 0xFE);
    m_terminal = (my_score >= 24 || oppt_score >= 24) && some_value == 0xE7;
}


/* is end of game */
bool DoubleDunkSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t DoubleDunkSettings::getReward() const { 

    return m_reward; 
}


/* is an action part of the minimal set? */
bool DoubleDunkSettings::isMinimal(const Action &a) const {

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
void DoubleDunkSettings::reset(System& system, StellaEnvironment& environment) {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
    setMode(m_mode, system, environment);
}

        
/* saves the state of the rom settings */
void DoubleDunkSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void DoubleDunkSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

ActionVect DoubleDunkSettings::getStartingActions() {
    ActionVect startingActions;
    startingActions.push_back(PLAYER_A_UPFIRE);
    return startingActions;
}

// returns a list of mode that the game can be played in
ModeVect DoubleDunkSettings::getAvailableModes(){
    // this game has a menu that allows to define various yes/no options
    // setting these options define in a way a different mode
    // there are 4 relevant options, which makes 2^4=16 available modes
    ModeVect modes(16);
    for(unsigned i = 0; i < 16; i++){
        modes[i] = i;
    }
    return modes;
}

// set the mode of the game
// the given mode must be one returned by the previous function
void DoubleDunkSettings::setMode(game_mode_t m, System &system, StellaEnvironment& environment){
    if(m < 16){ /*m >= 0 is implicit, since m is an unsigned int*/
        m_mode = m;
        //push the select button to open the menu
        environment.pressSelect(1);
        //discard the first two entries (irrelevant)
        environment.act(PLAYER_A_DOWN, PLAYER_B_NOOP);
        environment.act(PLAYER_A_NOOP, PLAYER_B_NOOP);
        environment.act(PLAYER_A_DOWN, PLAYER_B_NOOP);
        environment.act(PLAYER_A_NOOP, PLAYER_B_NOOP);
        for(unsigned i = 0; i < 4; i++){
            cout << (m& (1 << i)) << endl;
            if((m & (1 << i)) != 0){ //test if the ith bit is set
                environment.act(PLAYER_A_RIGHT, PLAYER_B_NOOP);
                cout << "setting " << i << endl;
            }else{
                environment.act(PLAYER_A_LEFT, PLAYER_B_NOOP);
            }
            environment.act(PLAYER_A_NOOP, PLAYER_B_NOOP);
            environment.act(PLAYER_A_DOWN, PLAYER_B_NOOP);
            environment.act(PLAYER_A_NOOP, PLAYER_B_NOOP);
        }
        //reset the environment to apply changes.
        environment.soft_reset();
        //apply starting action
        environment.act(PLAYER_A_UPFIRE, PLAYER_B_NOOP);
        environment.act(PLAYER_A_NOOP, PLAYER_B_NOOP);
    }else{
        throw std::runtime_error("This mode doesn't currently exist for this game");
    }
}
