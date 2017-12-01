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


DoubleDunkSettings::DoubleDunkSettings() {

    reset();
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
void DoubleDunkSettings::reset() {
    
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
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
ModeVect DoubleDunkSettings::getAvailableModes() {
    // this game has a menu that allows to define various yes/no options
    // setting these options define in a way a different mode
    // there are 4 relevant options, which makes 2^4=16 available modes
    ModeVect modes(getNumModes());
    for (unsigned int i = 0; i < modes.size(); i++) {
        modes[i] = i;
    }
    return modes;
}

void DoubleDunkSettings::goDown(System &system,
                            std::unique_ptr<StellaEnvironmentWrapper> &environment) {
    // this game has a menu that allows to define various yes/no options
    // this function goes to the next option in the menu
    unsigned int previousSelection = readRam(&system, 0xB0);
    while(previousSelection == readRam(&system, 0xB0)){
        environment->act(PLAYER_A_DOWN, PLAYER_B_NOOP);
        environment->act(PLAYER_A_NOOP, PLAYER_B_NOOP);
    }
}

void DoubleDunkSettings::activateOption(System &system, unsigned int bitOfInterest,
                                    std::unique_ptr<StellaEnvironmentWrapper> &environment) {
    // once we are at the proper option in the menu,
    // if we want to enable it all we have to do is to go right
    while((readRam(&system, 0x80) & bitOfInterest) != bitOfInterest) {
        environment->act(PLAYER_A_RIGHT, PLAYER_B_NOOP);
        environment->act(PLAYER_A_NOOP, PLAYER_B_NOOP);
    }
}

void DoubleDunkSettings::deactivateOption(System &system, unsigned int bitOfInterest,
                                    std::unique_ptr<StellaEnvironmentWrapper> &environment) {
    // once we are at the proper optio in the menu,
    // if we want to disable it all we have to do is to go left
    while((readRam(&system, 0x80) & bitOfInterest) == bitOfInterest) {
        environment->act(PLAYER_A_LEFT, PLAYER_B_NOOP);
        environment->act(PLAYER_A_NOOP, PLAYER_B_NOOP);
    }
}

// set the mode of the game
// the given mode must be one returned by the previous function
void DoubleDunkSettings::setMode(game_mode_t m, System &system,
                                std::unique_ptr<StellaEnvironmentWrapper> environment) {

    if(m < getNumModes()) {
        environment->pressSelect();

        //discard the first two entries (irrelevant)
        goDown(system, environment);
        goDown(system, environment);

        //deal with the 3 points option
        if(m & 1) {
            activateOption(system, 0x08, environment);
        } else {
            deactivateOption(system, 0x08, environment);
        }

        //deal with the 10 seconds option
        goDown(system, environment);
        if(m & 2) {
            activateOption(system, 0x10, environment);
        } else {
            deactivateOption(system, 0x10, environment);   
        }

        //deal with the 3 seconds option
        goDown(system, environment);
        if(m & 4) {
            activateOption(system, 0x04, environment);
        } else {
            deactivateOption(system, 0x04, environment);   
        }

        //deal with the foul option
        goDown(system, environment);
        if(m & 8) {
            activateOption(system, 0x20, environment);
        } else {
            deactivateOption(system, 0x20, environment);   
        }

        //reset the environment to apply changes.
        environment->softReset();
        //apply starting action
        environment->act(PLAYER_A_UPFIRE, PLAYER_B_NOOP);
        environment->act(PLAYER_A_NOOP, PLAYER_B_NOOP);
        
    }
    else {
        throw std::runtime_error("This mode doesn't currently exist for this game");
    }
 }

