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
 *  PlayerAgent.cpp
 *
 * The implementation of the PlayerAgent abstract class
 **************************************************************************** */
#include <sstream>
#include "PlayerAgent.hpp"
#include "RandomAgent.hpp"
#include "SingleActionAgent.hpp"
#include "SDLKeyboardAgent.hpp"

/* **********************************************************************
    Constructor
 ********************************************************************* */
PlayerAgent::PlayerAgent(OSystem* _osystem, RomSettings* _settings) :
    p_osystem(_osystem), p_rom_settings(_settings),
    frame_number(0), episode_frame_number(0), episode_number(0),
    available_actions(_settings->getMinimalActionSet()),
    m_has_terminated(false) { 
  Settings& settings = p_osystem->settings();

  i_max_num_episodes = settings.getInt("max_num_episodes", true);
  i_max_num_frames = settings.getInt("max_num_frames", true);
  i_max_num_frames_per_episode = 
    settings.getInt("max_num_frames_per_episode", true);

  // By default this is false
  record_trajectory = settings.getBool("record_trajectory", false);

  // Default: false (not currently implemented)
  bool use_restricted_action_set = 
    settings.getBool("restricted_action_set", false);

  if (!use_restricted_action_set)
    available_actions = _settings->getAllActions();
}

/* **********************************************************************
    Destructor
 ********************************************************************* */
PlayerAgent::~PlayerAgent() {
}

/** This methods handles the basic agent functionality: bookeeping
  *  the number of episodes, frames, etc... It calls the method 
  *  act(), which will provide it with an action. act() which should 
  *  be overriden by subclasses.
  */
Action PlayerAgent::agent_step() {
  // Terminate if we have a maximum number of frames
  if (i_max_num_frames > 0 && frame_number >= i_max_num_frames)
    end_game();

  // Terminate this episode if we have reached the max. number of frames 
  if (i_max_num_frames_per_episode > 0 && 
      episode_frame_number >= i_max_num_frames_per_episode) {
    return RESET;
  }

  // Only take an action if manual control is disabled
  Action a;
  a = act();

  if (record_trajectory) record_action(a);

  frame_number++;
  episode_frame_number++;
  
  return a;
}

/* *********************************************************************
    This method is called when the game ends. The superclass 
    implementation takes care of counting number of episodes, and 
    saving the reward history. Any other update should be taken care of
    in the derived classes
******************************************************************** */
void PlayerAgent::episode_end(void) {
  episode_number++;

  if (i_max_num_episodes > 0 && episode_number >= i_max_num_episodes)
    end_game();
}

Action PlayerAgent::episode_start(void) {
  episode_frame_number = 0;

  Action a = act();
  if (record_trajectory) record_action(a);

  frame_number++;
  episode_frame_number++;

  return a;
}

void PlayerAgent::record_action(Action a) {
  if (episode_number == 0)
    trajectory.push_back(a);
}

void PlayerAgent::end_game() {
  // Post-processing
  if (record_trajectory) {
    cout << "Trajectory "; 
    for (size_t i = 0; i < trajectory.size(); i++) {
      cout << trajectory[i] << " ";
    }
    cout << "\n";
  }
  
  m_has_terminated = true;
}

bool PlayerAgent::has_terminated() {
  return m_has_terminated;
}

/* *********************************************************************
    Generates an instance of one of the PlayerAgent subclasses, based on
    "player_agent" value in the settings.  
    Returns a NULL pointer if the value of player_agent is invalid.
    Note 1: If you add a new player_agent subclass, you need to also 
            add it here
    Note 2: The caller is responsible for deleting the returned pointer
******************************************************************** */
PlayerAgent* PlayerAgent::generate_agent_instance(OSystem* _osystem,
                                                RomSettings * _settings) {
    string player_agent = _osystem->settings().getString("player_agent");
    PlayerAgent* new_agent = NULL;

    if (player_agent == "random_agent")
      new_agent = new RandomAgent(_osystem, _settings);
    else if (player_agent == "single_action_agent")
      new_agent = new SingleActionAgent(_osystem, _settings);
    else if (player_agent == "keyboard_agent")
      new_agent = new SDLKeyboardAgent(_osystem, _settings);
    else {
      std::cerr << "Invalid agent type requested: " << player_agent << ". Terminating." << std::endl;
      // We can't play without any agent, so exit now.
      exit(-1);
    }

    return new_agent;
}

