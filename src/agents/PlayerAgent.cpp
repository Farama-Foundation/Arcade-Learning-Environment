/* *****************************************************************************
 * A.L.E (Arcade Learning Environment)
 * Copyright (c) 2009-2012 by Yavar Naddaf, Joel Veness, Marc G. Bellemare and 
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
#include "game_controller.h"

/* **********************************************************************
    Constructor
 ********************************************************************* */
PlayerAgent::PlayerAgent(OSystem* _osystem, RomSettings* _settings) :
    p_osystem(_osystem), p_rom_settings(_settings),
    frame_number(0), episode_frame_number(0), episode_number(0),
    available_actions(_settings->getAvailableActions()),
    m_has_terminated(false), manual_control(false) {
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

  bool display_screen = settings.getBool("display_screen", false);
  if (display_screen) {
      p_osystem->p_display_screen->registerEventHandler(this);
  }
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
  if (manual_control) {
      a = waitForKeypress();
  } else {
      a = act();
  }

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
    Note 2: The caller is resposible for deleting the returned pointer
******************************************************************** */
PlayerAgent* PlayerAgent::generate_agent_instance(OSystem* _osystem,
                                                RomSettings * _settings) {
    string player_agent = _osystem->settings().getString("player_agent");
    PlayerAgent* new_agent = NULL;

    if (player_agent == "random_agent")
      new_agent = new RandomAgent(_osystem, _settings);
    else if (player_agent == "single_action_agent")
      new_agent = new SingleActionAgent(_osystem, _settings);
    else
      new_agent = NULL;

    return new_agent;
}


bool PlayerAgent::handleSDLEvent(const SDL_Event& event) {
    switch(event.type) {
    case SDL_KEYDOWN:
        switch(event.key.keysym.sym) {
        case SDLK_p:
            if (manual_control) {
                // Set the pause status to whatever it previously was
                p_osystem->p_display_screen->setPaused(returnToPause);
                cout << "Returning to Automatic Control." << endl;
            } else {
                // Capture the pause status and store
                returnToPause = p_osystem->p_display_screen->paused;
                cout << "ReturntoPause: " << returnToPause << endl;
                p_osystem->p_display_screen->setPaused(false);
                printf("Starting Manual Control. Commands are as follows:\n  -p: return to auto control\n  -arrow keys: joystick movement\n  -space: button/fire\n  -return: no-op\n");
            }
            manual_control = !manual_control;
            return true;
        default:
            // If a key is pressed while manual control is active, let the manual controller handle it
            if (manual_control) return true;
            break;
        }
    default:
        break;
    }
    return false;
}

void PlayerAgent::usage() {
    printf("  -p: Toggle manual control of the agent\n");
}

Action PlayerAgent::waitForKeypress() {
    Action a = UNDEFINED;
    // This loop is necessary because keypress events come in quickly
    while (a == UNDEFINED) {
        SDL_Delay(50); // Set amount of sleep time
        SDL_PumpEvents();
        Uint8* keymap = SDL_GetKeyState(NULL);

        // Break out of this loop if the 'p' key is pressed
        if (keymap[SDLK_p]) {
            return PLAYER_A_NOOP;

            // Triple Actions
        } else if (keymap[SDLK_UP] && keymap[SDLK_RIGHT] && keymap[SDLK_SPACE]) {
            a = PLAYER_A_UPRIGHTFIRE;
        } else if (keymap[SDLK_UP] && keymap[SDLK_LEFT] && keymap[SDLK_SPACE]) {
            a = PLAYER_A_UPLEFTFIRE;
        } else if (keymap[SDLK_DOWN] && keymap[SDLK_RIGHT] && keymap[SDLK_SPACE]) {
            a = PLAYER_A_DOWNRIGHTFIRE;
        } else if (keymap[SDLK_DOWN] && keymap[SDLK_LEFT] && keymap[SDLK_SPACE]) {
            a = PLAYER_A_DOWNLEFTFIRE;

            // Double Actions
        } else if (keymap[SDLK_UP] && keymap[SDLK_LEFT]) {
            a = PLAYER_A_UPLEFT;
        } else if (keymap[SDLK_UP] && keymap[SDLK_RIGHT]) {
            a = PLAYER_A_UPRIGHT;
        } else if (keymap[SDLK_DOWN] && keymap[SDLK_LEFT]) {
            a = PLAYER_A_DOWNLEFT;
        } else if (keymap[SDLK_DOWN] && keymap[SDLK_RIGHT]) {
            a = PLAYER_A_DOWNRIGHT;
        } else if (keymap[SDLK_UP] && keymap[SDLK_SPACE]) {
            a = PLAYER_A_UPFIRE;
        } else if (keymap[SDLK_DOWN] && keymap[SDLK_SPACE]) {
            a = PLAYER_A_DOWNFIRE;
        } else if (keymap[SDLK_LEFT] && keymap[SDLK_SPACE]) {
            a = PLAYER_A_LEFTFIRE;
        } else if (keymap[SDLK_RIGHT] && keymap[SDLK_SPACE]) {
            a = PLAYER_A_RIGHTFIRE;

            // Single Actions
        } else if (keymap[SDLK_SPACE]) {
            a = PLAYER_A_FIRE;
        } else if (keymap[SDLK_RETURN]) {
            a = PLAYER_A_NOOP;
        } else if (keymap[SDLK_LEFT]) {
            a = PLAYER_A_LEFT;
        } else if (keymap[SDLK_RIGHT]) {
            a = PLAYER_A_RIGHT;
        } else if (keymap[SDLK_UP]) {
            a = PLAYER_A_UP;
        } else if (keymap[SDLK_DOWN]) {
            a = PLAYER_A_DOWN;
        } 
    }
    return a;
}
