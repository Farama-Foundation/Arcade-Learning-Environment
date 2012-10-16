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
 */
#include "ALEState.hpp"
#include "System.hxx"
#include "Event.hxx"

int ALEState::left_paddle_curr_x = PADDLE_DEFAULT_VALUE; 
int ALEState::right_paddle_curr_x = PADDLE_DEFAULT_VALUE;

/** Default constructor - loads settings from system */ 
ALEState::ALEState(OSystem * osystem): m_osystem(osystem), m_settings(NULL) {
  if (osystem->console().properties().get(Controller_Left) == "PADDLES" ||
    osystem->console().properties().get(Controller_Right) == "PADDLES") {
      uses_paddles = true;
      default_paddles();
    } else {
      uses_paddles = false;
    }
 
  frame_number = 0;
  s_cartridge_md5 = m_osystem->console().properties().get(Cartridge_MD5);

  m_use_starting_actions = m_osystem->settings().getBool("use_starting_actions");
}

/** Copy constructor - copy everything */
ALEState::ALEState(ALEState & _state): 
  m_osystem(_state.m_osystem),
  m_settings(_state.m_settings),
  serialized(_state.serialized),
  s_cartridge_md5(_state.s_cartridge_md5), 
  frame_number(_state.frame_number),
  uses_paddles(_state.uses_paddles) 
{
  m_use_starting_actions = m_osystem->settings().getBool("use_starting_actions");
}

void ALEState::setSettings(RomSettings * settings) {
  m_settings = settings;
}

/** Resets ALE (emulator and ROM settings) to the state described by
  * this object. */
void ALEState::load() {
  assert(serialized.length() > 0);
  assert(m_settings != NULL);
  Deserializer deser(serialized);
  
  m_osystem->console().system().loadState(s_cartridge_md5, deser);
  m_settings->loadState(deser);
  
  left_paddle_curr_x = deser.getInt();
  right_paddle_curr_x = deser.getInt();
  frame_number = deser.getInt();
}

void ALEState::save() {
  assert(m_settings != NULL);
  Serializer ser;
  
  m_osystem->console().system().saveState(s_cartridge_md5, ser);
  m_settings->saveState(ser);
  
  ser.putInt(left_paddle_curr_x);
  ser.putInt(right_paddle_curr_x);
  ser.putInt(frame_number);

  serialized = ser.get_str();
}

void ALEState::reset(int numResetSteps) {
  // reset the rom
  m_settings->reset();
  
  // Reset the paddles
  default_paddles();

  // Reset the emulator
  m_osystem->console().system().reset();

  // NOOP for 60 steps
  for (int i = 0; i < 60; i++) {
    apply_action(PLAYER_A_NOOP, PLAYER_B_NOOP);
    simulate();
    // Don't count these frames
    frame_number--;
  }

  // reset for n steps
  for (int i = 0; i < numResetSteps; i++) {
    apply_action(RESET, PLAYER_B_NOOP);
    simulate();
    // Don't count these frames
    frame_number--;
  }

  // Apply necessary actions specified by the rom itself
  if (m_use_starting_actions) {
    ActionVect startingActions = m_settings->getStartingActions();
    for (size_t i = 0; i < startingActions.size(); i++) {
      apply_action(startingActions[i], PLAYER_B_NOOP);
      simulate();
      frame_number--;
    }
  }
}

/* ***************************************************************************
 *  Calculates the Paddle resistance, based on the given x val
 * ***************************************************************************/
int ALEState::calc_paddle_resistance(int x_val) {
  return x_val;  // this is different from the original stella implemebtation
}

void ALEState::default_paddles() {
  set_paddles(PADDLE_DEFAULT_VALUE, PADDLE_DEFAULT_VALUE);
}

void ALEState::set_paddles(int left, int right) {
  left_paddle_curr_x = left; 
  right_paddle_curr_x = right;
 
  Event * event = m_osystem->event();

  int left_resistance = calc_paddle_resistance(left_paddle_curr_x);
    event->set(Event::PaddleZeroResistance, left_resistance);
  int right_resistance = calc_paddle_resistance(right_paddle_curr_x);
    event->set(Event::PaddleOneResistance, right_resistance);
}

/* *********************************************************************
 *  Updates the positions of the paddles, and sets an event for 
 *  updating the corresponding paddle's resistance
 * ********************************************************************/
 void ALEState::update_paddles_positions(int delta_left, int delta_right) {
    Event * event = m_osystem->event();

    left_paddle_curr_x += delta_left;
    if (left_paddle_curr_x < PADDLE_MIN) {
        left_paddle_curr_x = PADDLE_MIN;
    } 
    if (left_paddle_curr_x >  PADDLE_MAX) {
        left_paddle_curr_x = PADDLE_MAX;
    }
    int left_resistance = calc_paddle_resistance(left_paddle_curr_x);
    
    event->set(Event::PaddleZeroResistance, left_resistance);
    right_paddle_curr_x += delta_right;
    if (right_paddle_curr_x < PADDLE_MIN) {
        right_paddle_curr_x = PADDLE_MIN;
    } 
    if (right_paddle_curr_x >  PADDLE_MAX) {
        right_paddle_curr_x = PADDLE_MAX;
    }
    int right_resistance = calc_paddle_resistance(right_paddle_curr_x);
    event->set(Event::PaddleOneResistance, right_resistance);
}


/* ***************************************************************************
 *  Function apply_action
 *  Applies the actions recieved from the controller for player A and B
 * ***************************************************************************/
void ALEState::apply_action(Action player_a_action, Action player_b_action) {
  // Convert illegal actions into NOOPs; actions such as reset are always legal
  if (player_a_action < (Action)PLAYER_B_NOOP && !m_settings->isLegal(player_a_action))
    player_a_action = (Action)PLAYER_A_NOOP;
  if (player_b_action < (Action)RESET && !m_settings->isLegal((Action)((int)player_b_action - PLAYER_B_NOOP)))
    player_b_action = (Action)PLAYER_B_NOOP;

  // Set keys
  reset_keys(m_osystem->event());

  if (uses_paddles)
    apply_action_paddles(m_osystem->event(), player_a_action, player_b_action);
  else
    apply_action_joysticks(m_osystem->event(), player_a_action, player_b_action);
}

void ALEState::simulate() {
  // Simulate forward
  m_osystem->console().mediaSource().update();
  m_settings->step(m_osystem->console().system());
  frame_number++;
}

void ALEState::apply_action_paddles(Event* event, 
                                    int player_a_action, int player_b_action) {
  // First compute whether we should increase or decrease the paddle position
  //  (for both left and right players)
  int delta_left;
  int delta_right;

    switch(player_a_action) {
        case PLAYER_A_RIGHT: 
        case PLAYER_A_RIGHTFIRE: 
        case PLAYER_A_UPRIGHT: 
        case PLAYER_A_DOWNRIGHT: 
        case PLAYER_A_UPRIGHTFIRE: 
        case PLAYER_A_DOWNRIGHTFIRE: 
          delta_left = -PADDLE_DELTA;   
            break; 
            
        case PLAYER_A_LEFT:
        case PLAYER_A_LEFTFIRE: 
        case PLAYER_A_UPLEFT: 
        case PLAYER_A_DOWNLEFT: 
        case PLAYER_A_UPLEFTFIRE: 
        case PLAYER_A_DOWNLEFTFIRE: 
      delta_left = PADDLE_DELTA;
            break;
    default:
      delta_left = 0;
      break;
    }

    switch(player_b_action) {
        case PLAYER_B_RIGHT: 
        case PLAYER_B_RIGHTFIRE: 
        case PLAYER_B_UPRIGHT: 
        case PLAYER_B_DOWNRIGHT: 
        case PLAYER_B_UPRIGHTFIRE: 
        case PLAYER_B_DOWNRIGHTFIRE: 
          delta_right = -PADDLE_DELTA;
            break; 
            
        case PLAYER_B_LEFT:
        case PLAYER_B_LEFTFIRE: 
        case PLAYER_B_UPLEFT: 
        case PLAYER_B_DOWNLEFT: 
        case PLAYER_B_UPLEFTFIRE: 
        case PLAYER_B_DOWNLEFTFIRE: 
      delta_right = PADDLE_DELTA;
            break;
    default:
      delta_right = 0;
      break;
    }

  // Now update the paddle positions
  update_paddles_positions(delta_left, delta_right);

  // Handle reset
  if (player_a_action == RESET || player_b_action == RESET)
    event->set(Event::ConsoleReset, 1);

  // Now add the fire event 
  switch (player_a_action) {
        case PLAYER_A_FIRE: 
        case PLAYER_A_UPFIRE: 
        case PLAYER_A_RIGHTFIRE: 
        case PLAYER_A_LEFTFIRE: 
        case PLAYER_A_DOWNFIRE: 
        case PLAYER_A_UPRIGHTFIRE: 
        case PLAYER_A_UPLEFTFIRE: 
        case PLAYER_A_DOWNRIGHTFIRE: 
        case PLAYER_A_DOWNLEFTFIRE: 
            event->set(Event::PaddleZeroFire, 1);
            break;
    default:
      // Nothing
      break;
  }
  
  switch (player_b_action) {
        case PLAYER_B_FIRE: 
        case PLAYER_B_UPFIRE: 
        case PLAYER_B_RIGHTFIRE: 
        case PLAYER_B_LEFTFIRE: 
        case PLAYER_B_DOWNFIRE: 
        case PLAYER_B_UPRIGHTFIRE: 
        case PLAYER_B_UPLEFTFIRE: 
        case PLAYER_B_DOWNRIGHTFIRE: 
        case PLAYER_B_DOWNLEFTFIRE: 
            event->set(Event::PaddleOneFire, 1);
            break;
    default:
      // Nothing
      break;
  }
}

void ALEState::apply_action_joysticks(Event* event, 
                                    int player_a_action, int player_b_action) {
    switch(player_a_action) {
        case PLAYER_A_NOOP: 
            break; 
            
        case PLAYER_A_FIRE: 
            event->set(Event::JoystickZeroFire, 1);
            break; 
            
        case PLAYER_A_UP: 
            event->set(Event::JoystickZeroUp, 1);
            break; 
            
        case PLAYER_A_RIGHT: 
            event->set(Event::JoystickZeroRight, 1);
            break; 
            
        case PLAYER_A_LEFT: 
            event->set(Event::JoystickZeroLeft, 1);
            break; 
            
        case PLAYER_A_DOWN: 
            event->set(Event::JoystickZeroDown, 1);
            break; 
            
        case PLAYER_A_UPRIGHT: 
            event->set(Event::JoystickZeroUp, 1);
            event->set(Event::JoystickZeroRight, 1);
            break; 
            
        case PLAYER_A_UPLEFT: 
            event->set(Event::JoystickZeroUp, 1);
            event->set(Event::JoystickZeroLeft, 1);
            break; 
            
        case PLAYER_A_DOWNRIGHT: 
            event->set(Event::JoystickZeroDown, 1);
            event->set(Event::JoystickZeroRight, 1);
            break; 
            
        case PLAYER_A_DOWNLEFT: 
            event->set(Event::JoystickZeroDown, 1);
            event->set(Event::JoystickZeroLeft, 1);
            break; 
            
        case PLAYER_A_UPFIRE: 
            event->set(Event::JoystickZeroUp, 1);
            event->set(Event::JoystickZeroFire, 1);
            break; 
            
        case PLAYER_A_RIGHTFIRE: 
            event->set(Event::JoystickZeroRight, 1);
            event->set(Event::JoystickZeroFire, 1); 
            break; 
            
        case PLAYER_A_LEFTFIRE: 
            event->set(Event::JoystickZeroLeft, 1);
            event->set(Event::JoystickZeroFire, 1); 
            break; 
            
        case PLAYER_A_DOWNFIRE: 
            event->set(Event::JoystickZeroDown, 1);
            event->set(Event::JoystickZeroFire, 1);
            break; 
            
        case PLAYER_A_UPRIGHTFIRE: 
            event->set(Event::JoystickZeroUp, 1);
            event->set(Event::JoystickZeroRight, 1);
            event->set(Event::JoystickZeroFire, 1);
            break; 
            
        case PLAYER_A_UPLEFTFIRE: 
            event->set(Event::JoystickZeroUp, 1);
            event->set(Event::JoystickZeroLeft, 1);
            event->set(Event::JoystickZeroFire, 1); 
            break; 
            
        case PLAYER_A_DOWNRIGHTFIRE: 
            event->set(Event::JoystickZeroDown, 1);
            event->set(Event::JoystickZeroRight, 1);
            event->set(Event::JoystickZeroFire, 1); 
            break; 
            
        case PLAYER_A_DOWNLEFTFIRE: 
            event->set(Event::JoystickZeroDown, 1);
            event->set(Event::JoystickZeroLeft, 1);
            event->set(Event::JoystickZeroFire, 1);
            break; 
        case RESET:
            event->set(Event::ConsoleReset, 1);
            break;
        default: 
            cerr << "Invalid Player A Action: " << player_a_action;
            exit(-1); 
        
    }

    switch(player_b_action) {
    case PLAYER_B_NOOP: 
            break; 
            
        case PLAYER_B_FIRE: 
            event->set(Event::JoystickOneFire, 1);
            break; 
            
        case PLAYER_B_UP: 
            event->set(Event::JoystickOneUp, 1);
            break; 
            
        case PLAYER_B_RIGHT: 
            event->set(Event::JoystickOneRight, 1);
            break; 
            
        case PLAYER_B_LEFT: 
            event->set(Event::JoystickOneLeft, 1);
            break; 
            
        case PLAYER_B_DOWN: 
            event->set(Event::JoystickOneDown, 1);
            break; 
            
        case PLAYER_B_UPRIGHT: 
            event->set(Event::JoystickOneUp, 1);
            event->set(Event::JoystickOneRight, 1);
            break; 
            
        case PLAYER_B_UPLEFT: 
            event->set(Event::JoystickOneUp, 1);
            event->set(Event::JoystickOneLeft, 1);
            break; 
            
        case PLAYER_B_DOWNRIGHT: 
            event->set(Event::JoystickOneDown, 1);
            event->set(Event::JoystickOneRight, 1);
            break; 
            
        case PLAYER_B_DOWNLEFT: 
            event->set(Event::JoystickOneDown, 1);
            event->set(Event::JoystickOneLeft, 1);
            break; 
            
        case PLAYER_B_UPFIRE: 
            event->set(Event::JoystickOneUp, 1);
            event->set(Event::JoystickOneFire, 1);
            break; 
            
        case PLAYER_B_RIGHTFIRE: 
            event->set(Event::JoystickOneRight, 1);
            event->set(Event::JoystickOneFire, 1); 
            break; 
            
        case PLAYER_B_LEFTFIRE: 
            event->set(Event::JoystickOneLeft, 1);
            event->set(Event::JoystickOneFire, 1); 
            break; 
            
        case PLAYER_B_DOWNFIRE: 
            event->set(Event::JoystickOneDown, 1);
            event->set(Event::JoystickOneFire, 1);
            break; 
            
        case PLAYER_B_UPRIGHTFIRE: 
            event->set(Event::JoystickOneUp, 1);
            event->set(Event::JoystickOneRight, 1);
            event->set(Event::JoystickOneFire, 1);
            break; 
            
        case PLAYER_B_UPLEFTFIRE: 
            event->set(Event::JoystickOneUp, 1);
            event->set(Event::JoystickOneLeft, 1);
            event->set(Event::JoystickOneFire, 1); 
            break; 
            
        case PLAYER_B_DOWNRIGHTFIRE: 
            event->set(Event::JoystickOneDown, 1);
            event->set(Event::JoystickOneRight, 1);
            event->set(Event::JoystickOneFire, 1); 
            break; 
            
        case PLAYER_B_DOWNLEFTFIRE: 
            event->set(Event::JoystickOneDown, 1);
            event->set(Event::JoystickOneLeft, 1);
            event->set(Event::JoystickOneFire, 1);
            break; 
        case RESET:
            event->set(Event::ConsoleReset, 1);
            cerr << "Sending Reset..." << endl;
            break;
        default: 
            cerr << "Invalid Player B Action: " << player_b_action << endl;
            exit(-1); 
    }
}

/* ***************************************************************************
    Function reset_keys
    Unpresses all control-relavant keys
 * ***************************************************************************/
void ALEState::reset_keys(Event* event) {
    event->set(Event::ConsoleReset, 0);
    event->set(Event::JoystickZeroFire, 0);
    event->set(Event::JoystickZeroUp, 0);
    event->set(Event::JoystickZeroDown, 0);
    event->set(Event::JoystickZeroRight, 0);
    event->set(Event::JoystickZeroLeft, 0);
    event->set(Event::JoystickOneFire, 0);
    event->set(Event::JoystickOneUp, 0);
    event->set(Event::JoystickOneDown, 0);
    event->set(Event::JoystickOneRight, 0);
    event->set(Event::JoystickOneLeft, 0);
    
    // also reset paddle fire
    event->set(Event::PaddleZeroFire, 0);
    event->set(Event::PaddleOneFire, 0);
}

bool ALEState::equals(ALEState &state) {
  return (state.serialized == this->serialized);
}
