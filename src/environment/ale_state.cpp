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

#include "ale_state.hpp"

#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>

#include "../emucore/m6502/src/System.hxx"
#include "../emucore/Event.hxx"
#include "../emucore/Deserializer.hxx"
#include "../emucore/Serializer.hxx"
#include "../common/Constants.h"
#include "../games/RomSettings.hpp"

namespace ale {

/** Default constructor - loads settings from system */
ALEState::ALEState()
    : m_paddle_min(PADDLE_MIN),
      m_paddle_max(PADDLE_MAX),
      m_frame_number(0),
      m_episode_frame_number(0),
      m_mode(0),
      m_difficulty(0) {
        for(int i = 0; i < 4; i++){
          m_paddle[i] = PADDLE_DEFAULT_VALUE;
        }
      }

ALEState::ALEState(const ALEState& rhs, const std::string& serialized)
    : m_paddle_min(rhs.m_paddle_min),
      m_paddle_max(rhs.m_paddle_max),
      m_frame_number(rhs.m_frame_number),
      m_episode_frame_number(rhs.m_episode_frame_number),
      m_serialized_state(serialized),
      m_mode(rhs.m_mode),
      m_difficulty(rhs.m_difficulty) {
        for(int i = 0; i < 4; i++){
          m_paddle[i] = rhs.m_paddle[i];
        }
      }

ALEState::ALEState(const std::string& serialized) {
  Deserializer des(serialized);
  for(int i = 0;i < 4; i++){
    this->m_paddle[i] = des.getInt();
  }
  this->m_frame_number = des.getInt();
  this->m_episode_frame_number = des.getInt();
  this->m_mode = des.getInt();
  this->m_difficulty = des.getInt();
  this->m_serialized_state = des.getString();
  this->m_paddle_min = des.getInt();
  this->m_paddle_max = des.getInt();
}

/** Restores ALE to the given previously saved state. */
void ALEState::load(OSystem* osystem, RomSettings* settings, std::string md5,
                    const ALEState& rhs, bool load_system) {
  assert(rhs.m_serialized_state.length() > 0);

  // Deserialize the stored string into the emulator state
  Deserializer deser(rhs.m_serialized_state);

  // A primitive check to produce a meaningful error if this state does not contain osystem info.
  if (deser.getBool() != load_system)
    throw new std::runtime_error("Attempting to load an ALEState which does "
                                 "not contain system information.");

  osystem->console().system().loadState(md5, deser);
  // If we have osystem data, load it as well
  if (load_system)
    osystem->loadState(deser);
  settings->loadState(deser);

  // Copy over other member variables
  for(int i = 0; i < 4; i++){
    m_paddle[i] = rhs.m_paddle[i];
  }
  m_paddle_min = rhs.m_paddle_min;
  m_paddle_max = rhs.m_paddle_max;
  m_frame_number = rhs.m_frame_number;
  m_episode_frame_number = rhs.m_episode_frame_number;
  m_mode = rhs.m_mode;
  m_difficulty = rhs.m_difficulty;
}

ALEState ALEState::save(OSystem* osystem, RomSettings* settings,
                        std::string md5, bool save_system) {
  // Use the emulator's built-in serialization to save the state
  Serializer ser;

  // We use 'save_system' as a check at load time.
  ser.putBool(save_system);

  osystem->console().system().saveState(md5, ser);
  if (save_system)
    osystem->saveState(ser);
  settings->saveState(ser);

  // Now make a copy of this state, also storing the emulator serialization
  return ALEState(*this, ser.get_str());
}

void ALEState::incrementFrame(int steps /* = 1 */) {
  m_frame_number += steps;
  m_episode_frame_number += steps;
}

void ALEState::resetEpisodeFrameNumber() { m_episode_frame_number = 0; }

std::string ALEState::serialize() {
  Serializer ser;

  for(int i = 0;i < 4; i++){
    ser.putInt(this->m_paddle[i]);
  }
  ser.putInt(this->m_frame_number);
  ser.putInt(this->m_episode_frame_number);
  ser.putInt(this->m_mode);
  ser.putInt(this->m_difficulty);
  ser.putString(this->m_serialized_state);
  ser.putInt(this->m_paddle_min);
  ser.putInt(this->m_paddle_max);

  return ser.get_str();
}

/* ***************************************************************************
 *  Calculates the Paddle resistance, based on the given x val
 * ***************************************************************************/
int ALEState::calcPaddleResistance(int x_val) {
  return x_val; // this is different from the original stella implemebtation
}

void ALEState::resetPaddles(Event* event) {
  int paddle_default = (m_paddle_min + m_paddle_max) / 2;
  for(int i = 0; i < 4; i++){
    setPaddles(event, paddle_default, i);
  }
}

void ALEState::setPaddles(Event* event, int paddle_val, int paddle_num) {
  m_paddle[paddle_num] = paddle_val;

  // Compute the "resistance" (this is for vestigal clarity)
  int resitance = calcPaddleResistance(paddle_val);

  Event::Type paddle_resists[] = {
    Event::PaddleZeroResistance,
    Event::PaddleOneResistance,
    Event::PaddleTwoResistance,
    Event::PaddleThreeResistance
  };
  // Update the events with the new resistances
  event->set(paddle_resists[paddle_num], resitance);
}

void ALEState::setPaddleLimits(int paddle_min_val, int paddle_max_val) {
  m_paddle_min = paddle_min_val;
  m_paddle_max = paddle_max_val;
  // Don't update paddle positions as this will send an event. Wait for next
  // paddle update and the positions will be clamped to the new min/max.
}

/* *********************************************************************
 *  Updates the positions of the paddles, and sets an event for
 *  updating the corresponding paddle's resistance
 * ********************************************************************/
void ALEState::updatePaddlePositions(Event* event, int delta,
                                     int paddle_num) {
  // Cap paddle outputs

  m_paddle[paddle_num] += delta;
  if (m_paddle[paddle_num] < m_paddle_min) {
    m_paddle[paddle_num] = m_paddle_min;
  }
  if (m_paddle[paddle_num] > m_paddle_max) {
    m_paddle[paddle_num] = m_paddle_max;
  }

  // Now set the paddle to their new value
  setPaddles(event, m_paddle[paddle_num], paddle_num);
}

void ALEState::applyActionPaddles(Event* event, int action, int pnum) {
  // Reset keys
  //resetKeys(event);
  // if(action >= 18)
  // std::cout << action << "\n";
  // std::cout << pnum << "\n";
  // First compute whether we should increase or decrease the paddle position
  //  (for both left and right players)
  int delta;

  switch (action) {
    case PLAYER_A_RIGHT:
    case PLAYER_A_RIGHTFIRE:
    case PLAYER_A_UPRIGHT:
    case PLAYER_A_DOWNRIGHT:
    case PLAYER_A_UPRIGHTFIRE:
    case PLAYER_A_DOWNRIGHTFIRE:
      delta = -PADDLE_DELTA;
      break;

    case PLAYER_A_LEFT:
    case PLAYER_A_LEFTFIRE:
    case PLAYER_A_UPLEFT:
    case PLAYER_A_DOWNLEFT:
    case PLAYER_A_UPLEFTFIRE:
    case PLAYER_A_DOWNLEFTFIRE:
      delta = PADDLE_DELTA;
      break;
    default:
      delta = 0;
      break;
  }

  // Now update the paddle positions
  updatePaddlePositions(event, delta, pnum);

  // Handle reset
  if (action == RESET)
    event->set(Event::ConsoleReset, 1);

  Event::Type paddle_fires[] = {
    Event::PaddleZeroFire,
    Event::PaddleOneFire,
    Event::PaddleTwoFire,
    Event::PaddleThreeFire
  };
  // Now add the fire event
  switch (action) {
    case PLAYER_A_FIRE:
    case PLAYER_A_UPFIRE:
    case PLAYER_A_RIGHTFIRE:
    case PLAYER_A_LEFTFIRE:
    case PLAYER_A_DOWNFIRE:
    case PLAYER_A_UPRIGHTFIRE:
    case PLAYER_A_UPLEFTFIRE:
    case PLAYER_A_DOWNRIGHTFIRE:
    case PLAYER_A_DOWNLEFTFIRE:
      event->set(paddle_fires[pnum], 1);
      break;
    default:
      // Nothing
      break;
  }
}

void ALEState::pressSelect(Event* event) {
  resetKeys(event);
  event->set(Event::ConsoleSelect, 1);
}

void ALEState::setDifficultySwitches(Event* event, unsigned int value) {
  // The difficulty switches stay in their position from time step to time step.
  // This means we don't call resetKeys() when setting their values.
  event->set(Event::ConsoleLeftDifficultyA, value & 1);
  event->set(Event::ConsoleLeftDifficultyB, !(value & 1));
  event->set(Event::ConsoleRightDifficultyA, (value & 2) >> 1);
  event->set(Event::ConsoleRightDifficultyB, !((value & 2) >> 1));
}

void ALEState::setActionJoysticks(Event* event, int player_a_action,
                                  int player_b_action) {
  // Reset keys
  resetKeys(event);

  switch (player_a_action) {
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
      Logger::Error << "Invalid Player A Action: " << player_a_action << "\n";
      std::exit(-1);
  }

  switch (player_b_action) {
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
      Logger::Info << "Sending Reset...\n";
      break;
    default:
      Logger::Error << "Invalid Player B Action: " << player_b_action << "\n";
      std::exit(-1);
  }
}

/* ***************************************************************************
    Function resetKeys
    Unpresses all control-relevant keys
 * ***************************************************************************/
void ALEState::resetKeys(Event* event) {
  event->set(Event::ConsoleReset, 0);
  event->set(Event::ConsoleSelect, 0);
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
  event->set(Event::PaddleTwoFire, 0);
  event->set(Event::PaddleThreeFire, 0);

  // Set the difficulty switches accordingly for this time step.
  setDifficultySwitches(event, m_difficulty);
}

bool ALEState::equals(ALEState& rhs) {
  return (rhs.m_serialized_state == this->m_serialized_state &&
          std::equal(rhs.m_paddle,rhs.m_paddle+4,this->m_paddle) &&
          rhs.m_frame_number == this->m_frame_number &&
          rhs.m_episode_frame_number == this->m_episode_frame_number &&
          rhs.m_mode == this->m_mode && rhs.m_difficulty == this->m_difficulty);
}

}  // namespace ale
