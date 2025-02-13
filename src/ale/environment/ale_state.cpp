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

#include "ale/environment/ale_state.hpp"

#include <cassert>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

#include "ale/emucore/System.hxx"
#include "ale/emucore/Event.hxx"
#include "ale/emucore/Deserializer.hxx"
#include "ale/emucore/Serializer.hxx"
#include "ale/emucore/Random.hxx"
#include "ale/common/Constants.h"
#include "ale/games/RomSettings.hpp"

namespace ale {
using namespace stella;   // System, Event, Deserializer, Serializer, Random

/** Default constructor - loads settings from system */
ALEState::ALEState()
    : m_left_paddle(PADDLE_DEFAULT_VALUE),
      m_right_paddle(PADDLE_DEFAULT_VALUE),
      m_paddle_min(PADDLE_MIN),
      m_paddle_max(PADDLE_MAX),
      m_frame_number(0),
      m_episode_frame_number(0),
      m_mode(0),
      m_difficulty(0) {}

ALEState::ALEState(const ALEState& rhs, const std::string& serialized)
    : m_left_paddle(rhs.m_left_paddle),
      m_right_paddle(rhs.m_right_paddle),
      m_paddle_min(rhs.m_paddle_min),
      m_paddle_max(rhs.m_paddle_max),
      m_frame_number(rhs.m_frame_number),
      m_episode_frame_number(rhs.m_episode_frame_number),
      m_serialized_state(serialized),
      m_mode(rhs.m_mode),
      m_difficulty(rhs.m_difficulty) {}

ALEState::ALEState(const std::string& serialized) {
  Deserializer des(serialized);
  this->m_left_paddle = des.getInt();
  this->m_right_paddle = des.getInt();
  this->m_frame_number = des.getInt();
  this->m_episode_frame_number = des.getInt();
  this->m_mode = des.getInt();
  this->m_difficulty = des.getInt();
  this->m_serialized_state = des.getString();
  this->m_paddle_min = des.getInt();
  this->m_paddle_max = des.getInt();
}

/** Restores ALE to the given previously saved state. */
void ALEState::load(OSystem* osystem, RomSettings* settings, Random* rng, std::string md5,
                    const ALEState& rhs) {
  assert(rhs.m_serialized_state.length() > 0);

  // Deserialize the stored string into the emulator state
  Deserializer deser(rhs.m_serialized_state);

  osystem->console().system().loadState(md5, deser);
  settings->loadState(deser);
  bool rng_included = deser.getBool();
  if (rng_included) {
    rng->loadState(deser);
  }

  // Copy over other member variables
  m_left_paddle = rhs.m_left_paddle;
  m_right_paddle = rhs.m_right_paddle;
  m_paddle_min = rhs.m_paddle_min;
  m_paddle_max = rhs.m_paddle_max;
  m_frame_number = rhs.m_frame_number;
  m_episode_frame_number = rhs.m_episode_frame_number;
  m_mode = rhs.m_mode;
  m_difficulty = rhs.m_difficulty;
}

ALEState ALEState::save(OSystem* osystem, RomSettings* settings, std::optional<Random*> rng,
                        std::string md5) {
  // Use the emulator's built-in serialization to save the state
  Serializer ser;

  osystem->console().system().saveState(md5, ser);
  settings->saveState(ser);
  ser.putBool(rng.has_value());
  if (rng.has_value()) {
    rng.value()->saveState(ser);
  }

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

  ser.putInt(this->m_left_paddle);
  ser.putInt(this->m_right_paddle);
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
  setPaddles(event, paddle_default, paddle_default);
}

void ALEState::setPaddles(Event* event, int left, int right) {
  m_left_paddle = left;
  m_right_paddle = right;

  // Compute the "resistance" (this is for vestigal clarity)
  int left_resistance = calcPaddleResistance(m_left_paddle);
  int right_resistance = calcPaddleResistance(m_right_paddle);

  // Update the events with the new resistances
  event->set(Event::PaddleZeroResistance, left_resistance);
  event->set(Event::PaddleOneResistance, right_resistance);
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
void ALEState::updatePaddlePositions(Event* event, int delta_left,
                                     int delta_right) {
  // Cap paddle outputs

  m_left_paddle += delta_left;
  if (m_left_paddle < m_paddle_min) {
    m_left_paddle = m_paddle_min;
  }
  if (m_left_paddle > m_paddle_max) {
    m_left_paddle = m_paddle_max;
  }

  m_right_paddle += delta_right;
  if (m_right_paddle < m_paddle_min) {
    m_right_paddle = m_paddle_min;
  }
  if (m_right_paddle > m_paddle_max) {
    m_right_paddle = m_paddle_max;
  }

  // Now set the paddle to their new value
  setPaddles(event, m_left_paddle, m_right_paddle);
}

void ALEState::applyActionPaddles(Event* event,
                                  int player_a_action, float paddle_a_strength,
                                  int player_b_action, float paddle_b_strength) {
  // Reset keys
  resetKeys(event);

  int delta_a = 0;
  int delta_b = 0;
  switch (player_a_action) {
    case RIGHT:
    case RIGHTFIRE:
    case UPRIGHT:
    case DOWNRIGHT:
    case UPRIGHTFIRE:
    case DOWNRIGHTFIRE:
      delta_a = static_cast<int>(-PADDLE_DELTA * fabs(paddle_a_strength));
      break;

    case LEFT:
    case LEFTFIRE:
    case UPLEFT:
    case DOWNLEFT:
    case UPLEFTFIRE:
    case DOWNLEFTFIRE:
      delta_a = static_cast<int>(PADDLE_DELTA * fabs(paddle_a_strength));
      break;

    default:
      break;
  }

  switch (player_b_action) {
    case RIGHT:
    case RIGHTFIRE:
    case UPRIGHT:
    case DOWNRIGHT:
    case UPRIGHTFIRE:
    case DOWNRIGHTFIRE:
      delta_b = static_cast<int>(-PADDLE_DELTA * fabs(paddle_b_strength));
      break;

    case LEFT:
    case LEFTFIRE:
    case UPLEFT:
    case DOWNLEFT:
    case UPLEFTFIRE:
    case DOWNLEFTFIRE:
      delta_b = static_cast<int>(PADDLE_DELTA * fabs(paddle_b_strength));
      break;

    default:
      break;
  }

  // Now update the paddle positions
  updatePaddlePositions(event, delta_a, delta_b);

  // Handle reset
  if (player_a_action == RESET || player_b_action == RESET)
    event->set(Event::ConsoleReset, 1);

  // Now add the fire event
  switch (player_a_action) {
    case FIRE:
    case UPFIRE:
    case RIGHTFIRE:
    case LEFTFIRE:
    case DOWNFIRE:
    case UPRIGHTFIRE:
    case UPLEFTFIRE:
    case DOWNRIGHTFIRE:
    case DOWNLEFTFIRE:
      event->set(Event::PaddleZeroFire, 1);
      break;
    default:
      // Nothing
      break;
  }

  switch (player_b_action) {
    case FIRE:
    case UPFIRE:
    case RIGHTFIRE:
    case LEFTFIRE:
    case DOWNFIRE:
    case UPRIGHTFIRE:
    case UPLEFTFIRE:
    case DOWNRIGHTFIRE:
    case DOWNLEFTFIRE:
      event->set(Event::PaddleOneFire, 1);
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

void ALEState::applyActionJoysticks(Event* event,
                                    int player_a_action, int player_b_action) {
  // Reset keys
  resetKeys(event);
  switch (player_a_action) {
    case NOOP:
      break;
    case FIRE:
      event->set(Event::JoystickZeroFire, 1);
      break;
    case UP:
      event->set(Event::JoystickZeroUp, 1);
      break;
    case RIGHT:
      event->set(Event::JoystickZeroRight, 1);
      break;
    case LEFT:
      event->set(Event::JoystickZeroLeft, 1);
      break;
    case DOWN:
      event->set(Event::JoystickZeroDown, 1);
      break;
    case UPRIGHT:
      event->set(Event::JoystickZeroUp, 1);
      event->set(Event::JoystickZeroRight, 1);
      break;
    case UPLEFT:
      event->set(Event::JoystickZeroUp, 1);
      event->set(Event::JoystickZeroLeft, 1);
      break;
    case DOWNRIGHT:
      event->set(Event::JoystickZeroDown, 1);
      event->set(Event::JoystickZeroRight, 1);
      break;
    case DOWNLEFT:
      event->set(Event::JoystickZeroDown, 1);
      event->set(Event::JoystickZeroLeft, 1);
      break;
    case UPFIRE:
      event->set(Event::JoystickZeroUp, 1);
      event->set(Event::JoystickZeroFire, 1);
      break;
    case RIGHTFIRE:
      event->set(Event::JoystickZeroRight, 1);
      event->set(Event::JoystickZeroFire, 1);
      break;
    case LEFTFIRE:
      event->set(Event::JoystickZeroLeft, 1);
      event->set(Event::JoystickZeroFire, 1);
      break;
    case DOWNFIRE:
      event->set(Event::JoystickZeroDown, 1);
      event->set(Event::JoystickZeroFire, 1);
      break;
    case UPRIGHTFIRE:
      event->set(Event::JoystickZeroUp, 1);
      event->set(Event::JoystickZeroRight, 1);
      event->set(Event::JoystickZeroFire, 1);
      break;
    case UPLEFTFIRE:
      event->set(Event::JoystickZeroUp, 1);
      event->set(Event::JoystickZeroLeft, 1);
      event->set(Event::JoystickZeroFire, 1);
      break;
    case DOWNRIGHTFIRE:
      event->set(Event::JoystickZeroDown, 1);
      event->set(Event::JoystickZeroRight, 1);
      event->set(Event::JoystickZeroFire, 1);
      break;
    case DOWNLEFTFIRE:
      event->set(Event::JoystickZeroDown, 1);
      event->set(Event::JoystickZeroLeft, 1);
      event->set(Event::JoystickZeroFire, 1);
      break;
    case RESET:
      event->set(Event::ConsoleReset, 1);
      Logger::Info << "Sending Reset...\n";
      break;
    default:
      Logger::Error << "Invalid Player A Action: " << player_a_action << "\n";
      std::exit(-1);
  }
  switch (player_b_action) {
    case NOOP:
      break;
    case FIRE:
      event->set(Event::JoystickOneFire, 1);
      break;
    case UP:
      event->set(Event::JoystickOneUp, 1);
      break;
    case RIGHT:
      event->set(Event::JoystickOneRight, 1);
      break;
    case LEFT:
      event->set(Event::JoystickOneLeft, 1);
      break;
    case DOWN:
      event->set(Event::JoystickOneDown, 1);
      break;
    case UPRIGHT:
      event->set(Event::JoystickOneUp, 1);
      event->set(Event::JoystickOneRight, 1);
      break;
    case UPLEFT:
      event->set(Event::JoystickOneUp, 1);
      event->set(Event::JoystickOneLeft, 1);
      break;
    case DOWNRIGHT:
      event->set(Event::JoystickOneDown, 1);
      event->set(Event::JoystickOneRight, 1);
      break;
    case DOWNLEFT:
      event->set(Event::JoystickOneDown, 1);
      event->set(Event::JoystickOneLeft, 1);
      break;
    case UPFIRE:
      event->set(Event::JoystickOneUp, 1);
      event->set(Event::JoystickOneFire, 1);
      break;
    case RIGHTFIRE:
      event->set(Event::JoystickOneRight, 1);
      event->set(Event::JoystickOneFire, 1);
      break;
    case LEFTFIRE:
      event->set(Event::JoystickOneLeft, 1);
      event->set(Event::JoystickOneFire, 1);
      break;
    case DOWNFIRE:
      event->set(Event::JoystickOneDown, 1);
      event->set(Event::JoystickOneFire, 1);
      break;
    case UPRIGHTFIRE:
      event->set(Event::JoystickOneUp, 1);
      event->set(Event::JoystickOneRight, 1);
      event->set(Event::JoystickOneFire, 1);
      break;
    case UPLEFTFIRE:
      event->set(Event::JoystickOneUp, 1);
      event->set(Event::JoystickOneLeft, 1);
      event->set(Event::JoystickOneFire, 1);
      break;
    case DOWNRIGHTFIRE:
      event->set(Event::JoystickOneDown, 1);
      event->set(Event::JoystickOneRight, 1);
      event->set(Event::JoystickOneFire, 1);
      break;
    case DOWNLEFTFIRE:
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

  // Set the difficulty switches accordingly for this time step.
  setDifficultySwitches(event, m_difficulty);
}

bool ALEState::equals(ALEState& rhs) {
  return (rhs.m_serialized_state == this->m_serialized_state &&
          rhs.m_left_paddle == this->m_left_paddle &&
          rhs.m_right_paddle == this->m_right_paddle &&
          rhs.m_frame_number == this->m_frame_number &&
          rhs.m_episode_frame_number == this->m_episode_frame_number &&
          rhs.m_mode == this->m_mode && rhs.m_difficulty == this->m_difficulty);
}

}  // namespace ale
