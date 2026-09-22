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
    : m_paddle_min(PADDLE_MIN),
      m_paddle_max(PADDLE_MAX),
      m_frame_number(0),
      m_episode_frame_number(0),
      m_mode(0),
      m_difficulty(0),
      m_num_players(1) {
        for (int i = 0; i < 4; i++) {
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
      m_difficulty(rhs.m_difficulty),
      m_num_players(rhs.m_num_players) {
        for (int i = 0; i < 4; i++) {
          m_paddle[i] = rhs.m_paddle[i];
        }
      }

// Sentinel marking the multiplayer (v2) serialization format. Negative so it
// can never collide with the first field of the legacy format, which was a
// (positive) paddle position.
static constexpr int kStateFormatV2 = -0x0A1E0002;

ALEState::ALEState(const std::string& serialized) {
  Deserializer des(serialized);
  int first = des.getInt();
  if (first == kStateFormatV2) {
    for (int i = 0; i < 4; i++) {
      this->m_paddle[i] = des.getInt();
    }
    this->m_frame_number = des.getInt();
    this->m_episode_frame_number = des.getInt();
    this->m_mode = des.getInt();
    this->m_difficulty = des.getInt();
    this->m_num_players = des.getInt();
  } else {
    // Legacy (pre-multiplayer) format: two paddle values, no player count.
    this->m_paddle[0] = first;
    this->m_paddle[1] = des.getInt();
    this->m_paddle[2] = PADDLE_DEFAULT_VALUE;
    this->m_paddle[3] = PADDLE_DEFAULT_VALUE;
    this->m_frame_number = des.getInt();
    this->m_episode_frame_number = des.getInt();
    this->m_mode = des.getInt();
    this->m_difficulty = des.getInt();
    this->m_num_players = 1;
  }
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

  try {
    osystem->console().system().loadState(md5, deser);
    settings->loadState(deser);
    bool rng_included = deser.getBool();
    if (rng_included) {
      rng->loadState(deser);
    }
  } catch (const char* msg) {
    // The emucore Serializer/Deserializer throw raw C strings; convert them
    // to a translatable exception. This is typically hit when restoring a
    // state that was serialized by a different ALE version (games that gained
    // multiplayer support serialize additional fields).
    throw std::runtime_error(
        std::string("Failed to restore ALEState (") + msg +
        "). The state is likely from an incompatible ALE version: games that "
        "gained multiplayer support serialize additional fields, so their "
        "states cannot be exchanged with older releases.");
  }

  // Copy over other member variables
  for (int i = 0; i < 4; i++) {
    m_paddle[i] = rhs.m_paddle[i];
  }
  m_paddle_min = rhs.m_paddle_min;
  m_paddle_max = rhs.m_paddle_max;
  m_frame_number = rhs.m_frame_number;
  m_episode_frame_number = rhs.m_episode_frame_number;
  m_mode = rhs.m_mode;
  m_difficulty = rhs.m_difficulty;
  m_num_players = rhs.m_num_players;
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

  ser.putInt(kStateFormatV2);
  for (int i = 0; i < 4; i++) {
    ser.putInt(this->m_paddle[i]);
  }
  ser.putInt(this->m_frame_number);
  ser.putInt(this->m_episode_frame_number);
  ser.putInt(this->m_mode);
  ser.putInt(this->m_difficulty);
  ser.putInt(this->m_num_players);
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
  // Only initialize the paddles that are in use: setting the resistance of
  // paddles 2/3 (the right controller port) changes what paddle games read
  // from INPT2/INPT3, altering emulation for existing 1/2-player modes.
  int num_paddles = m_num_players <= 2 ? 2 : 4;
  for (int i = 0; i < num_paddles; i++) {
    setPaddle(event, paddle_default, i);
  }
}

void ALEState::setPaddle(Event* event, int paddle_val, int paddle_num) {
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
 *  Updates the positions of the paddle indicated by paddle_num,
 *  and sets an event for updating the corresponding paddle's resistance
 * ********************************************************************/
void ALEState::updatePaddlePosition(Event* event, int delta,
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
  setPaddle(event, m_paddle[paddle_num], paddle_num);
}

/* *********************************************************************
 *  Updates positions of paddles 0 and 1 (for two-player paddle strength mode)
 * ********************************************************************/
void ALEState::updatePaddlePositions(Event* event, int delta_left, int delta_right) {
  updatePaddlePosition(event, delta_left, 0);
  updatePaddlePosition(event, delta_right, 1);
}

/* *********************************************************************
 *  Sets both paddles to given positions (for two-player mode)
 * ********************************************************************/
void ALEState::setPaddles(Event* event, int left, int right) {
  setPaddle(event, left, 0);
  setPaddle(event, right, 1);
}

// Apply the action for the paddle given by pnum (for multiplayer support)
void ALEState::applyActionPaddle(Event* event, int action, int pnum,
                                 float paddle_strength) {
  // First compute whether we should increase or decrease the paddle position
  int delta = 0;

  switch (action) {
    case PLAYER_A_RIGHT:
    case PLAYER_A_RIGHTFIRE:
    case PLAYER_A_UPRIGHT:
    case PLAYER_A_DOWNRIGHT:
    case PLAYER_A_UPRIGHTFIRE:
    case PLAYER_A_DOWNRIGHTFIRE:
      delta = static_cast<int>(-PADDLE_DELTA * fabs(paddle_strength));
      break;

    case PLAYER_A_LEFT:
    case PLAYER_A_LEFTFIRE:
    case PLAYER_A_UPLEFT:
    case PLAYER_A_DOWNLEFT:
    case PLAYER_A_UPLEFTFIRE:
    case PLAYER_A_DOWNLEFTFIRE:
      delta = static_cast<int>(PADDLE_DELTA * fabs(paddle_strength));
      break;

    default:
      delta = 0;
      break;
  }

  // Now update the paddle position
  updatePaddlePosition(event, delta, pnum);

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

// Apply actions for both paddles with continuous paddle strength (for single/two player)
void ALEState::applyActionPaddles(Event* event,
                                  int player_a_action, float paddle_a_strength,
                                  int player_b_action, float paddle_b_strength) {
  // Reset keys
  resetKeys(event);

  int delta_a = 0;
  int delta_b = 0;
  switch (player_a_action) {
    case PLAYER_A_RIGHT:
    case PLAYER_A_RIGHTFIRE:
    case PLAYER_A_UPRIGHT:
    case PLAYER_A_DOWNRIGHT:
    case PLAYER_A_UPRIGHTFIRE:
    case PLAYER_A_DOWNRIGHTFIRE:
      delta_a = static_cast<int>(-PADDLE_DELTA * fabs(paddle_a_strength));
      break;

    case PLAYER_A_LEFT:
    case PLAYER_A_LEFTFIRE:
    case PLAYER_A_UPLEFT:
    case PLAYER_A_DOWNLEFT:
    case PLAYER_A_UPLEFTFIRE:
    case PLAYER_A_DOWNLEFTFIRE:
      delta_a = static_cast<int>(PADDLE_DELTA * fabs(paddle_a_strength));
      break;

    default:
      break;
  }

  switch (player_b_action) {
    case PLAYER_B_RIGHT:
    case PLAYER_B_RIGHTFIRE:
    case PLAYER_B_UPRIGHT:
    case PLAYER_B_DOWNRIGHT:
    case PLAYER_B_UPRIGHTFIRE:
    case PLAYER_B_DOWNRIGHTFIRE:
      delta_b = static_cast<int>(-PADDLE_DELTA * fabs(paddle_b_strength));
      break;

    case PLAYER_B_LEFT:
    case PLAYER_B_LEFTFIRE:
    case PLAYER_B_UPLEFT:
    case PLAYER_B_DOWNLEFT:
    case PLAYER_B_UPLEFTFIRE:
    case PLAYER_B_DOWNLEFTFIRE:
      delta_b = static_cast<int>(PADDLE_DELTA * fabs(paddle_b_strength));
      break;

    default:
      break;
  }

  // Now update the paddle positions
  updatePaddlePositions(event, delta_a, delta_b);

  // Handle reset for player A
  if (player_a_action == RESET)
    event->set(Event::ConsoleReset, 1);

  // Handle fire events for player A
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
      break;
  }

  // Handle fire events for player B
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
      Logger::Info << "Sending Reset...\n";
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
          rhs.m_mode == this->m_mode &&
          rhs.m_difficulty == this->m_difficulty &&
          rhs.m_num_players == this->m_num_players);
}

}  // namespace ale
