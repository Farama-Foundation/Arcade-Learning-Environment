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

#include "ale/games/supported/Tennis.hpp"

#include "ale/games/RomUtils.hpp"

namespace ale {
using namespace stella;

TennisSettings::TennisSettings() { reset(); }

/* create a new instance of the rom */
RomSettings* TennisSettings::clone() const {
  return new TennisSettings(*this);
}

/* process the latest information from ALE */
void TennisSettings::step(const System& system) {
  // update the reward
  int my_score = readRam(&system, 0xC5);
  int oppt_score = readRam(&system, 0xC6);
  int my_points = readRam(&system, 0xC7);
  int oppt_points = readRam(&system, 0xC8);
  int delta_score = my_score - oppt_score;
  int delta_points = my_points - oppt_points;

  // a reward for the game
  if (m_prev_delta_points != delta_points) {
    m_reward = delta_points - m_prev_delta_points;
    turn_counter += 1;
  }
  // a reward for each point
  else if (m_prev_delta_score != delta_score)
    m_reward = delta_score - m_prev_delta_score;
  else
    m_reward = 0;
  m_reward_p2 = -m_reward;

  m_prev_delta_points = delta_points;
  m_prev_delta_score = delta_score;

  // update terminal status
  m_terminal = (my_points >= 6 && delta_points >= 2) ||
               (oppt_points >= 6 && -delta_points >= 2) ||
               (my_points == 7 || oppt_points == 7);

  if (two_player_mode) {
    // serve stalling is not possible alongside scoring, so this will not
    // overwrite previously calculated scores/terminal above
    int serve_stall_counter = readRam(&system, 0xcc);
    if (serve_stall_counter == 0) {
      no_serve_counter = 0;
    }
    no_serve_counter += 1;
    // times out the serve in two-player mode to disallow stalling
    if (max_turn_time > 0 && no_serve_counter >= max_turn_time) {
      if (turn_counter % 2 == 0) {
        m_reward = -1;
        m_reward_p2 = 0;
      } else {
        m_reward = 0;
        m_reward_p2 = -1;
      }
      no_serve_counter = 0;
    }
  }
}

/* is end of game */
bool TennisSettings::isTerminal() const { return m_terminal; };

/* get the most recently observed reward */
reward_t TennisSettings::getReward() const { return m_reward; }

reward_t TennisSettings::getRewardP2() const { return m_reward_p2; }

/* is an action part of the minimal set? */
bool TennisSettings::isMinimal(const Action& a) const {
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
  m_reward = 0;
  m_reward_p2 = 0;
  m_prev_delta_points = 0;
  m_prev_delta_score = 0;
  m_terminal = false;
  turn_counter = 0;
  no_serve_counter = 0;
}

/* saves the state of the rom settings */
void TennisSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putBool(m_terminal);

  ser.putInt(m_prev_delta_points);
  ser.putInt(m_prev_delta_score);

  ser.putInt(m_reward_p2);
  ser.putInt(turn_counter);
  ser.putInt(no_serve_counter);
  ser.putBool(two_player_mode);
}

// loads the state of the rom settings
void TennisSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_terminal = ser.getBool();

  m_prev_delta_points = ser.getInt();
  m_prev_delta_score = ser.getInt();

  m_reward_p2 = ser.getInt();
  turn_counter = ser.getInt();
  no_serve_counter = ser.getInt();
  two_player_mode = ser.getBool();
}

// returns a list of mode that the game can be played in
ModeVect TennisSettings::getAvailableModes() {
  return {0, 2};
}

// Modes 1 and 3 are the two-player variants of modes 0 and 2 (RAM 0x80
// holds the variant directly).
ModeVect TennisSettings::get2PlayerModes() {
  return {1, 3};
}

// set the mode of the game
// the given mode must be one returned by the previous function
void TennisSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {
  if (m < 4) {
    two_player_mode = (m == 1 || m == 3);
    // read the mode we are currently in
    unsigned char mode = readRam(&system, 0x80);
    // press select until the correct mode is reached
    while (mode != m) {
      environment->pressSelect(2);
      mode = readRam(&system, 0x80);
    }
    //reset the environment to apply changes.
    environment->softReset();
  } else {
    throw std::runtime_error("This mode doesn't currently exist for this game");
  }
}

DifficultyVect TennisSettings::getAvailableDifficulties() {
  return {0, 1, 2, 3};
}

void TennisSettings::modifyEnvironmentSettings(Settings& settings) {
  max_turn_time = settings.getInt("max_turn_time");
  if (max_turn_time == -1) {
    // times out a stalled serve after 3 seconds by default
    const int DEFAULT_STALL_LIMIT = 60 * 3;
    max_turn_time = DEFAULT_STALL_LIMIT;
  }
}

}  // namespace ale
