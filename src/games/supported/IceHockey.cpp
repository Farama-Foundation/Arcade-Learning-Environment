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

#include "games/supported/IceHockey.hpp"

#include <algorithm>

#include "games/RomUtils.hpp"

namespace ale {

IceHockeySettings::IceHockeySettings() { reset(); }

/* create a new instance of the rom */
RomSettings* IceHockeySettings::clone() const {
  return new IceHockeySettings(*this);
}

/* process the latest information from ALE */
void IceHockeySettings::step(const System& system) {
  // update the reward
  int my_score = std::max(getDecimalScore(0x8A, &system), 0);
  int oppt_score = std::max(getDecimalScore(0x8B, &system), 0);
  int score = my_score - oppt_score;
  int reward = std::min(score - m_score, 1);
  m_reward = reward;
  m_score = score;

  // update terminal status
  int minutes = readRam(&system, 0x87);
  int seconds = readRam(&system, 0x86);
  // end of game when out of time
  m_terminal = minutes == 0 && seconds == 0;
}

/* is end of game */
bool IceHockeySettings::isTerminal() const { return m_terminal; };

/* get the most recently observed reward */
reward_t IceHockeySettings::getReward() const { return m_reward; }

/* is an action part of the minimal set? */
bool IceHockeySettings::isMinimal(const Action& a) const {
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
void IceHockeySettings::reset() {
  m_reward = 0;
  m_score = 0;
  m_terminal = false;
}

/* saves the state of the rom settings */
void IceHockeySettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void IceHockeySettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

// returns a list of mode that the game can be played in
ModeVect IceHockeySettings::getAvailableModes() {
  return {0, 2};
}

// set the mode of the game
// the given mode must be one returned by the previous function
void IceHockeySettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {
  if (m == 0 || m == 2) {
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

DifficultyVect IceHockeySettings::getAvailableDifficulties() {
  return {0, 1, 2, 3};
}

}  // namespace ale
