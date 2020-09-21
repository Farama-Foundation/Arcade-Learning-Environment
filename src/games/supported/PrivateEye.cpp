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

#include "games/supported/PrivateEye.hpp"

#include "games/RomUtils.hpp"

namespace ale {

PrivateEyeSettings::PrivateEyeSettings() { reset(); }

/* create a new instance of the rom */
RomSettings* PrivateEyeSettings::clone() const {
  return new PrivateEyeSettings(*this);
}

/* process the latest information from ALE */
void PrivateEyeSettings::step(const System& system) {
  // update the reward
  int score = getDecimalScore(0xCA, 0xC9, 0xC8, &system);
  int reward = score - m_score;
  m_reward = reward;
  m_score = score;

  // update terminal status
  int copyright_timer = readRam(&system, 0xC2);
  // 00 when the game is running; 01 at start of game.
  m_terminal = copyright_timer != 0x00 && copyright_timer != 0x01;
}

/* is end of game */
bool PrivateEyeSettings::isTerminal() const { return m_terminal; };

/* get the most recently observed reward */
reward_t PrivateEyeSettings::getReward() const { return m_reward; }

/* is an action part of the minimal set? */
bool PrivateEyeSettings::isMinimal(const Action& a) const {
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
void PrivateEyeSettings::reset() {
  m_reward = 0;
  m_score = 1000;
  m_terminal = false;
}

/* saves the state of the rom settings */
void PrivateEyeSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void PrivateEyeSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

ActionVect PrivateEyeSettings::getStartingActions() {
  return {PLAYER_A_UP};
}

// returns a list of mode that the game can be played in
ModeVect PrivateEyeSettings::getAvailableModes() {
  ModeVect modes(getNumModes());
  for (unsigned int i = 0; i < modes.size(); i++) {
    modes[i] = i;
  }
  return modes;
}

// set the mode of the game
// the given mode must be one returned by the previous function
void PrivateEyeSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {
  if (m < getNumModes()) {
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

DifficultyVect PrivateEyeSettings::getAvailableDifficulties() {
  return {0, 1, 2, 3};
}

}  // namespace ale
