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

#include "ale/games/supported/Frogger.hpp"

#include "ale/games/RomUtils.hpp"

namespace ale {
using namespace stella;

FroggerSettings::FroggerSettings() { reset(); }

/* create a new instance of the rom */
RomSettings* FroggerSettings::clone() const {
  return new FroggerSettings(*this);
}

/* process the latest information from ALE */
void FroggerSettings::step(const System& system) {
  // update the reward
  int score = getDecimalScore(0xCE, 0xCC, &system);
  int reward = score - m_score;
  m_reward = reward;
  m_score = score;

  // update terminal status
  m_lives = readRam(&system, 0xD0);
  m_terminal = readRam(&system, 0xD0) == 0xFF;
}

/* is end of game */
bool FroggerSettings::isTerminal() const { return m_terminal; }

/* get the most recently observed reward */
reward_t FroggerSettings::getReward() const { return m_reward; }

/* is an action part of the minimal set? */
bool FroggerSettings::isMinimal(const Action& a) const {
  switch (a) {
    case PLAYER_A_NOOP:
    case PLAYER_A_UP:
    case PLAYER_A_RIGHT:
    case PLAYER_A_LEFT:
    case PLAYER_A_DOWN:
      return true;
    default:
      return false;
  }
}

/* reset the state of the game */
void FroggerSettings::reset() {
  m_reward = 0;
  m_score = 0;
  m_terminal = false;
  m_lives = 4;
}

/* saves the state of the rom settings */
void FroggerSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putInt(m_lives);
}

// loads the state of the rom settings
void FroggerSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_lives = ser.getInt();
}

// According to https://atariage.com/manual_html_page.php?SoftwareLabelID=194
// there are six variations of the game with three being for one player only.
// The game mode are described as easiest, more difficult and "speedy Frogger".
ModeVect FroggerSettings::getAvailableModes() {
  return {0, 1, 2};
}

void FroggerSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {
  if (m < 3) {
    // Read the mode we are currently in.
    int mode = readRam(&system, 0xdd);
    // Skip even numbered modes as these are for two players.
    int desired_mode = 1 + m * 2;

    // Press select until the correct mode is reached for single player only.
    while (mode != desired_mode) {
      environment->pressSelect(2);
      mode = readRam(&system, 0xdd);
    }

    // Reset the environment to apply changes.
    environment->softReset();
  } else {
    throw std::runtime_error("This game mode is not supported.");
  }
}

// According to https://atariage.com/manual_html_page.php?SoftwareLabelID=194
// the left difficulty switch sets whether the player loses a life when carried
// off screen on a floating object, or whether they reappear on the other side.
DifficultyVect FroggerSettings::getAvailableDifficulties() {
  return {0, 1};
}

}  // namespace ale
