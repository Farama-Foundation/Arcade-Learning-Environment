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

#include "ale/games/supported/Turmoil.hpp"

#include "ale/games/RomUtils.hpp"

namespace ale {
using namespace stella;

TurmoilSettings::TurmoilSettings() { reset(); }

/* create a new instance of the rom */
RomSettings* TurmoilSettings::clone() const {
  return new TurmoilSettings(*this);
}

/* process the latest information from ALE */
void TurmoilSettings::step(const System& system) {
  // update the reward
  int score = getDecimalScore(0x89, 0x8A, &system);
  score += readRam(&system, 0xD3);
  score *= 10;
  int reward = score - m_score;
  m_reward = reward;
  m_score = score;

  // update terminal status
  int lives_byte = readRam(&system, 0xB9);
  m_terminal = (lives_byte == 0) && readRam(&system, 0xC5) == 0x01;

  m_lives = lives_byte;
}

/* is end of game */
bool TurmoilSettings::isTerminal() const { return m_terminal; };

/* get the most recently observed reward */
reward_t TurmoilSettings::getReward() const { return m_reward; }

/* is an action part of the minimal set? */
bool TurmoilSettings::isMinimal(const Action& a) const {
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
    case PLAYER_A_RIGHTFIRE:
    case PLAYER_A_LEFTFIRE:
      return true;
    default:
      return false;
  }
}

/* reset the state of the game */
void TurmoilSettings::reset() {
  m_reward = 0;
  m_score = 0;
  m_terminal = false;
  m_lives = 4;
}

/* saves the state of the rom settings */
void TurmoilSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putInt(m_lives);
}

// loads the state of the rom settings
void TurmoilSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_lives = ser.getInt();
}

ActionVect TurmoilSettings::getStartingActions() {
  return {PLAYER_A_FIRE};
}

// According to https://atariage.com/manual_html_page.php?SoftwareLabelID=571
// there are 9 different game modes, of increasing speed.
ModeVect TurmoilSettings::getAvailableModes() {
  return {0, 1, 2, 3, 4, 5, 6, 7, 8};
}

void TurmoilSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {
  if (m < 9) {
    // Press select until the correct mode is reached.
    while (readRam(&system, 0xea) != m) { environment->pressSelect(2); }

    // Reset the environment to apply changes.
    environment->softReset();
  } else {
    throw std::runtime_error("This game mode is not supported.");
  }
}

}  // namespace ale
