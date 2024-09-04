/* *****************************************************************************
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License version 2
 * as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 * *****************************************************************************
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

#include "ale/games/supported/BasicMath.hpp"

#include "ale/games/RomUtils.hpp"

namespace ale {
using namespace stella;

BasicMathSettings::BasicMathSettings() { reset(); }

RomSettings* BasicMathSettings::clone() const {
  return new BasicMathSettings(*this);
}

void BasicMathSettings::step(const System& system) {
  int score = getDecimalScore(0x84, &system);
  m_reward = score - m_score;
  m_score = score;
  // Game terminates after 10 rounds. The round number is in address 0x85 but
  // if we end when this == 10 we miss the final point of reward. Instead
  // check we are on the final score screen, indicated by RAM 0x86 != 0.
  m_terminal = getDecimalScore(0x86, &system) != 0;
}

bool BasicMathSettings::isTerminal() const { return m_terminal; }

reward_t BasicMathSettings::getReward() const { return m_reward; }

bool BasicMathSettings::isMinimal(const Action& a) const {
  switch (a) {
    case PLAYER_A_NOOP:
    case PLAYER_A_FIRE:
    case PLAYER_A_UP:
    case PLAYER_A_RIGHT:
    case PLAYER_A_LEFT:
    case PLAYER_A_DOWN:
      // Fire button is select in this game so doesn't make sense to move
      // and press fire at the same time.
      return true;
    default:
      return false;
  }
}

void BasicMathSettings::reset() {
  m_reward = 0;
  m_score = 0;
  m_terminal = false;
}

void BasicMathSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

void BasicMathSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

// According to https://atariage.com/manual_html_page.php?SoftwareID=850
// the right difficulty switch determines if there is a time limit on the
// questions and the left difficulty switch determines short single digit
// questions or longer two digit questions.
DifficultyVect BasicMathSettings::getAvailableDifficulties() {
  return {0, 2, 3};
}

// According to https://atariage.com/manual_html_page.php?SoftwareID=850
// there are four valid game modes with random artihmetic problems.
ModeVect BasicMathSettings::getAvailableModes() {
  return {5, 6, 7, 8};
}

void BasicMathSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {
  if (isModeSupported(m)) {
    // Press select until the correct mode is reached.
    while (readRam(&system, 0xc5) != m) {
      environment->pressSelect(2);
    }

    // Reset the environment to apply changes.
    environment->softReset();
  } else {
    throw std::runtime_error("This game mode is not supported.");
  }
}

}  // namespace ale
