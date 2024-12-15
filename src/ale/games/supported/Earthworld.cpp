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

#include "ale/games/supported/Earthworld.hpp"

#include "ale/games/RomUtils.hpp"

namespace ale {
using namespace stella;

EarthworldSettings::EarthworldSettings() { reset(); }

RomSettings* EarthworldSettings::clone() const {
  return new EarthworldSettings(*this);
}

void EarthworldSettings::step(const System& system) {
  // Address 0xa7 contains the clue number which we're using a proxy for score.
  int score = getDecimalScore(0xa7, &system);
  m_reward = score - m_score;
  m_score = score;
  // Game terminates when the player finds the 10th clue.
  m_terminal = score == 10;
}

bool EarthworldSettings::isTerminal() const { return m_terminal; }

reward_t EarthworldSettings::getReward() const { return m_reward; }

bool EarthworldSettings::isMinimal(const Action& a) const {
  switch (a) {
    case NOOP:
    case FIRE:
    case UP:
    case RIGHT:
    case LEFT:
    case DOWN:
    case UPRIGHT:
    case UPLEFT:
    case DOWNRIGHT:
    case DOWNLEFT:
    case UPFIRE:
    case RIGHTFIRE:
    case LEFTFIRE:
    case DOWNFIRE:
    case UPRIGHTFIRE:
    case UPLEFTFIRE:
    case DOWNRIGHTFIRE:
    case DOWNLEFTFIRE:
      return true;
    default:
      return false;
  }
}

void EarthworldSettings::reset() {
  m_reward = 0;
  m_score = 0;
  m_terminal = false;
}

void EarthworldSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

void EarthworldSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

// According to https://atariage.com/manual_html_page.php?SoftwareLabelID=541
// the difficulty switches are not used and there is only a single game mode.

}  // namespace ale
