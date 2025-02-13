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

#include "ale/games/supported/Superman.hpp"

#include "ale/games/RomUtils.hpp"

namespace ale {
using namespace stella;

SupermanSettings::SupermanSettings() { reset(); }

RomSettings* SupermanSettings::clone() const {
  return new SupermanSettings(*this);
}

void SupermanSettings::step(const System& system) {
  int seconds = getDecimalScore(0xe2, &system);
  int minutes = getDecimalScore(0xe3, &system);
  m_time_in_seconds = minutes * 60 + seconds;

  int room_address = readRam(&system, 0x80) + (readRam(&system, 0x81) << 8);
  int is_clark_kent = readRam(&system, 0x9f) & 0x40;
  // Game ends when player enters the Daily Bugle room as Clark Kent.
  m_terminal = is_clark_kent && room_address == 0xf2ac;

  // Note that the in-game time will just wrap after 100 minutes.
  const int max_time_in_seconds = 99 * 60 + 59;
  // Reward is proportional to the speed at which the game is completed.
  m_reward = m_terminal ? max_time_in_seconds - m_time_in_seconds : 0;
}

bool SupermanSettings::isTerminal() const { return m_terminal; }

reward_t SupermanSettings::getReward() const { return m_reward; }

bool SupermanSettings::isMinimal(const Action& a) const {
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

void SupermanSettings::reset() {
  m_reward = 0;
  m_time_in_seconds = 0;
  m_terminal = false;
}

void SupermanSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_time_in_seconds);
  ser.putBool(m_terminal);
}

void SupermanSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_time_in_seconds = ser.getInt();
  m_terminal = ser.getBool();
}

// According to https://atariage.com/manual_html_page.php?SoftwareLabelID=941
// the right difficulty switch increases the speed of certain enemies and the
// left difficulty switch alters how easy it is to recover after being zapped.
DifficultyVect SupermanSettings::getAvailableDifficulties() {
  return {0, 1, 2, 3};
}

}  // namespace ale
