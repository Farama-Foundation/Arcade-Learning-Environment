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

#include "Catch.hpp"

#include "../RomUtils.hpp"

namespace ale {

CatchSettings::CatchSettings() { reset(); }

/* *****************************************************************************
 * Catch settings.
 *
 * Location ram_86 contains the game state: >0 means the game can terminate,
 * however the game only forces a reset when this value reaches 20.  This is so
 * the score remains visible long enough for the player to observe it.
 * Location ram_8c contains the score in BDC format.
 *
 * *****************************************************************************
 */

RomSettings* CatchSettings::clone() const {
  return new CatchSettings(*this);
}

void CatchSettings::step(const System& system) {
  // update terminal status - waits at least 10 frames before ending so the
  // score is visible.
  m_terminal = readRam(&system, 0x86) > 10;

  // update the reward
  int score = getDecimalScore(0x8c, &system);
  m_reward = m_terminal ? (score > 0 ? 1 : -1) : 0;
}

bool CatchSettings::isTerminal() const { return m_terminal; }

reward_t CatchSettings::getReward() const { return m_reward; }

bool CatchSettings::isMinimal(const Action& a) const {
  switch (a) {
    case PLAYER_A_NOOP:
    case PLAYER_A_RIGHT:
    case PLAYER_A_LEFT:
      return true;
    default:
      return false;
  }
}

void CatchSettings::reset() {
  m_reward = 0;
  m_terminal = false;
}

void CatchSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putBool(m_terminal);
}

void CatchSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_terminal = ser.getBool();
}

DifficultyVect CatchSettings::getAvailableDifficulties() {
  return {0, 1};
}

}  // namespace ale
