/* *****************************************************************************
 * The lines 62, 115, 125 and 133 are based on Xitari's code, from Google Inc.
 *
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

#include "ale/games/supported/MontezumaRevenge.hpp"

#include "ale/games/RomUtils.hpp"

namespace ale {
using namespace stella;

MontezumaRevengeSettings::MontezumaRevengeSettings() { reset(); }

/* create a new instance of the rom */
RomSettings* MontezumaRevengeSettings::clone() const {
  return new MontezumaRevengeSettings(*this);
}

/* process the latest information from ALE */
void MontezumaRevengeSettings::step(const System& system) {
  // update the reward
  int score = getDecimalScore(0x95, 0x94, 0x93, &system);
  int reward = score - m_score;
  m_reward = reward;
  m_score = score;

  // update terminal status
  int new_lives = readRam(&system, 0xBA);
  int some_byte = readRam(&system, 0xFE);
  m_terminal = new_lives == 0 && some_byte == 0x60;

  // Actually does not go up to 8, but that's alright
  m_lives = (new_lives & 0x7) + 1;
}

/* is end of game */
bool MontezumaRevengeSettings::isTerminal() const { return m_terminal; };

/* get the most recently observed reward */
reward_t MontezumaRevengeSettings::getReward() const { return m_reward; }

/* is an action part of the minimal set? */
bool MontezumaRevengeSettings::isMinimal(const Action& a) const {
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

/* reset the state of the game */
void MontezumaRevengeSettings::reset() {
  m_reward = 0;
  m_score = 0;
  m_terminal = false;
  m_lives = 6;
}

/* saves the state of the rom settings */
void MontezumaRevengeSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putInt(m_lives);
}

// loads the state of the rom settings
void MontezumaRevengeSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_lives = ser.getInt();
}

}  // namespace ale
