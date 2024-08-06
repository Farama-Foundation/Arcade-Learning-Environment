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

#include "ale/games/supported/Entombed.hpp"

#include "ale/games/RomUtils.hpp"

namespace ale {
using namespace stella;

EntombedSettings::EntombedSettings() { reset(); }

RomSettings* EntombedSettings::clone() const {
  return new EntombedSettings(*this);
}

void EntombedSettings::step(const System& system) {
  // Score is stored as hexadecimal in RAM 0xE3:
  int score = readRam(&system, 0xe3);
  m_reward = score - m_score;
  m_score = score;
  // Lives are stored as the bottom 2 bits of RAM 0xC7:
  int lives = readRam(&system, 0xc7) & 0x03;
  // Game terminates when the player runs out of lives.
  m_terminal = lives == 0;
}

bool EntombedSettings::isTerminal() const { return m_terminal; }

reward_t EntombedSettings::getReward() const { return m_reward; }

bool EntombedSettings::isMinimal(const Action& a) const {
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

void EntombedSettings::reset() {
  m_reward = 0;
  m_score = 0;
  m_terminal = false;
}

void EntombedSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

void EntombedSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

// According to https://atariage.com/manual_html_page.php?SoftwareLabelID=165
// the left difficulty switch sets the number of 'make-breaks' received when
// collecting the blicking blocks.
DifficultyVect EntombedSettings::getAvailableDifficulties() {
  return {0, 2};
}

// Need to press 'fire' to start, not 'reset', then wait a few frames for the
// game state to be set up.
ActionVect EntombedSettings::getStartingActions() {
  return {PLAYER_A_FIRE, PLAYER_A_NOOP, PLAYER_A_NOOP, PLAYER_A_NOOP,
          PLAYER_A_NOOP, PLAYER_A_NOOP};
}

}  // namespace ale
