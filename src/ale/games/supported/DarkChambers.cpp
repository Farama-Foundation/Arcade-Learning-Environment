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

#include "ale/games/supported/DarkChambers.hpp"

#include "ale/games/RomUtils.hpp"

namespace ale {
using namespace stella;

DarkChambersSettings::DarkChambersSettings() { reset(); }

RomSettings* DarkChambersSettings::clone() const {
  return new DarkChambersSettings(*this);
}

void DarkChambersSettings::step(const System& system) {
  m_reward = 0;
  int newLevel = readRam(&system, 0xD5);
  if (newLevel < m_level) {
    // As you progress through the game levels will go up.
    // If the new level is lower than the previous level we terminate the game.
    m_terminal = true;
  } else {
    m_level = newLevel;
    int newScore = getDecimalScore(0xCC, 0xCF, &system) * 10;
    if (newScore < m_score) {
      // We exceeded maximum score.
      m_terminal = true;
    } else {
      m_reward = newScore - m_score;
      m_score = newScore;
    }

    // First three bits of the health are used for items, we need to discard
    // them.
    m_health = readRam(&system, 0xCA) & 0x1F;
    if (m_health == 0) {
      m_terminal = true;
    }
  }
}

bool DarkChambersSettings::isTerminal() const { return m_terminal; }

reward_t DarkChambersSettings::getReward() const { return m_reward; }

bool DarkChambersSettings::isMinimal(const Action& a) const {
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

void DarkChambersSettings::reset() {
  m_reward = 0;
  m_terminal = false;
  m_health = 0;
  m_level = 0;
  m_score = 0;
}

void DarkChambersSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putBool(m_terminal);
  ser.putInt(m_health);
  ser.putInt(m_level);
  ser.putInt(m_score);
}

void DarkChambersSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_terminal = ser.getBool();
  m_health = ser.getInt();
  m_level = ser.getInt();
  m_score = ser.getInt();
}

// According to https://atariage.com/manual_html_page.php?SoftwareLabelID=122
// there are 3 game modes for single player and 3 game modes for 2 players.
// 0, 2, 4 represents easy, medium and hard modes for single player.
// Unfortunatelly our default implementation for setting mode does not work
// in this game: pressSelect() does not change that memory location. For now,
// we default to mode 0.
// TODO(b/147488509): ALE Atari | Unlock Dark Chambers games modes
ModeVect DarkChambersSettings::getAvailableModes() {
  return {0};
}

ActionVect DarkChambersSettings::getStartingActions() {
  // When this ROM is booted there is a short animation sequence before any
  // user input is accepted, even the 'start' button. This lasts for around
  // 8 seconds so we wait for 486 frames.
  return ActionVect(486, PLAYER_A_NOOP);
}

}  // namespace ale
