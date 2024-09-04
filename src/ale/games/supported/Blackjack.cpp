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

#include "ale/games/supported/Blackjack.hpp"

#include "ale/games/RomUtils.hpp"

namespace ale {
using namespace stella;

BlackjackSettings::BlackjackSettings() { reset(); }

RomSettings* BlackjackSettings::clone() const {
  return new BlackjackSettings(*this);
}

void BlackjackSettings::step(const System& system) {
  // When the player runs out of chips the score gets reset to the value '0bbb'.
  bool bust = readRam(&system, 0x86) == 0x0b && readRam(&system, 0x89) == 0xbb;
  // Player chip value stored as decimal value.
  int score = bust ? 0 : getDecimalScore(0x89, 0x86, &system);
  m_reward = score - m_score;
  m_score = score;
  // Game terminates either when the player runs out of chips or 'breaks the
  // bank' with 1000 chips or more.
  m_terminal = bust || score >= 1000;
}

bool BlackjackSettings::isTerminal() const { return m_terminal; }

reward_t BlackjackSettings::getReward() const { return m_reward; }

bool BlackjackSettings::isMinimal(const Action& a) const {
  switch (a) {
    case PLAYER_A_NOOP:
    case PLAYER_A_FIRE:
    case PLAYER_A_UP:
    case PLAYER_A_DOWN:
      return true;
    default:
      return false;
  }
}

void BlackjackSettings::reset() {
  m_reward = 0;
  m_score = 0;
  m_terminal = false;
}

void BlackjackSettings::modifyEnvironmentSettings(Settings& settings) {
  // Note that Blackjack uses the paddle controller, but a slightly different
  // mapping is required to reach the full range of betting values and to select
  // all possible betting actions. Recommend setting paddle_max = 795000.
  settings.setInt("paddle_max", 795000);
}

void BlackjackSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

void BlackjackSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

// According to https://atariage.com/manual_html_page.php?SoftwareID=869
// Left difficulty switch sets how frequently the computer shuffles the cards.
// Right difficulty switch sets the game rules to either 'casino' or 'private'
// Blackjack.
DifficultyVect BlackjackSettings::getAvailableDifficulties() {
  return {0, 1, 2, 3};
}

}  // namespace ale
