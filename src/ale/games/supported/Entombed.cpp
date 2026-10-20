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
  // Lives are stored as the bottom 2 bits of RAM 0xC7:
  lives_p1 = readRam(&system, 0xc7) & 0x03;
  // Player 2 lives are stored as bits 4-5 of RAM 0xC7:
  lives_p2 = (readRam(&system, 0xc7) & 0x30) >> 4;

  if (is_two_player) {
    if (is_cooperative) {
      // Both players are rewarded for every new maze section they reach.
      int cur_substage = readRam(&system, 0xef);
      if (cur_substage > cur_depth) {
        m_reward = 1;
      } else {
        m_reward = 0;
      }
      cur_depth = cur_substage;
    } else {
      // Competitive: reward the player that outlives the other.
      int score = lives_p1 - lives_p2;
      m_reward = score - m_score;
      m_score = score;
    }
    m_terminal = lives_p1 == 0 || lives_p2 == 0;
  } else {
    // Score is stored as hexadecimal in RAM 0xE3:
    int score = readRam(&system, 0xe3);
    m_reward = score - m_score;
    m_score = score;
    // Game terminates when the player runs out of lives.
    m_terminal = lives_p1 == 0;
  }
}

bool EntombedSettings::isTerminal() const { return m_terminal; }

reward_t EntombedSettings::getReward() const { return m_reward; }

// In the cooperative mode both players share the reward; in the
// competitive mode the game is zero-sum.
reward_t EntombedSettings::getRewardP2() const {
  return is_cooperative ? m_reward : -m_reward;
}

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
  lives_p1 = 0;
  lives_p2 = 0;
  cur_depth = 0;
}

void EntombedSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);

  ser.putInt(lives_p1);
  ser.putInt(lives_p2);
  ser.putInt(cur_depth);
  ser.putBool(is_two_player);
  ser.putBool(is_cooperative);
}

void EntombedSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();

  lives_p1 = ser.getInt();
  lives_p2 = ser.getInt();
  cur_depth = ser.getInt();
  is_two_player = ser.getBool();
  is_cooperative = ser.getBool();
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

// Mode 0 is the historic ALE default (the boot variant, no manipulation).
// Mode 2 is the competitive two-player game: each player is trying to
// outlive the other. Mode 3 is the cooperative two-player game: both
// players are rewarded for descending the maze. Both select the ROM's
// two-player variant (RAM 0xF4 == 0) and differ only in reward scheme.
ModeVect EntombedSettings::getAvailableModes() {
  return {0};
}

ModeVect EntombedSettings::get2PlayerModes() {
  return {2, 3};
}

void EntombedSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {
  // Mode 0 keeps the legacy behaviour: play the boot variant untouched.
  if (m == 0) {
    is_two_player = false;
    is_cooperative = false;
    return;
  }
  is_two_player = true;
  is_cooperative = (m == 3);
  while (readRam(&system, 0xf4) != 0) { environment->pressSelect(1); }
  // reset the environment to apply changes.
  environment->softReset();
}

}  // namespace ale
