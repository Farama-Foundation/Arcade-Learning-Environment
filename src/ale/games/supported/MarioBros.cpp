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

#include "ale/games/supported/MarioBros.hpp"

#include "ale/games/RomUtils.hpp"

namespace ale {
using namespace stella;

MarioBrosSettings::MarioBrosSettings() { reset(); }

RomSettings* MarioBrosSettings::clone() const {
  return new MarioBrosSettings(*this);
}

void MarioBrosSettings::step(const System& system) {
  int score = getDecimalScore(0x8A, 0x89, &system) * 100;
  m_reward = score - m_score;
  m_score = score;
  m_lives = readRam(&system, 0x87);

  int score_p2 = getDecimalScore(0x8C, 0x8B, &system) * 100;
  m_reward_p2 = score_p2 - m_score_p2;
  m_score_p2 = score_p2;
  m_lives_p2 = readRam(&system, 0x88);

  // Game terminates when the player(s) run out of lives.
  if (is_two_player) {
    m_terminal = m_lives == 0 && m_lives_p2 == 0;
  } else {
    m_terminal = m_lives == 0;
  }
}

bool MarioBrosSettings::isTerminal() const { return m_terminal; }

reward_t MarioBrosSettings::getReward() const { return m_reward; }

reward_t MarioBrosSettings::getRewardP2() const { return m_reward_p2; }

bool MarioBrosSettings::isMinimal(const Action& a) const {
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

void MarioBrosSettings::reset() {
  m_reward = 0;
  m_score = 0;
  m_lives = 0;
  m_reward_p2 = 0;
  m_score_p2 = 0;
  m_lives_p2 = 0;
  m_terminal = false;
}

void MarioBrosSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putInt(m_lives);
  ser.putBool(m_terminal);

  ser.putInt(m_reward_p2);
  ser.putInt(m_score_p2);
  ser.putInt(m_lives_p2);
  ser.putBool(is_two_player);
}

void MarioBrosSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_lives = ser.getInt();
  m_terminal = ser.getBool();

  m_reward_p2 = ser.getInt();
  m_score_p2 = ser.getInt();
  m_lives_p2 = ser.getInt();
  is_two_player = ser.getBool();
}

// According to https://atariage.com/manual_html_page.php?SoftwareLabelID=286
// there are eight game modes but only four are for one player. The game mode
// determines whether there are fireballs present and how many lives the player
// gets to start (3 or 5).
ModeVect MarioBrosSettings::getAvailableModes() {
  return {0, 2, 4, 6};
}

// The odd game variants are the two-player versions of the even ones
// (RAM 0x80 holds the variant directly).
ModeVect MarioBrosSettings::get2PlayerModes() {
  return {1, 3, 5, 7};
}

void MarioBrosSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {
  if (m < 8) {
    is_two_player = (m % 2 == 1);
    // Press select until the correct mode is reached.
    while (readRam(&system, 0x80) != static_cast<int>(m)) {
      environment->pressSelect(5);
    }

    // Reset the environment to apply changes.
    environment->softReset();
  } else {
    throw std::runtime_error("This game mode is not supported.");
  }
}

ActionVect MarioBrosSettings::getStartingActions() {
  // Must press fire to start the game but there is quite a strong debounce
  // in effect so need to wait 10 frames then hold fire for at least 7 frames
  // in order to begin.
  ActionVect startingActions;
  for (int i = 0; i < 10; ++i) {
    startingActions.push_back(PLAYER_A_NOOP);
  }
  for (int i = 0; i < 7; ++i) {
    startingActions.push_back(PLAYER_A_FIRE);
  }
  return startingActions;
}

}  // namespace ale
