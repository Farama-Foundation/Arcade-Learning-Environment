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

#include "ale/games/supported/Othello.hpp"

#include "ale/games/RomUtils.hpp"

namespace ale {
using namespace stella;

OthelloSettings::OthelloSettings() { reset(); }

RomSettings* OthelloSettings::clone() const {
  return new OthelloSettings(*this);
}

void OthelloSettings::step(const System& system) {
  int white_score = getDecimalScore(0xce, &system);
  int black_score = getDecimalScore(0xd0, &system);
  int score = white_score - black_score;
  m_reward = score - m_score;
  m_score = score;

  // Player indicator is 0xff if white's turn, 0x01 if black's turn, and 0x00
  // if the game is over. Also it is 0x00 in other situations, but only
  // temporarily.
  if (readRam(&system, 0xc0) == 0) {
    ++m_no_input;
  } else {
    m_no_input = 0;
  }

  // The game is over when there are no more valid moves not necessarily when
  // the board is full of counters. We must also wait for the counters to reach
  // their final colour for scoring. We detect this when the turn indicator
  // is 0, signalling no more player input.
  m_terminal = m_no_input > 50;
}

bool OthelloSettings::isTerminal() const { return m_terminal; }

reward_t OthelloSettings::getReward() const { return m_reward; }

bool OthelloSettings::isMinimal(const Action& a) const {
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
      // The joystick is used to move the active square on the board, then
      // fire selects that square. It does not make sense to press fire and
      // move at the same time.
      return true;
    default:
      return false;
  }
}

void OthelloSettings::reset() {
  m_reward = 0;
  m_score = 0;
  m_terminal = false;
  m_no_input = 0;
}

void OthelloSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putInt(m_no_input);
}

void OthelloSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_no_input = ser.getInt();
}

// According to https://atariage.com/manual_html_page.php?SoftwareLabelID=931
// The left difficulty switch must be kept in position 'b' for normal gameplay,
// whilst the right difficulty switch determines whether white (player) or black
// (computer) makes the first move.
DifficultyVect OthelloSettings::getAvailableDifficulties() {
  return {0, 2};
}

// According to https://atariage.com/manual_html_page.php?SoftwareLabelID=931
// there are three one player game modes which set the skill level as
// beginner, intermediate and expert.
ModeVect OthelloSettings::getAvailableModes() {
  return {0, 1, 2};
}

void OthelloSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {
  if (m < 3) {
    // Read the mode we are currently in.
    unsigned char mode = readRam(&system, 0xde) - 1;

    // Press select until the correct mode is reached.
    while (mode != m) {
      environment->pressSelect(2);
      mode = readRam(&system, 0xde) - 1;
    }

    // Reset the environment to apply changes.
    environment->softReset();
  } else {
    throw std::runtime_error("This game mode is not supported.");
  }
}

}  // namespace ale
