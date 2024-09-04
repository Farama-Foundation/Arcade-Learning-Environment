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

#include "ale/games/supported/TicTacToe3d.hpp"
#include "ale/games/RomUtils.hpp"

namespace ale {
using namespace stella;

TicTacToe3dSettings::TicTacToe3dSettings() { reset(); }

RomSettings* TicTacToe3dSettings::clone() const {
  return new TicTacToe3dSettings(*this);
}

void TicTacToe3dSettings::step(const System& system) {
  m_reward = 0;
  // When the game ends, a return address of 0xFE10 is put on the stack (two
  // last bytes of the ram). The winner is read from 0xE1 address where
  // 0x28 represents 'O's and 0x08 'X's.
  int addressLo = readRam(&system, 0xfe);
  int addressHi = readRam(&system, 0xff);
  int winner = readRam(&system, 0xe1);
  if (addressHi == 0xf3 && addressLo == 0x10) {
    if (winner == 0x08) {
      m_reward = 1;
    } else {
      m_reward = -1;
    }
    m_terminal = true;
  }
  // There is also a possibility of a draw. Here game pauses infinitely, so
  // to find that state we read all the 16 bytes representing all possible
  // grid values. If none of them is zero, we terminate the game with a draw.
  for (int i = 0x9a; i <= 0xd9 ; ++i) {
    auto gridValue = readRam(&system, i);
    if (gridValue == 0) {
      return;
    }
  }
  m_terminal = true;
}

bool TicTacToe3dSettings::isTerminal() const { return m_terminal; }

reward_t TicTacToe3dSettings::getReward() const { return m_reward; }

bool TicTacToe3dSettings::isMinimal(const Action& a) const {
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
      return true;
    default:
      return false;
  }
}

void TicTacToe3dSettings::reset() {
  m_reward = 0;
  m_terminal = false;
}

void TicTacToe3dSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putBool(m_terminal);
}

void TicTacToe3dSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_terminal = ser.getBool();
}

// According to https://atariage.com/manual_html_page.php?SoftwareID=798
// The right difficulty switch determines who will began a game. For one-player
// games, when the switch is in the a position, you start; when the switch is in
// the b position, the computer starts.
// The left difficulty switch may be used to create a "set up" mode on the
// screen. We're not using it.
DifficultyVect TicTacToe3dSettings::getAvailableDifficulties() {
  return {0, 2};
}

// Returns a list of mode that the game can be played in.
ModeVect TicTacToe3dSettings::getAvailableModes() {
  return {0, 1, 2, 3, 4, 5, 6, 7, 8};
}

// Set the mode of the game.
// The given mode must be one returned by the previous function.
void TicTacToe3dSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {
  if (0 <= m && m <= 8) {
    while (true) {
      // read the mode we are currently in
      unsigned char mode = readRam(&system, 0x88);
      // press select until the correct mode is reached in the welcome screen
      if (m == mode) {
        break;
      }
      environment->pressSelect(2);
    }
    // Reset the environment to apply changes.
    environment->softReset();
  } else {
    throw std::runtime_error("This mode doesn't currently exist for this game");
  }
}

}  // namespace ale
