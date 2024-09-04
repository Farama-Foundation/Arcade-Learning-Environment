/* *****************************************************************************
 * The lines 61, 102, 110 and 118 are based on Xitari's code, from Google Inc.
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

#include "ale/games/supported/Backgammon.hpp"

#include "ale/games/RomUtils.hpp"

namespace ale {
using namespace stella;

std::int8_t readPieces(const System* system, int offset) {
  int value = readRam(system, offset);
  if (value >= 128) {
    // Return player pieces as negative counters
    int res = value - 256;
    return res;
  } else {
    return value;
  }
}

BackgammonSettings::BackgammonSettings() { reset(); }

RomSettings* BackgammonSettings::clone() const {
  return new BackgammonSettings(*this);
}

void BackgammonSettings::step(const System& system) {
  // Memory map:
  // 80       = player out
  // 81 to 86 = board bottom left quadrant
  // 87       = left prison
  // 88 to 8d = board top left quadrant
  // 8e       = right player finished
  // 8f to 94 = board bottom right quadrant
  // 95       = right prison
  // 96 to 9b = board top right quadrant
  //
  // Data:
  //
  // Player pieces are stored as negative numbers (0xFF to 0xF1)
  // Computer pieces are stored as positive numbers (0x01 to 0x0F)
  //
  // Strategy:
  //
  // One complication is that during the computers turn it uses this space for
  // scratch. We cannot simply check the pieces that are out for each player as
  // this value is used for other purposes. In order to determine whether the
  // board state is valid we add up all the player and computers pieces and
  // only if both add up to 15 do we process the result.
  //
  // After the computers turn after restoring the board state the pieces are
  // inverted for a frame before being valid again, but since the win state
  // is picked up before the inversion occurs we don't need to worry about this.

  std::int8_t num_player_pieces_out = -readPieces(&system, 0x80);
  std::int8_t num_computer_pieces_out = readPieces(&system, 0x8E);

  std::int8_t num_player_pieces_in = 0;
  std::int8_t num_computer_pieces_in = 0;

  // Count all player/computer pieces in order to verify the board state is
  // valid.
  for (int address = 0x81; address <= 0x8d; ++address) {
    std::int8_t pieces = readPieces(&system, address);
    if (pieces > 0) {
      num_computer_pieces_in += pieces;
    } else if (pieces < 0) {
      num_player_pieces_in += -pieces;
    }
  }
  for (int address = 0x8f; address <= 0x9b; ++address) {
    std::int8_t pieces = readPieces(&system, address);
    if (pieces > 0) {
      num_computer_pieces_in += pieces;
    } else if (pieces < 0) {
      num_player_pieces_in += -pieces;
    }
  }

  // Make sure the board state is valid before checking the win condition
  bool is_valid = (num_computer_pieces_in + num_computer_pieces_out == 15 &&
                   num_player_pieces_in + num_player_pieces_out == 15);
  if (is_valid) {
    if (num_player_pieces_out == 15) {
      m_terminal = true;
      m_reward = 1;
    } else if (num_computer_pieces_out == 15) {
      m_terminal = true;
      m_reward = -1;
    }
  }
}

bool BackgammonSettings::isTerminal() const { return m_terminal; }

reward_t BackgammonSettings::getReward() const { return m_reward; }

bool BackgammonSettings::isMinimal(const Action& a) const {
  switch (a) {
    case PLAYER_A_FIRE:
    case PLAYER_A_RIGHT:
    case PLAYER_A_LEFT:
      return true;
    default:
      return false;
  }
}

void BackgammonSettings::reset() {
  m_reward = 0;
  m_terminal = false;
}

void BackgammonSettings::modifyEnvironmentSettings(Settings& settings) {
  // Note that Backgammon uses the paddle controller, but the range needs to be
  // extended to allow the cursor to move to all areas of the board.
  // Recommend setting paddle_max = 1000000.
  settings.setInt("paddle_max", 1000000);
}

void BackgammonSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void BackgammonSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_terminal = ser.getBool();
}

ModeVect BackgammonSettings::getAvailableModes() {
  // https://atariage.com/manual_html_page.php?SoftwareID=842
  // We only support mode 3 which is single player, no acey deucey and no
  // doubling cube.
  return {0};
}

void BackgammonSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {
  if (m == 0) {
    while (readRam(&system, 0xDC) != 3) { environment->pressSelect(1); }
    // reset the environment to apply changes.
    environment->softReset();
  } else {
    throw std::runtime_error("This game mode is not supported.");
  }
}

DifficultyVect BackgammonSettings::getAvailableDifficulties() {
  // This moves out of "setup mode" and actually plays a game
  return {3};
}

}  // namespace ale
