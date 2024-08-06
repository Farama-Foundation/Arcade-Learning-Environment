/* *****************************************************************************
 * The lines 62, 116, 124 and 132 are based on Xitari's code, from Google Inc.
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

#include "ale/games/supported/VideoCheckers.hpp"

#include "ale/games/RomUtils.hpp"

namespace {
void process_board_state(unsigned char state, unsigned char& num_black_pieces,
                         unsigned char& num_white_pieces) {
  if (state == 0x10 || state == 0x20) {
    ++num_black_pieces;
  } else if (state == 0x90 || state == 0xa0) {
    ++num_white_pieces;
  }
}
}  // namespace

namespace ale {
using namespace stella;

VideoCheckersSettings::VideoCheckersSettings() { reset(); }

RomSettings* VideoCheckersSettings::clone() const {
  return new VideoCheckersSettings(*this);
}

void VideoCheckersSettings::step(const System& system) {
  unsigned char num_black_pieces = 0;
  unsigned char num_white_pieces = 0;

  for (int address = 0x80; address <= 0x87; ++address) {
    unsigned char state = readRam(&system, address);
    process_board_state(state, num_black_pieces, num_white_pieces);
  }
  for (int address = 0x89; address <= 0x90; ++address) {
    unsigned char state = readRam(&system, address);
    process_board_state(state, num_black_pieces, num_white_pieces);
  }
  for (int address = 0x92; address <= 0x99; ++address) {
    unsigned char state = readRam(&system, address);
    process_board_state(state, num_black_pieces, num_white_pieces);
  }
  for (int address = 0x9b; address <= 0xa2; ++address) {
    unsigned char state = readRam(&system, address);
    process_board_state(state, num_black_pieces, num_white_pieces);
  }

  if (num_black_pieces == 0) {
    m_reward = m_reverse_checkers ? +1 : -1;
    m_terminal = true;
  } else if (num_white_pieces == 0) {
    m_reward = m_reverse_checkers ? -1 : +1;
    m_terminal = true;
  }
}

bool VideoCheckersSettings::isTerminal() const { return m_terminal; }

reward_t VideoCheckersSettings::getReward() const { return m_reward; }

bool VideoCheckersSettings::isMinimal(const Action& a) const {
  switch (a) {
    case PLAYER_A_FIRE:
    case PLAYER_A_UPRIGHT:
    case PLAYER_A_UPLEFT:
    case PLAYER_A_DOWNRIGHT:
    case PLAYER_A_DOWNLEFT:
      return true;
    default:
      return false;
  }
}

void VideoCheckersSettings::reset() {
  m_reward = 0;
  m_terminal = false;
}

void VideoCheckersSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putBool(m_terminal);
}

void VideoCheckersSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_terminal = ser.getBool();
}

ModeVect VideoCheckersSettings::getAvailableModes() {
  // https://atariage.com/manual_html_page.php?SoftwareID=579
  // Games 1 to 9 are regular checkers in increasing difficulty,
  // displayed onscreen and stored in memory as 1 to 9.
  // Games 11 to 19 are reverse checkers in increasing difficulty,
  // displayed onscreen as 11 to 19 but stored in memory as 17 to 25.
  return {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19};
}

// Set the game mode.
// The given mode must be one returned by the previous function.
void VideoCheckersSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {
  auto availableModes = getAvailableModes();
  if (isModeSupported(m)) {
    m_reverse_checkers = m >= 11;
    // apply offset to match corresponding value in memory
    if (m_reverse_checkers) {
      m += 6;
    }

    while (readRam(&system, 0xF6) != m) { environment->pressSelect(1); }

    // reset the environment to apply changes.
    environment->softReset();
  } else {
    throw std::runtime_error("This game mode is not supported.");
  }
}

}  // namespace ale
