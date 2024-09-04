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

#include "ale/games/supported/Klax.hpp"

#include "ale/games/RomUtils.hpp"
#include "ale/common/Constants.h"

namespace ale {
using namespace stella;

KlaxSettings::KlaxSettings() { reset(); }

RomSettings* KlaxSettings::clone() const {
  return new KlaxSettings(*this);
}

/* extracts a decimal value from 3 bytes of memory mapped RAM */
int KlaxSettings::getKlaxScore(int lower_index, int middle_index,
                               int higher_index, const System* system) {
  int score = 0;
  int lower_digits_val = readMappedRam(system, lower_index);
  int lower_right_digit = lower_digits_val & 0xf;
  int lower_left_digit = (lower_digits_val - lower_right_digit) >> 4;
  score += ((10 * lower_left_digit) + lower_right_digit);

  int middle_digits_val = readMappedRam(system, middle_index);
  int middle_right_digit = middle_digits_val & 0xf;
  int middle_left_digit = (middle_digits_val - middle_right_digit) >> 4;
  score += ((1000 * middle_left_digit) + 100 * middle_right_digit);

  int higher_digits_val = readMappedRam(system, higher_index);
  int higher_right_digit = higher_digits_val & 0xf;
  int higher_left_digit = (higher_digits_val - higher_right_digit) >> 4;
  score += ((100000 * higher_left_digit) + 10000 * higher_right_digit);
  return score;
}

void KlaxSettings::step(const System& system) {
  // Score stored as 3 decimal bytes in extended cartridge RAM.
  int score = getKlaxScore(0xf0b4, 0xf0b5, 0xf0b6, &system);
  m_reward = score - m_score;
  m_score = score;

  // Number of missed tiles and maximum misses stored in extended cartridge RAM.
  int misses = readMappedRam(&system, 0xf0ee);
  int max_misses = readMappedRam(&system, 0xf0e9);
  // When the game is active RAM address 0xa8 has the value 4.
  bool game_active = readRam(&system, 0xa8) == 4;

  // The 25 blocks at the bottom of the screen are stored as 25 consecutive
  // bytes from RAM address 0xb3 onwards.
  int num_blocks = 0;
  for (int i = 0; i < 25; ++i) {
    int block_type = readRam(&system, 0xb3 + i);
    // Type 0 is an empty block. When a level is completed a bonus score is
    // calculated by filling the remaining empty blocks with flashing grey ones.
    // These are ignored when counting how many actual blocks are in play.
    if (block_type != 0 && block_type != 2 && block_type != 6
        && block_type != 10 && block_type != 14) {
      ++num_blocks;
    }
  }

  int level = readMappedRam(&system, 0xf09d);
  // The game is over if we miss too many blocks, the block area is full or we
  // reach level 100 (which is indicated as decimal 0x99).
  m_terminal = (max_misses > 0 && misses == max_misses)
      || (game_active && num_blocks == 25) || level == 0x99;
}

bool KlaxSettings::isTerminal() const { return m_terminal; }

reward_t KlaxSettings::getReward() const { return m_reward; }

bool KlaxSettings::isMinimal(const Action& a) const {
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

void KlaxSettings::reset() {
  m_reward = 0;
  m_score = 0;
  m_terminal = false;
}

void KlaxSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

void KlaxSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

ModeVect KlaxSettings::getAvailableModes() {
  return {0, 1, 2};
}

void KlaxSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {
  if (m < 3) {
    // Press select until the correct mode is reached.
    while (readMappedRam(&system, 0xf0ea) != m) {
      environment->pressSelect(2);
    }

    // Reset the environment to apply changes.
    environment->softReset();
  } else {
    throw std::runtime_error("This game mode is not supported.");
  }
}

ActionVect KlaxSettings::getStartingActions() {
  return {PLAYER_A_FIRE, PLAYER_A_NOOP};
}

}  // namespace ale
