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

#include "ale/games/supported/HumanCannonball.hpp"

#include "ale/games/RomUtils.hpp"

namespace ale {
using namespace stella;

HumanCannonballSettings::HumanCannonballSettings() { reset(); }

RomSettings* HumanCannonballSettings::clone() const {
  return new HumanCannonballSettings(*this);
}

void HumanCannonballSettings::step(const System& system) {
  int score = getDecimalScore(0xb6, &system);
  m_reward = score - m_score;
  m_score = score;
  // Game terminates either when the player gets 7 points of 7 misses.
  m_misses = getDecimalScore(0xb7, &system);
  m_terminal = score == 7 || m_misses == 7;
}

bool HumanCannonballSettings::isTerminal() const { return m_terminal; }

reward_t HumanCannonballSettings::getReward() const { return m_reward; }

bool HumanCannonballSettings::isMinimal(const Action& a) const {
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

void HumanCannonballSettings::reset() {
  m_reward = 0;
  m_score = 0;
  m_misses = 0;
  m_terminal = false;
}

void HumanCannonballSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putInt(m_misses);
  ser.putBool(m_terminal);
}

void HumanCannonballSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_misses = ser.getInt();
  m_terminal = ser.getBool();
}

// According to https://atariage.com/manual_html_page.php?SoftwareLabelID=237
// the left difficulty switch sets the width of the water tower (target).
DifficultyVect HumanCannonballSettings::getAvailableDifficulties() {
  return {0, 1};
}

// According to https://atariage.com/manual_html_page.php?SoftwareLabelID=237
// there are 8 variations of the game with both a one and two-player version.
// Game modes alter the cannon control, speed control, cannon angle control,
// whether the tower (target) is movable and the appearance of a moving window
// obstacle.
ModeVect HumanCannonballSettings::getAvailableModes() {
  return {0, 1, 2, 3, 4, 5, 6, 7};
}

void HumanCannonballSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {
  if (m < 8) {
    // Read the mode we are currently in along with number of players.
    unsigned char mode = readRam(&system, 0xb6) - 1;
    unsigned char players = readRam(&system, 0xb7);

    // Press select until the correct mode is reached for single player only.
    while (mode != m || players != 1) {
      environment->pressSelect(2);
      mode = readRam(&system, 0xb6) - 1;
      players = readRam(&system, 0xb7);
    }

    // Reset the environment to apply changes.
    environment->softReset();
  } else {
    throw std::runtime_error("This game mode is not supported.");
  }
}

}  // namespace ale
