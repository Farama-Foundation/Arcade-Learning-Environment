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

#include "ale/games/supported/Crossbow.hpp"

#include "ale/games/RomUtils.hpp"

namespace ale {
using namespace stella;

CrossbowSettings::CrossbowSettings() { reset(); }

RomSettings* CrossbowSettings::clone() const {
  return new CrossbowSettings(*this);
}

void CrossbowSettings::step(const System& system) {
  int score = getDecimalScore(0x8D, 0x8C, 0x8B, &system);
  m_reward = score - m_score;
  m_score = score;
  // The game does have lives but the method for updating these is opaque.
  // RAM address 0xE7 appears to be a game mode flag which is set explicitly
  // when entering the front end (0x80), level select (0x81), in-game (0x00)
  // and the game over screen (0x82). We consider 'game over' to be terminal.
  m_terminal = readRam(&system, 0xE7) == 0x82;
}

bool CrossbowSettings::isTerminal() const { return m_terminal; }

reward_t CrossbowSettings::getReward() const { return m_reward; }

bool CrossbowSettings::isMinimal(const Action& a) const {
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

void CrossbowSettings::reset() {
  m_reward = 0;
  m_score = 0;
  m_terminal = false;
}

void CrossbowSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

void CrossbowSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

// According to https://atariage.com/manual_html_page.php?SoftwareID=956
// the left difficulty switch is used to control the speed of the crossbow.
DifficultyVect CrossbowSettings::getAvailableDifficulties() {
  return {0, 1};
}

// According to https://atariage.com/manual_html_page.php?SoftwareID=956
// there are four single player game modes which alter whether the crossbow
// sight is visible and whether your friends are protected from friendly fire.
ModeVect CrossbowSettings::getAvailableModes() {
  return {0, 2, 4, 6};
}

void CrossbowSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {
  if (m == 0 || m == 2 || m == 4 || m == 6) {
    // Read the currently selected mode.
    unsigned char mode = readRam(&system, 0x8D) - 1;

    // Press select until the correct mode is reached.
    while (mode != m) {
      environment->pressSelect(2);
      mode = readRam(&system, 0x8D) - 1;
    }

    // Reset the environment to apply changes.
    environment->softReset();
  } else {
    throw std::runtime_error("This game mode is not supported.");
  }
}

}  // namespace ale
