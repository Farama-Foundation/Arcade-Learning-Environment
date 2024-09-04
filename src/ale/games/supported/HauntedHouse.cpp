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

#include "ale/games/supported/HauntedHouse.hpp"

#include "ale/games/RomUtils.hpp"

namespace ale {
using namespace stella;

HauntedHouseSettings::HauntedHouseSettings() { reset(); }

RomSettings* HauntedHouseSettings::clone() const {
  return new HauntedHouseSettings(*this);
}

void HauntedHouseSettings::step(const System& system) {
  m_reward = 0;
  int matches = getDecimalScore(0x82, &system);
  // This value is not clamped so will wrap around from 99 to 0 but still means
  // that we used a match. Note we can't use multiple matches per frame.
  if (matches != m_matches) {
    // Penalty of -1 for using a match.
    --m_reward;
    m_matches = matches;
  }
  // Track number of lives left.
  m_lives = readRam(&system, 0x96);;
  // Win state is when the completed urn is taken to the exit of the mansion.
  bool escaped_with_urn = readRam(&system, 0x99) == 0x44;
  if (escaped_with_urn) {
    // Give an arbitrary reward of +100 when completing the game.
    m_reward += 100;
  }
  // Game ends when running out of lives or escaping with the urn.
  m_terminal = m_lives == 0 || escaped_with_urn;
}

bool HauntedHouseSettings::isTerminal() const { return m_terminal; }

reward_t HauntedHouseSettings::getReward() const { return m_reward; }

bool HauntedHouseSettings::isMinimal(const Action& a) const {
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

void HauntedHouseSettings::reset() {
  m_reward = 0;
  m_matches = 0;
  m_lives = 9;
  m_terminal = false;
}

void HauntedHouseSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_matches);
  ser.putInt(m_lives);
  ser.putBool(m_terminal);
}

void HauntedHouseSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_matches = ser.getInt();
  m_lives = ser.getInt();
  m_terminal = ser.getBool();
}

// According to https://atariage.com/manual_html_page.php?SoftwareLabelID=233
// the left difficulty switch sets whether there are periodic flashes of
// lightning to make the game easier.
DifficultyVect HauntedHouseSettings::getAvailableDifficulties() {
  return {0, 1};
}

// According to https://atariage.com/manual_html_page.php?SoftwareLabelID=233
// there are 9 variations of the game which vary the number and speed of
// enemies, whether there are locked doors and whether items are randomised.
ModeVect HauntedHouseSettings::getAvailableModes() {
  return {0, 1, 2, 3, 4, 5, 6, 7, 8};
}

void HauntedHouseSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {
  if (m < 9) {
    while (true) {
      // Read the mode we are currently in.
      unsigned char mode = readRam(&system, 0xcc);

      // Press select until the correct mode is reached.
      if (mode == m) {
        break;
      }
      environment->pressSelect(2);
    }

    // Reset the environment to apply changes.
    environment->softReset();
  } else {
    throw std::runtime_error("This game mode is not supported.");
  }
}

}  // namespace ale
