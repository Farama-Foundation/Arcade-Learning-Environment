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

#include "ale/games/supported/Et.hpp"

#include "ale/games/RomUtils.hpp"

namespace ale {
using namespace stella;

EtSettings::EtSettings() { reset(); }

RomSettings* EtSettings::clone() const {
  return new EtSettings(*this);
}

void EtSettings::step(const System& system) {
  // Score is not actually awarded until the interstitial screen when either
  // ET gets in his ship or runs out of lives.
  int score = getDecimalScore(0xe1, 0xe0, 0xdf, &system);
  m_reward = score - m_score;
  m_score = score;

  // NB. energy = getDecimalScore(0xd4, 0xd3, &system);
  // Number of lives is not displayed on the screen but ET can be revived a
  // number of times after his energy runs out.
  // NB. internal counter does wrap around to 0xff to signify no lives left.
  m_lives = readRam(&system, 0xe5) + 1;

  // The game state (title screen, intro or in game) is stored at 0x80.
  int game_state = readRam(&system, 0x80);

  // Game ends when we're on the title screen with no lives left.
  m_terminal = m_lives == 0 && game_state == 8;
}

bool EtSettings::isTerminal() const { return m_terminal; }

reward_t EtSettings::getReward() const { return m_reward; }

bool EtSettings::isMinimal(const Action& a) const {
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

void EtSettings::reset() {
  m_reward = 0;
  m_score = 0;
  m_lives = 0;
  m_terminal = false;
}

void EtSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putInt(m_lives);
  ser.putBool(m_terminal);
}

void EtSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_lives = ser.getInt();
  m_terminal = ser.getBool();
}

// According to https://atariage.com/manual_html_page.php?SoftwareLabelID=157
// the left difficulty switch determines the landing conditions for ET's ship
// and the right difficulty switch determines the speed of the human opponents.
DifficultyVect EtSettings::getAvailableDifficulties() {
  return {0, 1, 2, 3};
}

// According to https://atariage.com/manual_html_page.php?SoftwareLabelID=157
// The variations are:
// Game 1--All humans present in game.
// Game 2--Elliott and FBI agent present, no scientist.
// Game 3--Only Elliott present in game.
ModeVect EtSettings::getAvailableModes() {
  return {0, 1, 2};
}

void EtSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {
  if (m < 3) {
    // Read the mode we are currently in.
    unsigned char mode = readRam(&system, 0xea) - 1;
    int attempts = 0;

    // Press select until the correct mode is reached.
    // NB. Need quite a long press time on the switch to register in ET.
    while (mode != m && attempts < 100) {
      environment->pressSelect(25);
      mode = readRam(&system, 0xea) - 1;
      attempts++;
    }

    if (mode != m) {
      throw std::runtime_error("Failed to select game mode.");
    }

    // Reset the environment to apply changes.
    environment->softReset();
  } else {
    throw std::runtime_error("This game mode is not supported.");
  }
}

}  // namespace ale
