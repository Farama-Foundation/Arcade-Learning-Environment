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
#include "ale/games/supported/WordZapper.hpp"

#include "ale/games/RomUtils.hpp"

namespace ale {
using namespace stella;

WordZapperSettings::WordZapperSettings() { reset(); }

RomSettings* WordZapperSettings::clone() const {
  return new WordZapperSettings(*this);
}

void WordZapperSettings::step(const System& system) {
  int wall_clock = getDecimalScore(0xcf, &system);
  // If the wall clock isn't running then our game state is not valid.
  if (wall_clock > 0) {
    // Rounds remaining (out of 3) is in RAM address 0xdc.
    // After the final round this value will be 0xff (wrapped round from 0x00),
    // so we must treat this value as a signed char to get the correct score we
    // require for reward.
    int score = 2 - static_cast<signed char>(readRam(&system, 0xdc));
    m_reward = score - m_score;
    m_score = score;
    // Game terminates after three rounds, after the time runs out or the player
    // gets hit by the Doomsday asteroid. The timer is set to zero after being
    // hit by the asteroid, so this is a sufficient termination test.
    int time_remaining = getDecimalScore(0xde, &system);
    m_terminal = score == 3 || time_remaining == 0;
  }
}

bool WordZapperSettings::isTerminal() const { return m_terminal; }

reward_t WordZapperSettings::getReward() const { return m_reward; }

bool WordZapperSettings::isMinimal(const Action& a) const {
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

void WordZapperSettings::reset() {
  m_reward = 0;
  m_score = 0;
  m_terminal = false;
}

void WordZapperSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

void WordZapperSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

ActionVect WordZapperSettings::getStartingActions() {
  // When this ROM is booted there is a short animation sequence before any
  // user input is accepted, even the 'start' button. This lasts for around
  // 8 seconds so we wait for 486 frames to be sure our subsequent action to
  // press 'fire' is effective.
  ActionVect startingActions(486, PLAYER_A_NOOP);

  // Press fire to start and wait a couple of frames.
  startingActions.push_back(PLAYER_A_FIRE);
  startingActions.push_back(PLAYER_A_NOOP);
  startingActions.push_back(PLAYER_A_NOOP);

  return startingActions;
}

// According to https://atariage.com/manual_html_page.php?SoftwareID=1448
// the left difficulty switch sets whether the scroller gets scrambled when the
// player hits a scroller asteroid. The right difficulty switch sets whether
// the Doomsday asteroid is present in the game.
DifficultyVect WordZapperSettings::getAvailableDifficulties() {
  return {0, 1, 2, 3};
}

// According to https://atariage.com/manual_html_page.php?SoftwareID=1448
// there are 24 variations of the game that alter the scroll speed, matching
// real words vs random letter strings, meteor speed, meteor density and
// whether a 'freebie' is present to begin the game.
ModeVect WordZapperSettings::getAvailableModes() {
  return {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
          12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
}

void WordZapperSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {
  if (m < 24) {
    // Read the mode we are currently in.
    unsigned char mode = readRam(&system, 0xdb);

    // Press select until the correct mode is reached.
    while (mode != m) {
      environment->pressSelect(2);
      mode = readRam(&system, 0xdb);
    }

    // Reset the environment to apply changes.
    environment->softReset();
  } else {
    throw std::runtime_error("This game mode is not supported.");
  }
}

}  // namespace ale
