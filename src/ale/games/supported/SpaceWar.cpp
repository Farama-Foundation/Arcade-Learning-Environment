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

#include "ale/games/supported/SpaceWar.hpp"

#include "ale/games/RomUtils.hpp"

namespace ale {
using namespace stella;

SpaceWarSettings::SpaceWarSettings() { reset(); }

RomSettings* SpaceWarSettings::clone() const {
  return new SpaceWarSettings(*this);
}

void SpaceWarSettings::step(const System& system) {
  int score = getDecimalScore(0xa7, &system);
  m_reward = score - m_score;
  m_score = score;
  // Game terminates either when the player gets 10 points or the 10 minute
  // timer expires. The timer counts up every 256 vsyncs, incrementing from 0x74
  // until it wraps around to 0x00. 35840 vsyncs ~= 600 seconds = 10 minutes.
  int timer = readRam(&system, 0x80);
  m_terminal = score == 10 || timer == 0;
}

bool SpaceWarSettings::isTerminal() const { return m_terminal; }

reward_t SpaceWarSettings::getReward() const { return m_reward; }

bool SpaceWarSettings::isMinimal(const Action& a) const {
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

void SpaceWarSettings::reset() {
  m_reward = 0;
  m_score = 0;
  m_terminal = false;
}

void SpaceWarSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

void SpaceWarSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

// According to https://atariage.com/manual_html_page.php?SoftwareLabelID=470
// game modes 1-7 are Space War (shooting) games, modes 8-13 are 2-player games
// and 14-17 are 1-player Space Shuttle (docking) games. In modes 1-5 it is not
// possible to obtain score high enough to end the game (10) without input from
// the second player, as player one will run out of missiles that will not be
// replenished. We therefore remove the first five modes but the rest [6-17] are
// valid, with the second (inert) player acting as a distractor when present.
ModeVect SpaceWarSettings::getAvailableModes() {
  return {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
}

void SpaceWarSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {
  if (isModeSupported(m)) {
    // Press select until the correct mode is reached.
    while (getDecimalScore(0xa7, &system) != m) {
      environment->pressSelect(2);
    }

    // Reset the environment to apply changes.
    environment->softReset();
  } else {
    throw std::runtime_error("This game mode is not supported.");
  }
}

}  // namespace ale
