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

#include "ale/games/supported/MiniatureGolf.hpp"

#include "ale/games/RomUtils.hpp"

namespace ale {
using namespace stella;

MiniatureGolfSettings::MiniatureGolfSettings() { reset(); }

RomSettings* MiniatureGolfSettings::clone() const {
  return new MiniatureGolfSettings(*this);
}

void MiniatureGolfSettings::updateRewardWhenLevelFinishes(int levelNumber) {
  if (levelNumber != m_levelNumber) {
    // Level has just completed.

    // Update the reward.
    int totalHits = m_leftStatus + m_hits;

    int previousLevelHits = totalHits - m_hitsAtStartOfLevel;
    if (previousLevelHits > 0) {
      m_reward = m_levelPar - previousLevelHits;
    }

    // Terminate if the whole game finished.
    if (levelNumber == 0) {
      m_terminal = true;
    }

    // Reset level data.
    m_levelNumber = levelNumber;
    m_hits = 0;
    m_hitsAtStartOfLevel = m_leftStatus;
  }
}

void MiniatureGolfSettings::step(const System& system) {
  m_reward = 0;
  int leftStatus = getDecimalScore(0x87, &system);
  int rightStatus = getDecimalScore(0x88, &system);
  int levelNumber = getDecimalScore(0xAF, &system);

  updateRewardWhenLevelFinishes(levelNumber);

  if (rightStatus != 0) {
    // We are in the lobby mode which happens
    // when we switch to new level but level
    // has not started yet.
    // In this mode the left status displays level number
    // and the right status displays PAR for this level.
    m_levelPar = rightStatus;
  } else {
    // When the right status is 0 (and levelNumber in [1..9]) we
    // are in playing mode where left status displays the cumulative
    // number of hits since the start of the game.
    if (leftStatus < m_leftStatus) {
      // Points in the left status wrap around after 99.
      // When that happens we update the totalHits count.
      m_hits += m_leftStatus;
    }

    m_leftStatus = leftStatus;
  }
}

bool MiniatureGolfSettings::isTerminal() const { return m_terminal; }

reward_t MiniatureGolfSettings::getReward() const { return m_reward; }

bool MiniatureGolfSettings::isMinimal(const Action& a) const {
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

void MiniatureGolfSettings::reset() {
  m_reward = 0;
  m_terminal = false;
  m_levelNumber = 0;
  m_levelPar = 0;
  m_hits = 0;
  m_leftStatus = 0;
  m_hitsAtStartOfLevel = 0;
}

void MiniatureGolfSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putBool(m_terminal);
  ser.putBool(m_levelNumber);
  ser.putBool(m_levelPar);
  ser.putBool(m_hits);
  ser.putBool(m_leftStatus);
  ser.putBool(m_hitsAtStartOfLevel);
}

void MiniatureGolfSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_terminal = ser.getBool();
  m_levelNumber = ser.getInt();
  m_levelPar = ser.getInt();
  m_hits = ser.getInt();
  m_leftStatus = ser.getInt();
  m_hitsAtStartOfLevel = ser.getInt();
}

// According to https://atariage.com/manual_html_page.php?SoftwareLabelID=30
// The ball travels a much longer distance when the difficulty switch is in the
// "a" position.
DifficultyVect MiniatureGolfSettings::getAvailableDifficulties() {
  return {0, 1};
}

}  // namespace ale
