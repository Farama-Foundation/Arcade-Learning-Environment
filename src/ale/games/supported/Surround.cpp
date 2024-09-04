/* *****************************************************************************
 * Xitari
 *
 * Copyright 2014 Google Inc.
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
#include "ale/games/supported/Surround.hpp"

#include "ale/games/RomUtils.hpp"

namespace ale {
using namespace stella;

SurroundSettings::SurroundSettings() { reset(); }

/* create a new instance of the rom */
RomSettings* SurroundSettings::clone() const {
  return new SurroundSettings(*this);
}

/* process the latest information from ALE */
void SurroundSettings::step(const System& system) {
  int theirScore = getDecimalScore(0xF6, &system);
  int myScore = getDecimalScore(0xF7, &system);

  // The RL score is the difference in scores
  int newScore = myScore - theirScore;
  m_reward = newScore - m_score;
  m_score = newScore;

  // The game ends when we reach 10 points
  m_terminal = (theirScore == 10 || myScore == 10);
}

/* is end of game */
bool SurroundSettings::isTerminal() const { return m_terminal; }

/* get the most recently observed reward */
reward_t SurroundSettings::getReward() const { return m_reward; }

/* is an action part of the minimal set? */
bool SurroundSettings::isMinimal(const Action& a) const {
  switch (a) {
    case PLAYER_A_NOOP:
    case PLAYER_A_LEFT:
    case PLAYER_A_RIGHT:
    case PLAYER_A_UP:
    case PLAYER_A_DOWN:
      return true;
    default:
      return false;
  }
}

/* reset the state of the game */
void SurroundSettings::reset() {
    m_reward   = 0;
    m_score    = 0;
    m_terminal = false;
}

/* saves the state of the rom settings */
void SurroundSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void SurroundSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

// https://atariage.com/manual_html_page.php?SoftwareLabelID=535
// Left difficulty switch sets the computer skill level, whilst the right
// difficulty switch prevents the player from backing onto their previous
// track block (making the game easier).
DifficultyVect SurroundSettings::getAvailableDifficulties() {
  return {0, 1, 2, 3};
}

// https://atariage.com/manual_html_page.php?SoftwareLabelID=535
// There are only two single player modes, the second is faster than the first.
ModeVect SurroundSettings::getAvailableModes() {
  return {0, 2};
}

void SurroundSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {
  if (m == 0 || m == 2) {
    // Read the game mode from RAM address 0xf9.
    unsigned char mode = readRam(&system, 0xf9);
    int desired_mode = m + 1;

    // Press select until the correct mode is reached for single player only.
    while (mode != desired_mode) {
      environment->pressSelect(2);
      mode = readRam(&system, 0xf9);
    }

    // Reset the environment to apply changes.
    environment->softReset();
  } else {
    throw std::runtime_error("This game mode is not supported.");
  }
}

}  // namespace ale
