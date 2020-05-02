/* *****************************************************************************
 * The method lives() is based on Xitari's code, from Google Inc.
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

#include "SpaceInvaders.hpp"

#include "../RomUtils.hpp"

namespace ale {


SpaceInvadersSettings::SpaceInvadersSettings() { reset(); }

/* create a new instance of the rom */
RomSettings* SpaceInvadersSettings::clone() const {
  return new SpaceInvadersSettings(*this);
}

/* process the latest information from ALE */
void SpaceInvadersSettings::step(const System& system) {
  // update the reward
  int score = getDecimalScore(0xE8, 0xE6, &system);
  int scoreP2 = getDecimalScore(0xE9, 0xE7, &system);
  // reward cannot get negative in this game. When it does, it means that the score has looped
  // (overflow)
  // 10000 is the highest possible score
  const int maximumScore = 10000;

  m_reward_p2 = scoreP2 - m_score_p2;
  if (m_reward_p2 < 0) {
    m_reward_p2 = (maximumScore - m_score_p2) + scoreP2;
  }
  m_score_p2 = scoreP2;

  m_reward = score - m_score;
  if (m_reward < 0) {
    m_reward = (maximumScore - m_score) + score;
  }
  m_score = score;

  m_lives = readRam(&system, 0xC9) - 1;

  // update terminal status
  // If bit 0x80 is on, then game is over
  int some_byte = readRam(&system, 0x98);
  m_terminal = (some_byte & 0x80) || m_lives == -1;
}

/* is end of game */
bool SpaceInvadersSettings::isTerminal() const { return m_terminal; };

int SpaceInvadersSettings::lives() {
 return m_lives;
}
int SpaceInvadersSettings::livesP2() {
  //wierd but correct, both players share the same number of lives even though
  //their rewards are different
   return m_lives;
}

/* get the most recently observed reward */
reward_t SpaceInvadersSettings::getReward() const { return m_reward; }
reward_t SpaceInvadersSettings::getRewardP2() const { return m_reward_p2; }

/* is an action part of the minimal set? */
bool SpaceInvadersSettings::isMinimal(const Action& a) const {
  switch (a) {
    case PLAYER_A_NOOP:
    case PLAYER_A_LEFT:
    case PLAYER_A_RIGHT:
    case PLAYER_A_FIRE:
    case PLAYER_A_LEFTFIRE:
    case PLAYER_A_RIGHTFIRE:
      return true;
    default:
      return false;
  }
}

/* reset the state of the game */
void SpaceInvadersSettings::reset() {
  m_reward = 0;
  m_reward_p2 = 0;
  m_score = 0;
  m_score_p2 = 0;
  m_terminal = false;
  m_lives = 2;
}

/* saves the state of the rom settings */
void SpaceInvadersSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_reward_p2);
  ser.putInt(m_score);
  ser.putInt(m_score_p2);
  ser.putBool(m_terminal);
  ser.putInt(m_lives);
}

// loads the state of the rom settings
void SpaceInvadersSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_reward_p2 = ser.getInt();
  m_score = ser.getInt();
  m_score_p2 = ser.getInt();
  m_terminal = ser.getBool();
  m_lives = ser.getInt();
}

// returns a list of mode that the game can be played in
ModeVect SpaceInvadersSettings::getAvailableModes() {
  // all modes from 1 - 16
  ModeVect modes(16);
  for (unsigned int i = 0; i < 16; i++) {
    modes[i] = i+1;
  }
  return modes;
}
ModeVect SpaceInvadersSettings::get2PlayerModes() {
  // all the modes from 33 - 64
  ModeVect modes(32);
  for (unsigned int i = 32; i < 64; i++) {
    modes[i-32] = i+1;
  }
  return modes;
}

// set the mode of the game
// the given mode must be one returned by the previous function
void SpaceInvadersSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {

  game_mode_t target = m - 1;
  // read the mode we are currently in
  // press select until the correct mode is reached
  while (readRam(&system, 0xDC) != target) {
    environment->pressSelect(2);
  }
  //reset the environment to apply changes.
  environment->softReset();
}

DifficultyVect SpaceInvadersSettings::getAvailableDifficulties() {
  return {0, 1};
}

}  // namespace ale
