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

#include "DemonAttack.hpp"

#include "../RomUtils.hpp"

namespace ale {

DemonAttackSettings::DemonAttackSettings() { reset(); }

/* create a new instance of the rom */
RomSettings* DemonAttackSettings::clone() const {
  return new DemonAttackSettings(*this);
}

/* process the latest information from ALE */
void DemonAttackSettings::step(const System& system) {
  // update the reward
  int score = getDecimalScore(0x85, 0x83, 0x81, &system);
  // MGB: something funny with the RAM; it is not initialized to 0?
  if (readRam(&system, 0x81) == 0xAB && readRam(&system, 0x83) == 0xCD &&
      readRam(&system, 0x85) == 0xEA)
    score = 0;
  m_reward = score - m_score;
  m_score = score;

  // update terminal status
  int lives_displayed = readRam(&system, 0xF2);
  int display_flag = readRam(&system, 0xF1);
  // for terminal checking, we must make sure that we do not detect incorrectly a level change as a game-over
  m_terminal =
      (lives_displayed == 0) && display_flag == 0xBD && !m_level_change;
  m_lives = lives_displayed +
            1; // Once we reach terminal, lives() will correctly return 0
  m_level_change = false;
}

/* is end of game */
bool DemonAttackSettings::isTerminal() const { return m_terminal; };

/* get the most recently observed reward */
reward_t DemonAttackSettings::getReward() const { return m_reward; }

/* is an action part of the minimal set? */
bool DemonAttackSettings::isMinimal(const Action& a) const {
  switch (a) {
    case PLAYER_A_NOOP:
    case PLAYER_A_FIRE:
    case PLAYER_A_RIGHT:
    case PLAYER_A_LEFT:
    case PLAYER_A_RIGHTFIRE:
    case PLAYER_A_LEFTFIRE:
      return true;
    default:
      return false;
  }
}

/* reset the state of the game */
void DemonAttackSettings::reset() {
  m_reward = 0;
  m_score = 0;
  m_terminal = false;
  m_lives = 4;
  m_level_change = false;
}

/* saves the state of the rom settings */
void DemonAttackSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putInt(m_lives);
}

// loads the state of the rom settings
void DemonAttackSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_lives = ser.getInt();
}

// returns a list of mode that the game can be played in
ModeVect DemonAttackSettings::getAvailableModes() {
  return {1, 3, 5, 7};
}

// set the mode of the game
// the given mode must be one returned by the previous function
void DemonAttackSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {
  if (m == 0) {
    m = 1; // The default mode is not valid here
  }
  if (m == 1 || m == 3 || m == 5 || m == 7) {
    // read the mode we are currently in
    unsigned char mode = readRam(&system, 0xEA);
    // press select until the correct mode is reached
    while (mode != m) {
      environment->pressSelect(1);
      mode = readRam(&system, 0xEA);
    }
    m_level_change = true;
    //reset the environment to apply changes.
    environment->softReset();
  } else {
    throw std::runtime_error("This mode doesn't currently exist for this game");
  }
}

DifficultyVect DemonAttackSettings::getAvailableDifficulties() {
  return {0, 1};
}

}  // namespace ale
