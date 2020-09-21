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

#include "games/supported/Gopher.hpp"

#include "games/RomUtils.hpp"

namespace ale {

GopherSettings::GopherSettings() { reset(); }

/* create a new instance of the rom */
RomSettings* GopherSettings::clone() const {
  return new GopherSettings(*this);
}

/* process the latest information from ALE */
void GopherSettings::step(const System& system) {
  // update the reward
  int score = getDecimalScore(0xB2, 0xB1, 0xB0, &system);
  int reward = score - m_score;
  m_reward = reward;
  m_score = score;

  // update terminal status
  int carrot_bits = readRam(&system, 0xB4) & 0x7;
  m_terminal = carrot_bits == 0;

  // A very crude popcount
  static int livesFromCarrots[] = {0, 1, 1, 2, 1, 2, 2, 3};
  m_lives = livesFromCarrots[carrot_bits];
}

/* is end of game */
bool GopherSettings::isTerminal() const { return m_terminal; };

/* get the most recently observed reward */
reward_t GopherSettings::getReward() const { return m_reward; }

/* is an action part of the minimal set? */
bool GopherSettings::isMinimal(const Action& a) const {
  switch (a) {
    case PLAYER_A_NOOP:
    case PLAYER_A_FIRE:
    case PLAYER_A_UP:
    case PLAYER_A_RIGHT:
    case PLAYER_A_LEFT:
    case PLAYER_A_UPFIRE:
    case PLAYER_A_RIGHTFIRE:
    case PLAYER_A_LEFTFIRE:
      return true;
    default:
      return false;
  }
}

/* reset the state of the game */
void GopherSettings::reset() {
  m_reward = 0;
  m_score = 0;
  m_terminal = false;
  m_lives = 3;
}

/* saves the state of the rom settings */
void GopherSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putInt(m_lives);
}

// loads the state of the rom settings
void GopherSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_lives = ser.getInt();
}

ActionVect GopherSettings::getStartingActions() {
  return {PLAYER_A_FIRE};
}

// returns a list of mode that the game can be played in
ModeVect GopherSettings::getAvailableModes() {
  return {0, 2};
}

// set the mode of the game
// the given mode must be one returned by the previous function
void GopherSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {
  if (m == 0 || m == 2) {
    environment->softReset();
    // read the mode we are currently in
    unsigned char mode = readRam(&system, 0xD3);
    // press select until the correct mode is reached
    while (mode != m) {
      environment->pressSelect(5);
      mode = readRam(&system, 0xD3);
    }
    //reset the environment to apply changes.
    environment->softReset();
  } else {
    throw std::runtime_error("This mode doesn't currently exist for this game");
  }
}

DifficultyVect GopherSettings::getAvailableDifficulties() {
  return {0, 1};
}

}  // namespace ale
