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
 * Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team,
 *               2017 Google Inc
 *
 * *****************************************************************************
 */

#include "ale/games/supported/Atlantis2.hpp"

#include "ale/games/RomUtils.hpp"

namespace ale {
using namespace stella;

Atlantis2Settings::Atlantis2Settings() { reset(); }

/* create a new instance of the rom */
RomSettings* Atlantis2Settings::clone() const {
  return new Atlantis2Settings(*this);
}

/* process the latest information from ALE */
void Atlantis2Settings::step(const System& system) {
  // update the reward

  // update terminal status
  m_lives = readRam(&system, 0xF1);
  m_terminal = (m_lives == 0xFF);

  // ignore terminal score.
  if (m_terminal) {
    m_reward = 0;
  } else {
    reward_t score = getDecimalScore(0xA1, 0xA2, 0xA3, &system);
    m_reward = score - m_score;
    m_score = score;
  }
}

/* is end of game */
bool Atlantis2Settings::isTerminal() const { return m_terminal; }

/* get the most recently observed reward */
reward_t Atlantis2Settings::getReward() const { return m_reward; }

/* is an action part of the minimal set? */
bool Atlantis2Settings::isMinimal(const Action& a) const {
  switch (a) {
    case PLAYER_A_NOOP:
    case PLAYER_A_FIRE:
    case PLAYER_A_RIGHTFIRE:
    case PLAYER_A_LEFTFIRE:
      return true;
    default:
      return false;
  }
}

/* reset the state of the game */
void Atlantis2Settings::reset() {
  m_reward = 0;
  m_score = 0;
  m_terminal = false;
}

/* saves the state of the rom settings */
void Atlantis2Settings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void Atlantis2Settings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

}  // namespace ale
