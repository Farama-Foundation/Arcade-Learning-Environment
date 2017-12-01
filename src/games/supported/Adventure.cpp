/* *****************************************************************************
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
#include "Adventure.hpp"

#include "../RomUtils.hpp"

AdventureSettings::AdventureSettings() {

  reset();
}

/* create a new instance of the rom */
RomSettings* AdventureSettings::clone() const {

  RomSettings* rval = new AdventureSettings();
  *rval = *this;
  return rval;
}

/* process the latest information from ALE */
void AdventureSettings::step(const System& system) {

  int chalice_status = readRam(&system, 0xB9);
  bool chalice_in_yellow_castle = chalice_status == 0x12;

  if (chalice_in_yellow_castle) {
    m_reward = 1;
  }

  int player_status = readRam(&system, 0xE0);
  bool player_eaten = player_status == 2;

  m_terminal = player_eaten || chalice_in_yellow_castle;
}

/* is end of game */
bool AdventureSettings::isTerminal() const {

  return m_terminal;
}

/* get the most recently observed reward */
reward_t AdventureSettings::getReward() const {

  return m_reward;
}

/* is an action part of the minimal set? */
bool AdventureSettings::isMinimal(const Action &a) const {

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

/* reset the state of the game */
void AdventureSettings::reset() {

  m_reward = 0;
  m_terminal = false;
}

/* saves the state of the rom settings */
void AdventureSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void AdventureSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_terminal = ser.getBool();
}

