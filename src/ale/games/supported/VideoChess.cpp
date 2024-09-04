/* *****************************************************************************
 *
 * This wrapper is based on code authored by Stig Petersen, March 2014
 *
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

#include "ale/games/supported/VideoChess.hpp"

#include "ale/games/RomUtils.hpp"

namespace ale {
using namespace stella;

VideoChessSettings::VideoChessSettings() { reset(); }

/* create a new instance of the rom */
RomSettings* VideoChessSettings::clone() const {
  return new VideoChessSettings(*this);
}

/* process the latest information from ALE */
void VideoChessSettings::step(const System& system) {
  // TURN_BLACK = 0x0;
  const int TURN_WHITE = 0x82;
  int currentPlayer = readRam(&system, 0xE1);

  if (currentPlayer == TURN_WHITE) {
    m_reward = 0;

    // 0xEE == 0: check mate black
    // 0xEE == 1: check mate white
    // 0xEE == 3: game ongoing
    const int CHECKMATE_BLACK = 0x00;
    const int CHECKMATE_WHITE = 0x01;
    int checkMateByte = readRam(&system, 0xEE);

    const int CHECKMATE_REWARD = 1;

    if (checkMateByte == CHECKMATE_BLACK) {
      m_reward += CHECKMATE_REWARD;
      m_terminal = true;
    } else if (checkMateByte == CHECKMATE_WHITE) {
      m_reward -= CHECKMATE_REWARD;
      m_terminal = true;
    }
  } else {
    // Atari AI simulates moves while it searches the tree, so we want to
    // ignore those.
    m_reward = 0;
  }

  // 0xE4: Number of moves by black
  // Notes on addresses that change when players make a move
  // E0*, E1=82/0 , E2*, E3*, E4*, E5=FF/80/0C, E6*, E7*, E8*, E9*, EA*
  // EB=40/98, EC*, ED*, *EE, *EF
  // *change wildly during colour flashes
}

/* is end of game */
bool VideoChessSettings::isTerminal() const { return m_terminal; }

/* get the most recently observed reward */
reward_t VideoChessSettings::getReward() const { return m_reward; }

/* is an action part of the minimal set? */
bool VideoChessSettings::isMinimal(const Action& a) const {
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
      return true;
    default:
      return false;
  }
}

/* reset the state of the game */
void VideoChessSettings::reset() {
  m_reward = 0;
  m_terminal = false;
  m_lives = 1;
}

/* saves the state of the rom settings */
void VideoChessSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putBool(m_terminal);
  ser.putInt(m_lives);
}

// loads the state of the rom settings
void VideoChessSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_terminal = ser.getBool();
  m_lives = ser.getInt();
}

// According to Wikipedia (https://en.wikipedia.org/wiki/Video_Chess) has 8
// available difficulty levels which are set via game mode. The level determines
// how long the computer 'thinks' about a move, ranging from 10 seconds to 10
// hours. However, the higher difficulty levels are known to have bugs that can
// cause the computer to make multiple moves per turn, so limiting to the first
// 5 to be safe.

// Returns the five available game modes.
ModeVect VideoChessSettings::getAvailableModes() {
  return {0, 1, 2, 3, 4};
}

// Set the game mode.
// The given mode must be one returned by the previous function.
void VideoChessSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {
  if (m < 5) {
    // read the mode we are currently in
    unsigned char mode = readRam(&system, 0xEA);
    // press select until the correct mode is reached
    while (mode != m) {
      environment->pressSelect(1);
      mode = readRam(&system, 0xEA);
    }
    // reset the environment to apply changes.
    environment->softReset();
  } else {
    throw std::runtime_error("This game mode is not supported.");
  }
}

}  // namespace ale
