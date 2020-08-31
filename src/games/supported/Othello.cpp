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

#include "Othello.hpp"

#include "../RomUtils.hpp"

namespace ale {

OthelloSettings::OthelloSettings() { reset(); }

RomSettings* OthelloSettings::clone() const {
  return new OthelloSettings(*this);
}

void OthelloSettings::step(const System& system) {
  int white_score = getDecimalScore(0xce, &system);
  int black_score = getDecimalScore(0xd0, &system);
  int score = white_score - black_score;
  m_reward_m1 = score - m_score;
  m_reward_m2 = -m_reward_m1;
  m_score = score;

  // Player indicator is zero if game is over
  if (readRam(&system, 0xc0) == 0) {
    ++m_cursor_inactive;
  } else {
    m_cursor_inactive = 0;
  }

  // The game is over when there are no more valid moves not necessarily when
  // the board is full of counters. We must also wait for the counters to reach
  // their final colour for scoring. We detect this when the cursor stops
  // flashing for at least one second, signalling no more player input.
  m_terminal = m_cursor_inactive > 75;

  if(two_player_mode){
    if (m_reward_m1 != 0){
      // reward is non-zero every time there is a turn change
      turn_same_count = 0;
    }
    turn_same_count += 1;
    // 15 second timer to move
    if (max_turn_time > 0 && turn_same_count >= max_turn_time){
      unsigned char active_player = readRam(&system, 0xc0);
      if (active_player == 0xff){
        m_reward_m1 = -1;
      }
      else{
        m_reward_m2 = -1;
      }
      turn_same_count = 0;
    }
  }
}

bool OthelloSettings::isTerminal() const { return m_terminal; }

reward_t OthelloSettings::getReward() const { return m_reward_m1; }
reward_t OthelloSettings::getRewardP2() const { return m_reward_m2; }

bool OthelloSettings::isMinimal(const Action& a) const {
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
      // The joystick is used to move the active square on the board, then
      // fire selects that square. It does not make sense to press fire and
      // move at the same time.
      return true;
    default:
      return false;
  }
}

void OthelloSettings::reset() {
  m_reward_m1 = 0;
  m_reward_m2 = 0;
  m_score = 0;
  turn_same_count = 0;
  m_terminal = false;
  m_cursor_inactive = 0;
}

void OthelloSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward_m1);
  ser.putInt(m_reward_m2);
  ser.putInt(m_score);
  ser.putInt(turn_same_count);
  ser.putBool(m_terminal);
  ser.putBool(two_player_mode);
  ser.putInt(m_cursor_inactive);
  ser.putInt(max_turn_time);
}

void OthelloSettings::loadState(Deserializer& ser) {
  m_reward_m1 = ser.getInt();
  m_reward_m2 = ser.getInt();
  m_score = ser.getInt();
  turn_same_count = ser.getInt();
  m_terminal = ser.getBool();
  two_player_mode = ser.getBool();
  m_cursor_inactive = ser.getInt();
  max_turn_time = ser.getInt();
}

// According to https://atariage.com/manual_html_page.php?SoftwareLabelID=931
// The left difficulty switch must be kept in position 'b' for normal gameplay,
// whilst the right difficulty switch determines whether white (player) or black
// (computer) makes the first move.
DifficultyVect OthelloSettings::getAvailableDifficulties() {
  return {0, 2};
}

// According to https://atariage.com/manual_html_page.php?SoftwareLabelID=931
// there are three one player game modes which set the skill level as
// beginner, intermediate and expert.
ModeVect OthelloSettings::getAvailableModes() {
  return {1, 2, 3};
}
ModeVect OthelloSettings::get2PlayerModes() {
  return {4};
}

void OthelloSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {

  two_player_mode = (m == 4);
  if (m <= 4) {
    // Read the mode we are currently in.
    unsigned char mode = readRam(&system, 0xde);

    // Press select until the correct mode is reached.
    while (mode != m) {
      environment->pressSelect(2);
      mode = readRam(&system, 0xde);
    }

    // Reset the environment to apply changes.
    environment->softReset();
  } else {
    throw std::runtime_error("This game mode is not supported.");
  }
}


void OthelloSettings::modifyEnvironmentSettings(Settings& settings) {
  int default_setting = -1;
  max_turn_time = settings.getInt("max_turn_time");
  if(max_turn_time == default_setting){
    const int DEFAULT_STALL_LIMIT = 60*10;
    max_turn_time = DEFAULT_STALL_LIMIT;
  }
}


}  // namespace ale
