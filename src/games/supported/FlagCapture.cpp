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

#include "FlagCapture.hpp"

#include "../RomUtils.hpp"

namespace ale {

FlagCaptureSettings::FlagCaptureSettings() { reset(); }

RomSettings* FlagCaptureSettings::clone() const {
  return new FlagCaptureSettings(*this);
}

void FlagCaptureSettings::step(const System& system) {
  if (is_two_player){
    int score_p1 = getDecimalScore(0xab, &system);
    int score_p2 = getDecimalScore(0xeb, &system);
    m_reward = score_p1 - m_score;
    m_reward_p2 = score_p2 - m_score_p2;
    m_score = score_p1;
    m_score_p2 = score_p2;
    //game terminates when score maxes out
    m_terminal = m_score == 99 || m_score_p2 == 99;
  }
  else{
    int score = getDecimalScore(0xea, &system);
    m_reward = score - m_score;
    m_score = score;
    // Game terminates when timer stored at RAM 0xeb expires after 75 seconds.
    m_terminal = getDecimalScore(0xeb, &system) == 0 || score == 99;
  }
}

bool FlagCaptureSettings::isTerminal() const { return m_terminal; }

reward_t FlagCaptureSettings::getReward() const { return m_reward; }
reward_t FlagCaptureSettings::getRewardP2() const { return m_reward_p2; }

bool FlagCaptureSettings::isMinimal(const Action& a) const {
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

void FlagCaptureSettings::reset() {
  m_reward = 0;
  m_reward_p2 = 0;
  m_score = 0;
  m_score_p2 = 0;
  m_terminal = false;
}

void FlagCaptureSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_reward_p2);
  ser.putInt(m_score);
  ser.putInt(m_score_p2);
  ser.putBool(m_terminal);
  ser.putBool(is_two_player);
}

void FlagCaptureSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_reward_p2 = ser.getInt();
  m_score = ser.getInt();
  m_score_p2 = ser.getInt();
  m_terminal = ser.getBool();
  is_two_player = ser.getBool();
}

// According to https://atariage.com/manual_html_page.php?SoftwareID=1022
// there are 10 game mode variations but only 3 of them are valid for a single
// player. These determine whether the flag is stationary or moving and are
// timed against a fixed 75 second clock.
ModeVect FlagCaptureSettings::getAvailableModes() {
  return {8, 9, 10};
}
ModeVect FlagCaptureSettings::get2PlayerModes() {
  return {1, 2, 3, 4};
}
void FlagCaptureSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {

    // Read the mode we are currently in.
    int mode = readRam(&system, 0xd6);
    // Single player modes are [8, 10]
    int desired_mode = m;

    // Press select until the correct mode is reached for single player only.
    while (mode != desired_mode) {
      environment->pressSelect(10);
      mode = readRam(&system, 0xd6);
    }

    is_two_player = m <= 4;

    // Reset the environment to apply changes.
    environment->softReset();
}

ActionVect FlagCaptureSettings::getStartingActions() {
  ActionVect startingActions;
  for (int i = 0; i < 10; ++i) {
    startingActions.push_back(PLAYER_A_FIRE);
  }
  return startingActions;
}
}  // namespace ale
