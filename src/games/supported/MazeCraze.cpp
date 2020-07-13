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

#include "MazeCraze.hpp"

#include "../RomUtils.hpp"

namespace ale {

MazeCrazeSettings::MazeCrazeSettings() { reset(); }

RomSettings* MazeCrazeSettings::clone() const {
  return new MazeCrazeSettings(*this);
}

void MazeCrazeSettings::step(const System& system) {
  int player1_score = readRam(&system,0xEC) == 0xff ? 1 : 0;
  int player2_score = readRam(&system,0xED) == 0xff ? 1 : 0;

  m_reward_p1 = 0;
  m_reward_p2 = 0;
  //a player is killed if the bit saying it cannot move
  //is set for a couple seconds.
  if(p1_isalive && readRam(&system,0xEA)&0x40) {
    p1_isalive = false;
    m_reward_p1 = -1;
  }
  if(p2_isalive && readRam(&system,0xEB)&0x40) {
    p2_isalive = false;
    m_reward_p2 = -1;
  }
  int completion_score = player1_score - player2_score;
  if(completion_score != 0){
    if(!p1_isalive){
      m_reward_p2 = 1;
    }
    else if(!p2_isalive){
      m_reward_p1 = 1;
    }
    else{
      m_reward_p1 = completion_score;
      m_reward_p2 = -completion_score;
    }
  }

  // game is over when some player wins/ i.e. reward is not zero
  // or both players are dead
  m_terminal = completion_score != 0 || (!p1_isalive && !p2_isalive);
}

bool MazeCrazeSettings::isTerminal() const { return m_terminal; }

reward_t MazeCrazeSettings::getReward() const { return m_reward_p1; }
reward_t MazeCrazeSettings::getRewardP2() const { return m_reward_p2; }

int MazeCrazeSettings::lives() { return p1_isalive ? 0 : -1; }
int MazeCrazeSettings::livesP2() { return p2_isalive ? 0 : -1; }

bool MazeCrazeSettings::isMinimal(const Action& a) const {
  switch (a) {
    // the joystick ususlaly doesn't need to fire. Only used for the
    // "player peek" functionality
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

void MazeCrazeSettings::reset() {
  m_reward_p1 = 0;
  m_reward_p2 = 0;
  m_score = 0;
  p1_isalive = true;
  p2_isalive = true;
  m_terminal = false;
}

void MazeCrazeSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward_p1);
  ser.putInt(m_reward_p2);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putBool(p1_isalive);
  ser.putBool(p2_isalive);
}

void MazeCrazeSettings::loadState(Deserializer& ser) {
  m_reward_p1 = ser.getInt();
  m_reward_p2 = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  p1_isalive = ser.getBool();
  p2_isalive = ser.getBool();
}

DifficultyVect MazeCrazeSettings::getAvailableDifficulties() {
  // According to https://atariage.com/manual_html_page.php?SoftwareLabelID=931
  // The left difficulty controls player a speed,
  // right difficulty switch controls player b speed
  return {0};
}
ModeVect MazeCrazeSettings::getAvailableModes() {
  return {};
}
ModeVect MazeCrazeSettings::get2PlayerModes() {
  //all modes two player
  ModeVect modes;
  for(int j = 0; j < 16; j++){
    for(int i = 0; i < 4; i++){
      modes.push_back(j*4+i);
    }
  }
  return modes;
}

void MazeCrazeSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {
  // Modes are combined with speed setting. We are ignoring speed, so

  // Press select until the correct mode is reached.
  while (readRam(&system, 0xbd) != m) {
    environment->pressSelect(2);
  }

  // Reset the environment to apply changes.
  environment->softReset();
}


}  // namespace ale
