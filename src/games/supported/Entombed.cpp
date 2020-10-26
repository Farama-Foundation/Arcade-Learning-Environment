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

#include "Entombed.hpp"

#include "../RomUtils.hpp"

namespace ale {

EntombedSettings::EntombedSettings() { reset(); }

RomSettings* EntombedSettings::clone() const {
  return new EntombedSettings(*this);
}

void EntombedSettings::step(const System& system) {
  // Lives are stored as the bottom 2 bits of RAM 0xC7:
  int new_lives1 = readRam(&system, 0xc7) & 0x03;
  // Livesp2 are stored as bits in the middle of RAM 0xC7:
  int new_lives2 = (readRam(&system, 0xc7) & 0x30) >> 4;

  int cur_substage = readRam(&system, 0xef);

  bool has_lost_life = lives_p1 > new_lives1 || lives_p2 > new_lives2;
  lives_p1 = new_lives1;
  lives_p2 = new_lives2;

  if(is_two_player){
    if(is_cooperative){
      if(cur_substage > cur_depth){
        //rewards every 5 seconds after the first 10 seconds after starting a stage
        m_reward = 1;
      }
      else{
        m_reward = 0;
      }
      // m_reward -= has_lost_life;
      cur_depth = cur_substage;
    }
    else{
      int score = lives_p1 - lives_p2;
      m_reward = score - m_score;
      m_score = score;
    }
    m_terminal = lives_p1 == 0 || lives_p2 == 0;
  }
  else{
    // Score is stored as hexadecimal in RAM 0xE3:
    int depth_reached = readRam(&system, 0xe3);
    m_reward = depth_reached - m_score;
    m_score = depth_reached;
    m_terminal = lives_p1 == 0;
  }
}

bool EntombedSettings::isTerminal() const { return m_terminal; }

reward_t EntombedSettings::getReward() const { return m_reward; }
reward_t EntombedSettings::getRewardP2() const { return is_cooperative ? m_reward : -m_reward; }

int EntombedSettings::lives() { return lives_p1; }
int EntombedSettings::livesP2() { return lives_p2; }

bool EntombedSettings::isMinimal(const Action& a) const {
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

void EntombedSettings::reset() {
  cur_depth = 0;
  m_reward = 0;
  m_score = 0;
  lives_p1 = 0;
  lives_p2 = 0;
  m_terminal = false;
}

void EntombedSettings::saveState(Serializer& ser) {
  ser.putInt(cur_depth);
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putInt(lives_p1);
  ser.putInt(lives_p2);
  ser.putBool(m_terminal);
  ser.putBool(is_two_player);
  ser.putBool(is_cooperative);
}

void EntombedSettings::loadState(Deserializer& ser) {
  cur_depth = ser.getInt();
  m_reward = ser.getInt();
  m_score = ser.getInt();
  lives_p1 = ser.getInt();
  lives_p2 = ser.getInt();
  m_terminal = ser.getBool();
  is_two_player = ser.getBool();
  is_cooperative = ser.getBool();
}

// According to https://atariage.com/manual_html_page.php?SoftwareLabelID=165
// the left difficulty switch sets the number of 'make-breaks' received when
// collecting the blicking blocks.
DifficultyVect EntombedSettings::getAvailableDifficulties() {
  return {0, 2};
}

// Need to press 'fire' to start, not 'reset', then wait a few frames for the
// game state to be set up.
ActionVect EntombedSettings::getStartingActions() {
  return {PLAYER_A_FIRE, PLAYER_A_NOOP, PLAYER_A_NOOP, PLAYER_A_NOOP,
          PLAYER_A_NOOP, PLAYER_A_NOOP};
}
ModeVect EntombedSettings::getAvailableModes() {
  return {1};
}

ModeVect EntombedSettings::get2PlayerModes() {
  //2 is competitive reward: you are trying to make the other player die before you
  //3 is cooperative reward. You want to maximize the depth you both reach.
  return {2,3};
}

void EntombedSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {

  is_two_player = (m > 1);
  is_cooperative = (m == 3);

  game_mode_t byte_value = (m == 1 ? 1 : 0);

  while (readRam(&system, 0xf4) != byte_value) { environment->pressSelect(1); }
  // reset the environment to apply changes.
  environment->softReset();

}

}  // namespace ale
