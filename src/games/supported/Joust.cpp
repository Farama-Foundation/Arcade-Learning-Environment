/* *****************************************************************************
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

#include "Joust.hpp"

#include <algorithm>

#include "../RomUtils.hpp"

namespace ale {

JoustSettings::JoustSettings() { reset(); }

/* create a new instance of the rom */
RomSettings* JoustSettings::clone() const {
  return new JoustSettings(*this);
}

/* process the latest information from ALE */
void JoustSettings::step(const System& system) {
  // update the reward
  int score = std::max(getDecimalScore(0x87, 0x85, 0x83, &system), 0);
  int score_p2 = std::max(getDecimalScore(0x88, 0x86, 0x84, &system), 0);
  m_reward = score - m_score;
  m_score = score;
  m_reward_p2 = score_p2 - m_score_p2;
  m_score_p2 = score_p2;

  lives_p1 = (int)(char)(readRam(&system,0x81));
  lives_p2 = (int)(char)(readRam(&system,0x82));

  if(is_two_player){
    m_terminal = score >= 990000 || score_p2 >= 990000
                || (lives_p1 == -1 && lives_p2 == -1);
  }
  else {
    m_terminal = score >= 990000 || lives_p1 == -1;
  }
}

/* is end of game */
bool JoustSettings::isTerminal() const { return m_terminal; };

/* get the most recently observed reward */
reward_t JoustSettings::getReward() const { return m_reward; }
reward_t JoustSettings::getRewardP2() const { return m_reward_p2; }

int JoustSettings::lives() { return lives_p1; }
int JoustSettings::livesP2() { return lives_p2; }

/* is an action part of the minimal set? */
bool JoustSettings::isMinimal(const Action& a) const {
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
void JoustSettings::reset() {
  m_reward = 0;
  m_reward_p2 = 0;
  m_score = 0;
  m_score_p2 = 0;
  lives_p1 = 5;
  lives_p2 = 5;
  m_terminal = false;
  is_two_player = false;
}

/* saves the state of the rom settings */
void JoustSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_reward_p2);
  ser.putInt(m_score);
  ser.putInt(m_score_p2);
  ser.putInt(lives_p1);
  ser.putInt(lives_p2);
  ser.putBool(m_terminal);
  ser.putBool(is_two_player);
}

// loads the state of the rom settings
void JoustSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_reward_p2 = ser.getInt();
  m_score = ser.getInt();
  m_score_p2 = ser.getInt();
  lives_p1 = ser.getInt();
  lives_p2 = ser.getInt();
  m_terminal = ser.getBool();
  is_two_player = ser.getBool();
}

DifficultyVect JoustSettings::getAvailableDifficulties() {
  return {0};
}
ModeVect JoustSettings::getAvailableModes() {
  return {0, 2};
}
ModeVect JoustSettings::get2PlayerModes() {
  return {1, 3};
}

void JoustSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {

  game_mode_t byte_value = m;

  while (readRam(&system, 0xf8)%4 != byte_value) { environment->pressSelect(1); }
  // reset the environment to apply changes.
  environment->softReset();

  is_two_player = (m % 2 == 1);

}

}  // namespace ale
