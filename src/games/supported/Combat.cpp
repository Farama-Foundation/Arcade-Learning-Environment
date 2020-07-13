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

#include "Combat.hpp"

#include <algorithm>

#include "../RomUtils.hpp"

namespace ale {

CombatSettings::CombatSettings() { reset(); }

/* create a new instance of the rom */
RomSettings* CombatSettings::clone() const {
  return new CombatSettings(*this);
}

/* process the latest information from ALE */
void CombatSettings::step(const System& system) {
  // update the reward
  int my_score = std::max(getDecimalScore(0xa1, &system), 0);
  int oppt_score = std::max(getDecimalScore(0xa2, &system), 0);
  int score = my_score - oppt_score;
  m_reward = score - m_score;
  m_score = score;

  int over_flag = readRam(&system, 0x88);

  m_terminal = my_score == 99 || oppt_score == 99 || over_flag == 0;
}

/* is end of game */
bool CombatSettings::isTerminal() const { return m_terminal; };

/* get the most recently observed reward */
reward_t CombatSettings::getReward() const { return m_reward; }
reward_t CombatSettings::getRewardP2() const { return -m_reward; }

/* is an action part of the minimal set? */
bool CombatSettings::isMinimal(const Action& a) const {
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
void CombatSettings::reset() {
  m_reward = 0;
  m_score = 0;
  m_terminal = false;
}

/* saves the state of the rom settings */
void CombatSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void CombatSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

DifficultyVect CombatSettings::getAvailableDifficulties() {
  return {0, 1, 2, 3};
}
ModeVect CombatSettings::getAvailableModes() {
  // this isn't actually single player.
  //don't use this game in single player mode.
  return {};
}

ModeVect CombatSettings::get2PlayerModes() {
  return {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27};
}
void CombatSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {

  game_mode_t byte_value = m - 1;

  while (readRam(&system, 0x80) != byte_value) { environment->pressSelect(1); }
  // reset the environment to apply changes.
  environment->softReset();

}

}  // namespace ale
