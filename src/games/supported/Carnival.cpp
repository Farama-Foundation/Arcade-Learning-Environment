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

#include "games/supported/Carnival.hpp"

#include "games/RomUtils.hpp"

namespace ale {

CarnivalSettings::CarnivalSettings() { reset(); }

/* create a new instance of the rom */
RomSettings* CarnivalSettings::clone() const {
  return new CarnivalSettings(*this);
}

/* process the latest information from ALE */
void CarnivalSettings::step(const System& system) {
  // update the reward
  reward_t score = getDecimalScore(0xAE, 0xAD, &system);
  score *= 10;
  m_reward = score - m_score;
  m_score = score;

  // update terminal status
  int ammo = readRam(&system, 0x83);
  m_terminal = ammo < 1;
}

/* is end of game */
bool CarnivalSettings::isTerminal() const { return m_terminal; };

/* get the most recently observed reward */
reward_t CarnivalSettings::getReward() const { return m_reward; }

/* is an action part of the minimal set? */
bool CarnivalSettings::isMinimal(const Action& a) const {
  switch (a) {
    case PLAYER_A_NOOP:
    case PLAYER_A_FIRE:
    case PLAYER_A_RIGHT:
    case PLAYER_A_LEFT:
    case PLAYER_A_RIGHTFIRE:
    case PLAYER_A_LEFTFIRE:
      return true;
    default:
      return false;
  }
}

/* reset the state of the game */
void CarnivalSettings::reset() {
  m_reward = 0;
  m_score = 0;
  m_terminal = false;
}

/* saves the state of the rom settings */
void CarnivalSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void CarnivalSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

}  // namespace ale
