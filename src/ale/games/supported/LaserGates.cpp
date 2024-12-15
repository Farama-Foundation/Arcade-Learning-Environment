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

#include "ale/games/supported/LaserGates.hpp"

#include "ale/games/RomUtils.hpp"

namespace ale {
using namespace stella;

LaserGatesSettings::LaserGatesSettings() { reset(); }

/* create a new instance of the rom */
RomSettings* LaserGatesSettings::clone() const {
  return new LaserGatesSettings(*this);
}

/* process the latest information from ALE */
void LaserGatesSettings::step(const System& system) {
  // update the reward
  int score = getDecimalScore(0x82, 0x81, 0x80, &system);
  int reward = score - m_score;
  m_reward = reward;
  m_score = score;

  // update terminal status
  m_terminal = readRam(&system, 0x83) == 0x00;
}

/* is end of game */
bool LaserGatesSettings::isTerminal() const { return m_terminal; };

/* get the most recently observed reward */
reward_t LaserGatesSettings::getReward() const { return m_reward; }

/* is an action part of the minimal set? */
bool LaserGatesSettings::isMinimal(const Action& a) const {
  switch (a) {
    case NOOP:
    case FIRE:
    case UP:
    case RIGHT:
    case LEFT:
    case DOWN:
    case UPRIGHT:
    case UPLEFT:
    case DOWNRIGHT:
    case DOWNLEFT:
    case UPFIRE:
    case RIGHTFIRE:
    case LEFTFIRE:
    case DOWNFIRE:
    case UPRIGHTFIRE:
    case UPLEFTFIRE:
    case DOWNRIGHTFIRE:
    case DOWNLEFTFIRE:
      return true;
    default:
      return false;
  }
}

/* reset the state of the game */
void LaserGatesSettings::reset() {
  m_reward = 0;
  m_score = 0;
  m_terminal = false;
}

/* saves the state of the rom settings */
void LaserGatesSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void LaserGatesSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

ActionVect LaserGatesSettings::getStartingActions() {
  return {RESET};
}

}  // namespace ale
