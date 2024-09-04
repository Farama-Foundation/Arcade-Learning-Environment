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

#include "ale/games/supported/MrDo.hpp"

#include "ale/games/RomUtils.hpp"

namespace ale {
using namespace stella;

MrDoSettings::MrDoSettings() { reset(); }

/* create a new instance of the rom */
RomSettings* MrDoSettings::clone() const {
  return new MrDoSettings(*this);
}

/* process the latest information from ALE */
void MrDoSettings::step(const System& system) {
  // update the reward
  int score = getDecimalScore(0x82, 0x83, &system);
  score *= 10;
  int reward = score - m_score;
  m_reward = reward;
  m_score = score;

  // update terminal status
  m_lives = readRam(&system, 0xDB);
  m_terminal = readRam(&system, 0xDA) == 0x40;
}

/* is end of game */
bool MrDoSettings::isTerminal() const { return m_terminal; };

/* get the most recently observed reward */
reward_t MrDoSettings::getReward() const { return m_reward; }

/* is an action part of the minimal set? */
bool MrDoSettings::isMinimal(const Action& a) const {
  switch (a) {
    case PLAYER_A_NOOP:
    case PLAYER_A_FIRE:
    case PLAYER_A_UP:
    case PLAYER_A_RIGHT:
    case PLAYER_A_LEFT:
    case PLAYER_A_DOWN:
    case PLAYER_A_UPFIRE:
    case PLAYER_A_RIGHTFIRE:
    case PLAYER_A_LEFTFIRE:
    case PLAYER_A_DOWNFIRE:
      return true;
    default:
      return false;
  }
}

/* reset the state of the game */
void MrDoSettings::reset() {
  m_reward = 0;
  m_score = 0;
  m_terminal = false;
  m_lives = 4;
}

/* saves the state of the rom settings */
void MrDoSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putInt(m_lives);
}

// loads the state of the rom settings
void MrDoSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_lives = ser.getInt();
}

ActionVect MrDoSettings::getStartingActions() {
  return {PLAYER_A_FIRE};
}

// According to https://atariage.com/manual_html_page.php?SoftwareLabelID=318
// there are four game modes of increasing difficulty.
ModeVect MrDoSettings::getAvailableModes() {
  return {0, 1, 2, 3};
}

void MrDoSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {
  if (m < 4) {
    // Press select until the correct mode is reached.
    while (readRam(&system, 0x80) != m) { environment->pressSelect(5); }

    // Reset the environment to apply changes.
    environment->softReset();
  } else {
    throw std::runtime_error("This game mode is not supported.");
  }
}

}  // namespace ale
