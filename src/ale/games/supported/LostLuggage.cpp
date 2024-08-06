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

#include "ale/games/supported/LostLuggage.hpp"

#include "ale/games/RomUtils.hpp"

namespace ale {
using namespace stella;

LostLuggageSettings::LostLuggageSettings() { reset(); }

/* create a new instance of the rom */
RomSettings* LostLuggageSettings::clone() const {
  return new LostLuggageSettings(*this);
}

/* process the latest information from ALE */
void LostLuggageSettings::step(const System& system) {
  // update the reward
  int score = getDecimalScore(0x96, 0x95, 0x94, &system);
  int reward = score - m_score;
  m_reward = reward;
  m_score = score;

  // update terminal status
  m_lives = readRam(&system, 0xCA);
  m_terminal = (m_lives == 0) && readRam(&system, 0xC8) == 0x0A &&
               readRam(&system, 0xA5) == 0x00 && readRam(&system, 0xA9) == 0x00;
}

/* is end of game */
bool LostLuggageSettings::isTerminal() const { return m_terminal; };

/* get the most recently observed reward */
reward_t LostLuggageSettings::getReward() const { return m_reward; }

/* is an action part of the minimal set? */
bool LostLuggageSettings::isMinimal(const Action& a) const {
  switch (a) {
    case PLAYER_A_NOOP:
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

bool LostLuggageSettings::isLegal(const Action& a) const {
  switch (a) {
    // Don't allow pressing 'fire'
    case PLAYER_A_FIRE:
    case PLAYER_A_UPFIRE:
    case PLAYER_A_DOWNFIRE:
    case PLAYER_A_LEFTFIRE:
    case PLAYER_A_RIGHTFIRE:
    case PLAYER_A_UPLEFTFIRE:
    case PLAYER_A_UPRIGHTFIRE:
    case PLAYER_A_DOWNLEFTFIRE:
    case PLAYER_A_DOWNRIGHTFIRE:
      return false;
    default:
      return true;
  }
}

/* reset the state of the game */
void LostLuggageSettings::reset() {
  m_reward = 0;
  m_score = 0;
  m_terminal = false;
  m_lives = 3;
}

/* saves the state of the rom settings */
void LostLuggageSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putInt(m_lives);
}

// loads the state of the rom settings
void LostLuggageSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  m_lives = ser.getInt();
}

ActionVect LostLuggageSettings::getStartingActions() {
  return {PLAYER_A_FIRE};
}

// According to https://atariage.com/manual_html_page.php?SoftwareLabelID=277
// the left difficulty switch sets whether there are one or two catchers.
DifficultyVect LostLuggageSettings::getAvailableDifficulties() {
  return {0, 1};
}

// According to https://atariage.com/manual_html_page.php?SoftwareLabelID=277
// there are 8 variations of the game but only modes 1 and 4 are for a single
// player. The additional game mode adds the presence of a critical suitcase
// that must be caught or the game is immediately over.
ModeVect LostLuggageSettings::getAvailableModes() {
  return {0, 1};
}

void LostLuggageSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {
  if (isModeSupported(m)) {
    const int desired_mode = 1 + m * 3;
    // Press select until the correct mode is reached.
    while (readRam(&system, 0x94) != desired_mode) {
      environment->pressSelect(2);
    }

    // Reset the environment to apply changes.
    environment->softReset();
  } else {
    throw std::runtime_error("This game mode is not supported.");
  }
}

}  // namespace ale
