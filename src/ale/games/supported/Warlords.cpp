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

#include "ale/games/supported/Warlords.hpp"

#include "ale/games/RomUtils.hpp"

namespace ale {
using namespace stella;

WarlordsSettings::WarlordsSettings() { reset(); }

/* create a new instance of the rom */
RomSettings* WarlordsSettings::clone() const {
  return new WarlordsSettings(*this);
}

/* process the latest information from ALE */
void WarlordsSettings::step(const System& system) {
  // Each player's bit in RAM 0xEE is set when their wall's warlord is hit,
  // eliminating them from the game. A player alive has 0 lives; a player who
  // has been eliminated has -1 lives.
  int lives_byte = readRam(&system, 0xee);
  int new_lives[4] = {
    (0x80 & lives_byte) ? -1 : 0,
    (0x40 & lives_byte) ? -1 : 0,
    (0x20 & lives_byte) ? -1 : 0,
    (0x10 & lives_byte) ? -1 : 0
  };

  int num_alive = 4;
  for (int i = 0; i < 4; i++) {
    num_alive += new_lives[i];
    // reward -1 on the step a player is eliminated
    m_rewards[i] = new_lives[i] < m_lives[i] ? -1 : 0;
    m_lives[i] = new_lives[i];
  }
  m_terminal = num_alive <= 1;
  // the last player standing gets a +1 reward
  if (m_terminal) {
    for (int i = 0; i < 4; i++) {
      if (m_lives[i] == 0) {
        m_rewards[i] = 1;
      }
    }
  }
}

/* is end of game */
bool WarlordsSettings::isTerminal() const { return m_terminal; };

/* get the most recently observed reward */
reward_t WarlordsSettings::getReward() const { return m_rewards[0]; }
reward_t WarlordsSettings::getRewardP2() const { return m_rewards[1]; }
reward_t WarlordsSettings::getRewardP3() const { return m_rewards[2]; }
reward_t WarlordsSettings::getRewardP4() const { return m_rewards[3]; }

int WarlordsSettings::lives() { return m_lives[0]; }
int WarlordsSettings::livesP2() { return m_lives[1]; }
int WarlordsSettings::livesP3() { return m_lives[2]; }
int WarlordsSettings::livesP4() { return m_lives[3]; }

/* is an action part of the minimal set? */
bool WarlordsSettings::isMinimal(const Action& a) const {
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
void WarlordsSettings::reset() {
  for (size_t i = 0; i < 4; i++) {
    m_rewards[i] = 0;
    m_scores[i] = 0;
    m_lives[i] = 0;
  }
  m_terminal = false;
}

/* saves the state of the rom settings */
void WarlordsSettings::saveState(Serializer& ser) {
  for (size_t i = 0; i < 4; i++) {
    ser.putInt(m_rewards[i]);
    ser.putInt(m_scores[i]);
    ser.putInt(m_lives[i]);
  }
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void WarlordsSettings::loadState(Deserializer& ser) {
  for (size_t i = 0; i < 4; i++) {
    m_rewards[i] = ser.getInt();
    m_scores[i] = ser.getInt();
    m_lives[i] = ser.getInt();
  }
  m_terminal = ser.getBool();
}

DifficultyVect WarlordsSettings::getAvailableDifficulties() {
  return {0};
}

// The 23 game variants mix the number of human players with whether the game
// is played in "doubles" and at what speed; RAM 0xD9 holds the variant minus
// one. Variants not listed here need three human players, which the Atari
// 2600 paddle layout cannot express in ALE.
ModeVect WarlordsSettings::getAvailableModes() {
  return {4, 9, 14, 19};
}

ModeVect WarlordsSettings::get2PlayerModes() {
  return {3, 5, 8, 10, 13, 15, 18, 20, 23};
}

ModeVect WarlordsSettings::get4PlayerModes() {
  return {1, 6, 11, 16, 21};
}

void WarlordsSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {
  game_mode_t byte_value = m - 1;

  // Press select until the correct mode is reached.
  while (readRam(&system, 0xd9) != byte_value) {
    environment->pressSelect(1);
  }
  // reset the environment to apply changes.
  environment->softReset();
}

}  // namespace ale
