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

#include "Warlords.hpp"

#include <algorithm>

#include "../RomUtils.hpp"

namespace ale {

WarlordsSettings::WarlordsSettings() { reset(); }

/* create a new instance of the rom */
RomSettings* WarlordsSettings::clone() const {
  return new WarlordsSettings(*this);
}

/* process the latest information from ALE */
void WarlordsSettings::step(const System& system) {
  int lives_byte = readRam(&system, 0xee);
  int new_lives[4] = {
    (0x80 & lives_byte) ? -1 : 0,
    (0x40 & lives_byte) ? -1 : 0,
    (0x20 & lives_byte) ? -1 : 0,
    (0x10 & lives_byte) ? -1 : 0
  };
  // update
  int num_alive = 4;
  for(int i = 0; i < 4; i++){
    num_alive += new_lives[i];
    assert(new_lives[i] <= m_lives[i] && "did not register lives correctly, reset before terminate");

    m_rewards[i] = new_lives[i] < m_lives[i] ? -1 : 0;
    m_lives[i] = new_lives[i];
  }
  m_terminal = num_alive <= 1;
  //set reward to +1 for winner
  if(m_terminal){
    for(int i = 0; i < 4; i++){
      if(m_lives[i] == 0){
        m_rewards[i] = 1;
      }
    }
  }
  // update the reward (obsolete since reward now tracks lives directly)
  // for(size_t i = 0; i < 4; i++){
  //   int score_byte = readRam(&system, 0xe0+i);
  //   //yes, this is the correct formula. I don't understand it either, but it works.
  //   int score = (score_byte - 5) / 6;
  //   m_rewards[i] = score - m_scores[i];
  //   m_scores[i] = score;
  //
  //   if(score >= 10){
  //     m_terminal = true;
  //   }
  // }
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
  for(size_t i = 0; i < 4; i++){
    m_rewards[i] = 0;
    m_scores[i] = 0;
    m_lives[i] = 0;
  }
  m_terminal = false;
}

/* saves the state of the rom settings */
void WarlordsSettings::saveState(Serializer& ser) {
  for(size_t i = 0; i < 4; i++){
    ser.putInt(m_rewards[i]);
    ser.putInt(m_scores[i]);
    ser.putInt(m_lives[i]);
  }
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void WarlordsSettings::loadState(Deserializer& ser) {
  for(size_t i = 0; i < 4; i++){
    m_rewards[i] = ser.getInt();
    m_scores[i] = ser.getInt();
    m_lives[i] = ser.getInt();
  }
  m_terminal = ser.getBool();
}

void WarlordsSettings::modifyEnvironmentSettings(Settings& settings) {
  // Note that Warlords uses the paddle controller. Often in paddle
  // games the the range needs to be extended to allow the cursor
  // to move to all areas of the board. But its not clear that this is necessary

  //settings.setInt("paddle_max", 790196); // this is the default paddle value
}
DifficultyVect WarlordsSettings::getAvailableDifficulties() {
  return {0};
}
ModeVect WarlordsSettings::getAvailableModes() {
  return {4,9,14,19};
}
ModeVect WarlordsSettings::get2PlayerModes() {
  return {3,5,8,10,13,15,18,20,23};
}
ModeVect WarlordsSettings::get4PlayerModes() {
  return {1,6,11,16,21};
}
void WarlordsSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {

  game_mode_t byte_value = m - 1;

  while (readRam(&system, 0xd9) != byte_value) { environment->pressSelect(1); }
  // reset the environment to apply changes.
  environment->softReset();

}

}  // namespace ale
