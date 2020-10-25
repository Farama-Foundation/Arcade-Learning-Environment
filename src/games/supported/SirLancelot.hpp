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

#ifndef __SIRLANCELOT_HPP__
#define __SIRLANCELOT_HPP__

#include "games/RomSettings.hpp"

namespace ale {

/* RL wrapper for Up N Down */
class SirLancelotSettings : public RomSettings {
 public:
  SirLancelotSettings();

  // reset
  void reset() override;

  // is end of game
  bool isTerminal() const override;

  // get the most recently observed reward
  reward_t getReward() const override;

  // the rom-name
  // MD5 7ead257e8b5a44cac538f5f54c7a0023
  const char* rom() const override { return "sir_lancelot"; }

  // The md5 checksum of the ROM that this game supports
  const char* md5() const override { return "dd0cbe5351551a538414fb9e37fc56e8"; }

  // create a new instance of the rom
  RomSettings* clone() const override;

  // is an action part of the minimal set?
  bool isMinimal(const Action& a) const override;

  // process the latest information from ALE
  void step(const System& system) override;

  // saves the state of the rom settings
  void saveState(Serializer& ser) override;

  // loads the state of the rom settings
  void loadState(Deserializer& ser) override;

  // SirLancelot requires the reset+left action to start the game
  ActionVect getStartingActions() override;

  int lives() override { return isTerminal() ? 0 : m_lives; }

 private:
  bool m_terminal;
  reward_t m_reward;
  reward_t m_score;
  int m_lives;
};

}  // namespace ale

#endif  // __SIRLANCELOT_HPP__
