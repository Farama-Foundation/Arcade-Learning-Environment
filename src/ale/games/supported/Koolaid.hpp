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

#ifndef __KOOLAID_HPP__
#define __KOOLAID_HPP__

#include "ale/games/RomSettings.hpp"

namespace ale {

/* RL wrapper for Koolaid */
class KoolaidSettings : public RomSettings {
 public:
  KoolaidSettings();

  // reset
  void reset() override;

  // is end of game
  bool isTerminal() const override;

  // get the most recently observed reward
  reward_t getReward() const override;

  // the rom-name
  const char* rom() const override { return "koolaid"; }

  // The md5 checksum of the ROM that this game supports
  const char* md5() const override { return "534e23210dd1993c828d944c6ac4d9fb"; }

  // create a new instance of the rom
  RomSettings* clone() const override;

  // is an action part of the minimal set?
  bool isMinimal(const Action& a) const override;

  // process the latest information from ALE
  void step(const stella::System& system) override;

  // saves the state of the rom settings
  void saveState(stella::Serializer& ser) override;

  // loads the state of the rom settings
  void loadState(stella::Deserializer& ser) override;

  int lives() override { return 0; }

 private:
  bool m_terminal;
  reward_t m_reward;
  reward_t m_score;
};

}  // namespace ale

#endif  // __KOOLAID_HPP__
