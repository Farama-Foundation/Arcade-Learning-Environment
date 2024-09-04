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

#ifndef __LASERGATES_HPP__
#define __LASERGATES_HPP__

#include "ale/games/RomSettings.hpp"

namespace ale {

/* RL wrapper for Laser Gates */
class LaserGatesSettings : public RomSettings {
 public:
  LaserGatesSettings();

  // reset
  void reset() override;

  // is end of game
  bool isTerminal() const override;

  // get the most recently observed reward
  reward_t getReward() const override;

  // the rom-name
  // MD5 1fa58679d4a39052bd9db059e8cda4ad
  const char* rom() const override { return "laser_gates"; }

  // The md5 checksum of the ROM that this game supports
  const char* md5() const override { return "8e4cd60d93fcde8065c1a2b972a26377"; }

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

  // LaserGates requires the fire action to start the game
  ActionVect getStartingActions() override;

  int lives() override { return 0; }

 private:
  bool m_terminal;
  reward_t m_reward;
  reward_t m_score;
};

}  // namespace ale

#endif  // __LASERGATES_HPP__
