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

#ifndef __TURMOIL_HPP__
#define __TURMOIL_HPP__

#include "ale/games/RomSettings.hpp"

namespace ale {

/* RL wrapper for Turmoil */
class TurmoilSettings : public RomSettings {
 public:
  TurmoilSettings();

  // reset
  void reset() override;

  // is end of game
  bool isTerminal() const override;

  // get the most recently observed reward
  reward_t getReward() const override;

  // the rom-name
  // MD5 sum of ROM file:
  // 7a5463545dfb2dcfdafa6074b2f2c15e  Turmoil.bin
  const char* rom() const override { return "turmoil"; }

  // The md5 checksum of the ROM that this game supports
  const char* md5() const override { return "7a5463545dfb2dcfdafa6074b2f2c15e"; }

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

  // Turmoil requires the fire action to start the game
  ActionVect getStartingActions() override;

  int lives() override { return isTerminal() ? 0 : m_lives; }

  ModeVect getAvailableModes() override;

  void setMode(game_mode_t m, stella::System& system,
               std::unique_ptr<StellaEnvironmentWrapper> environment) override;

 private:
  bool m_terminal;
  reward_t m_reward;
  reward_t m_score;
  int m_lives;
};

}  // namespace ale

#endif  // __TURMOIL_HPP__
