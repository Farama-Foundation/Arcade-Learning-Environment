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

#ifndef __MRDO_HPP__
#define __MRDO_HPP__

#include "games/RomSettings.hpp"

namespace ale {

/* RL wrapper for MrDo */
class MrDoSettings : public RomSettings {
 public:
  MrDoSettings();

  // reset
  void reset() override;

  // is end of game
  bool isTerminal() const override;

  // get the most recently observed reward
  reward_t getReward() const override;

  // the rom-name
  const char* rom() const override { return "mr_do"; }

  // The md5 checksum of the ROM that this game supports
  const char* md5() const override { return "aa7bb54d2c189a31bb1fa20099e42859"; }

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

  // Mr. Do requires the fire action to start the game
  ActionVect getStartingActions() override;

  int lives() override { return isTerminal() ? 0 : m_lives; }

  ModeVect getAvailableModes() override;

  void setMode(game_mode_t m, System& system,
               std::unique_ptr<StellaEnvironmentWrapper> environment) override;

 private:
  bool m_terminal;
  reward_t m_reward;
  reward_t m_score;
  int m_lives;
};

}  // namespace ale

#endif  // __MRDO_HPP__
