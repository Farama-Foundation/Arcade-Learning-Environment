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

#ifndef __LOSTLUGGAGE_HPP__
#define __LOSTLUGGAGE_HPP__

#include "ale/games/RomSettings.hpp"

namespace ale {

/* RL wrapper for Lost Luggage */
class LostLuggageSettings : public RomSettings {
 public:
  LostLuggageSettings();

  // reset
  void reset() override;

  // is end of game
  bool isTerminal() const override;

  // get the most recently observed reward
  reward_t getReward() const override;

  // the rom-name
  // MD5 7c00e7a205d3fda98eb20da7c9c50a55
  const char* rom() const override { return "lost_luggage"; }

  // The md5 checksum of the ROM that this game supports
  const char* md5() const override { return "2d76c5d1aad506442b9e9fb67765e051"; }

  // create a new instance of the rom
  RomSettings* clone() const override;

  // is an action part of the minimal set?
  bool isMinimal(const Action& a) const override;

  // FIRE resets the game, prevent this
  bool isLegal(const Action& a) const override;

  // process the latest information from ALE
  void step(const stella::System& system) override;

  // saves the state of the rom settings
  void saveState(stella::Serializer& ser) override;

  // loads the state of the rom settings
  void loadState(stella::Deserializer& ser) override;

  // LostLuggage requires the fire action to start the game
  ActionVect getStartingActions() override;

  int lives() override { return isTerminal() ? 0 : m_lives; }

  ModeVect getAvailableModes() override;

  void setMode(game_mode_t m, stella::System& system,
               std::unique_ptr<StellaEnvironmentWrapper> environment) override;

  DifficultyVect getAvailableDifficulties() override;

 private:
  bool m_terminal;
  reward_t m_reward;
  reward_t m_score;
  int m_lives;
};

}  // namespace ale

#endif  // __LOSTLUGGAGE_HPP__
