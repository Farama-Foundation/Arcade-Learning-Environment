/* *****************************************************************************
 * The method lives() is based on Xitari's code, from Google Inc.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License version 2
 * as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 * *****************************************************************************
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

#ifndef __PONG_HPP__
#define __PONG_HPP__

#include "ale/games/RomSettings4P.hpp"

namespace ale {

/* RL wrapper for Pong */
class PongSettings : public RomSettings4P {
 public:
  PongSettings();

  // reset
  void reset() override;

  // is end of game
  bool isTerminal() const override;

  // get the most recently observed reward
  reward_t getReward() const override;
  reward_t getRewardP2() const override;
  reward_t getRewardP3() const override;
  reward_t getRewardP4() const override;

  // the rom-name
  const char* rom() const override { return "pong"; }

  // The md5 checksum of the ROM that this game supports
  const char* md5() const override { return "60e0ea3cbe0913d39803477945e9e5ec"; }

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

  // returns a list of difficulties that the game can be played in
  // in this game, there are 2 available difficulties
  DifficultyVect getAvailableDifficulties() override;

  // returns a list of mode that the game can be played in
  // in this game, there are 2 available modes
  ModeVect getAvailableModes() override;
  ModeVect get2PlayerModes() override;
  ModeVect get4PlayerModes() override;

  // set the mode of the game
  // the given mode must be one returned by the previous function
  void setMode(game_mode_t, stella::System& system,
               std::unique_ptr<StellaEnvironmentWrapper> environment) override;

 private:
  bool m_terminal;
  reward_t m_reward;
  reward_t m_score;
};

}  // namespace ale

#endif  // __PONG_HPP__
