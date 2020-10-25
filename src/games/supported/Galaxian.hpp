/* *****************************************************************************
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

#ifndef __GALAXIAN_HPP__
#define __GALAXIAN_HPP__

#include "games/RomSettings.hpp"

namespace ale {

// RL wrapper for Galaxian
class GalaxianSettings : public RomSettings {
 public:
  GalaxianSettings();

  // reset
  void reset() override;

  // is end of game
  bool isTerminal() const override;

  // get the most recently observed reward
  reward_t getReward() const override;

  // the rom-name
  const char* rom() const override { return "galaxian"; }

  // The md5 checksum of the ROM that this game supports
  const char* md5() const override { return "211774f4c5739042618be8ff67351177"; }

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

  int lives() override { return isTerminal() ? 0 : m_lives; }

  // get the available number of modes
  unsigned int getNumModes() const { return 9; }

  // returns a list of mode that the game can be played in
  ModeVect getAvailableModes() override;

  // set the mode of the game
  // the given mode must be one returned by the previous function
  void setMode(game_mode_t mode, System& system,
               std::unique_ptr<StellaEnvironmentWrapper> environment) override;

  // Returns a list of difficulties that the game can be played in.
  // 2 difficulties: 0 is left B, 1 is left A
  DifficultyVect getAvailableDifficulties() override { return {0, 1}; }

 private:
  bool m_terminal;
  reward_t m_reward;
  reward_t m_score;
  int m_lives;

  static ActionVect actions;
};

}  // namespace ale

#endif  // __GALAXIAN_HPP__
