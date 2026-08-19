/* *****************************************************************************
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

#ifndef __MAZECRAZE_HPP__
#define __MAZECRAZE_HPP__

#include "ale/games/RomSettings2P.hpp"

namespace ale {

/* RL wrapper for Maze Craze settings. Maze Craze is a two-player only game. */
class MazeCrazeSettings : public RomSettings2P {
 public:
  MazeCrazeSettings();

  // reset
  void reset() override;

  // is end of game
  bool isTerminal() const override;

  // get the most recently observed reward
  reward_t getReward() const override;
  reward_t getRewardP2() const override;

  int lives() override;
  int livesP2() override;

  // the rom-name
  const char* rom() const override { return "maze_craze"; }

  // The md5 checksum of the ROM that this game supports
  const char* md5() const override { return "ed2218b3075d15eaa34e3356025ccca3"; }

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

  // returns a list of mode that the game can be played in
  ModeVect getAvailableModes() override;
  ModeVect get2PlayerModes() override;

  // set the mode of the game
  void setMode(game_mode_t m, stella::System& system,
               std::unique_ptr<StellaEnvironmentWrapper> environment) override;

  // returns a list of difficulties that the game can be played in
  DifficultyVect getAvailableDifficulties() override;

 private:
  bool m_terminal;
  bool p1_isalive;
  bool p2_isalive;
  reward_t m_reward_p1;
  reward_t m_reward_p2;
  int m_score;
};

}  // namespace ale

#endif  // __MAZECRAZE_HPP__
