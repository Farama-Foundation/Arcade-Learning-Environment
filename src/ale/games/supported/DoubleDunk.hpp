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

#ifndef __DOUBLEDUNK_HPP__
#define __DOUBLEDUNK_HPP__

#include "ale/games/RomSettings.hpp"

namespace ale {

/* RL wrapper for Double Dunk settings */
class DoubleDunkSettings : public RomSettings {
 public:
  DoubleDunkSettings();

  // reset
  void reset() override;

  // is end of game
  bool isTerminal() const override;

  // get the most recently observed reward
  reward_t getReward() const override;

  // the rom-name
  const char* rom() const override { return "double_dunk"; }

  // The md5 checksum of the ROM that this game supports
  const char* md5() const override { return "368d88a6c071caba60b4f778615aae94"; }

  // get the available number of modes
  unsigned int getNumModes() const { return 16; }

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

  ActionVect getStartingActions() override;

  int lives() override { return 0; }

  // returns a list of mode that the game can be played in
  // in this game, there are 16 available modes
  ModeVect getAvailableModes() override;

  // set the mode of the game
  // the given mode must be one returned by the previous function
  void setMode(game_mode_t, stella::System& system,
               std::unique_ptr<StellaEnvironmentWrapper> environment) override;

 private:
  bool m_terminal;
  reward_t m_reward;
  reward_t m_score;

  // this game has a menu that allows to define various yes/no options
  // this function goes to the next option in the menu
  void goDown(stella::System& system,
              std::unique_ptr<StellaEnvironmentWrapper>& environment);

  // once we are at the proper option in the menu,
  // if we want to enable it all we have to do is to go right
  void activateOption(stella::System& system, unsigned int bitOfInterest,
                      std::unique_ptr<StellaEnvironmentWrapper>& environment);

  // once we are at the proper optio in the menu,
  // if we want to disable it all we have to do is to go left
  void deactivateOption(stella::System& system, unsigned int bitOfInterest,
                        std::unique_ptr<StellaEnvironmentWrapper>& environment);
};

}  // namespace ale

#endif  // __DOUBLEDUNK_HPP__
