/* *****************************************************************************
 * The line 78 is based on Xitari's code, from Google Inc.
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
 *
 * RomSettings.hpp
 *
 * The interface to describe games as RL environments. It provides terminal and
 *  reward information.
 * *****************************************************************************
 */

#ifndef __ROMSETTINGS2P_HPP__
#define __ROMSETTINGS2P_HPP__

#include "RomSettings.hpp"


namespace ale {

// rom support interface
class RomSettings2P : public RomSettings {
 public:
  RomSettings2P();

  virtual ~RomSettings2P() {}

  // get the most recently observed reward
  // default is the negative of getReward()
  virtual reward_t getRewardP2() const { return 0; };

  // is an action part of the minimal set?
  virtual bool isMinimalP2(const Action& a) const;

  // is an action legal (default: yes)
  virtual bool isLegalP2(const Action& a) const;

  // does the environment support two players (always True)
  virtual bool supportsTwoPlayers() const;

  // Remaining lives.
  virtual int livesP2() {
    return isTerminal() ? 0 : 1;
  }

  // Returns a restricted (minimal) set of actions.
  // If not overriden, this is the player 2 version of all player 1 actions.
  virtual ActionVect getMinimalActionSetP2();

  // Returns the set of all legal actions for player 2
  // by default, this just returns getAllActions()
  ActionVect getAllActionsP2();

  // Returns a list of mode that the game can be played with two players.
  // By default, there is only one available mode with two players.
  // note that this list will be disjoint from getAvailableModes
  virtual ModeVect get2PlayerModes() = 0;
  virtual void setMode(game_mode_t m, System&, std::unique_ptr<StellaEnvironmentWrapper>) = 0;

protected:
  //helper method to get 2 player modes
  ModeVect oppositeModes(int num_modes) ;
};

}  // namespace ale

#endif  // __ROMSETTINGS2P_HPP__
