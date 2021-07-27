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
 * RomSettings4P.hpp
 *
 * The interface to describe games as RL environments. It provides terminal and
 *  reward information.
 * *****************************************************************************
 */

#ifndef __ROMSETTINGS4P_HPP__
#define __ROMSETTINGS4P_HPP__

#include "RomSettings2P.hpp"


namespace ale {

// rom support interface
class RomSettings4P : public RomSettings2P {
 public:
  RomSettings4P() {}

  virtual ~RomSettings4P() {}

  // gets reward for players 3 and 4
  virtual reward_t getRewardP3() const = 0;
  virtual reward_t getRewardP4() const = 0;
  // gets lives left for players 3 and 4
  virtual int livesP3() { return isTerminal() ? 0 : 1; }
  virtual int livesP4() { return isTerminal() ? 0 : 1; }
  // gets list of avaliable modes for player 4
  virtual ModeVect get4PlayerModes() = 0;
  // enforces that 4 player games must override the setMode function
  virtual void setMode(game_mode_t m, System&, std::unique_ptr<StellaEnvironmentWrapper>) = 0;
};

}  // namespace ale

#endif  // __ROMSETTINGS4P_HPP__
