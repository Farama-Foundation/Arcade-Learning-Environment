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

#ifndef __ROMSETTINGS4P_HPP__
#define __ROMSETTINGS4P_HPP__

#include "RomSettings2P.hpp"


namespace ale {

// rom support interface
class RomSettings4P : public RomSettings2P {
 public:
  RomSettings4P();

  virtual ~RomSettings4P() {}

  virtual reward_t getRewardP3() const = 0;
  virtual reward_t getRewardP4() const = 0;
  virtual int livesP3();
  virtual int livesP4();
  virtual ModeVect get4PlayerModes() = 0;
  virtual void setMode(game_mode_t m, System&, std::unique_ptr<StellaEnvironmentWrapper>) = 0;

};

}  // namespace ale

#endif  // __ROMSETTINGS4P_HPP__
