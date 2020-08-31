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

#ifndef __ROMSETTINGS_HPP__
#define __ROMSETTINGS_HPP__

#include <algorithm>
#include <memory>
#include <stdexcept>

#include "../common/Constants.h"
#include "../emucore/Serializer.hxx"
#include "../emucore/Deserializer.hxx"
#include "../emucore/Settings.hxx"
#include "../environment/stella_environment_wrapper.hpp"

class System;

namespace ale {

using RewardVec = std::vector<reward_t>;
using ModePlayerVect = std::vector<ModeVect>;

// rom support interface
class RomSettings {
 public:
  RomSettings();

  virtual ~RomSettings() {}

  // reset
  virtual void reset() {}

  // This method is called with the collection of settings that will be used to
  // create the Stella emulator for the environment. Overriders may modify these
  // settings when this is needed for the correct behavior of the ROM in Stella.
  virtual void modifyEnvironmentSettings(Settings& settings) {}

  // is end of game
  virtual bool isTerminal() const = 0;

  // get the most recently observed reward
  virtual reward_t getReward() const = 0;

  // the rom-name
  virtual const char* rom() const = 0;

  virtual const char * md5() const = 0;

  // create a new instance of the rom
  virtual RomSettings* clone() const = 0;

  // is an action part of the minimal set?
  virtual bool isMinimal(const Action& a) const = 0;

  // process the latest information from ALE
  virtual void step(const System& system) = 0;

  // saves the state of the rom settings
  virtual void saveState(Serializer& ser) = 0;

  // loads the state of the rom settings
  virtual void loadState(Deserializer& ser) = 0;

  // is an action legal (default: yes)
  virtual bool isLegal(const Action& a) const;

  // Remaining lives.
  virtual int lives();

  // Returns a restricted (minimal) set of actions. If not overriden, this is all actions.
  virtual ActionVect getMinimalActionSet();

  // Returns the set of all legal actions
  ActionVect getAllActions();

  // Returns a list of actions that are required to start the game.
  // By default this is an empty list.
  virtual ActionVect getStartingActions();

  // Returns a list of mode that the game can be played in.
  // By default, there is only one available mode.
  virtual ModeVect getAvailableModes();

  // Set the mode of the game. The given mode must be
  // one returned by the previous function.
  virtual void setMode(
      game_mode_t, System& system,
      std::unique_ptr<StellaEnvironmentWrapper> environment);

  // Returns a list of difficulties that the game can be played in.
  // By default, there is only one available difficulty.
  virtual DifficultyVect getAvailableDifficulties();

  //two player methods. all fail when on a single player game
  virtual reward_t getRewardP2() const;
  virtual int livesP2();
  virtual ModeVect get2PlayerModes();

  //four player methods. all fail when on a
  virtual reward_t getRewardP3() const;
  virtual reward_t getRewardP4() const;
  virtual int livesP3();
  virtual int livesP4();
  virtual ModeVect get3PlayerModes();
  virtual ModeVect get4PlayerModes();

  //multiplayer methods
 protected:
  // Helper function that checks if our settings support this given mode.
  bool isModeSupported(game_mode_t m, int players=1);
};

}  // namespace ale

#endif  // __ROMSETTINGS_HPP__
