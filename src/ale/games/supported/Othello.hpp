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

#ifndef __OTHELLO_HPP__
#define __OTHELLO_HPP__

#include "ale/games/RomSettings.hpp"

namespace ale {

class OthelloSettings : public RomSettings {
 public:
  OthelloSettings();

  virtual void reset();

  virtual bool isTerminal() const;

  virtual reward_t getReward() const;

  virtual const char* rom() const { return "othello"; }

  // The md5 checksum of the ROM that this game supports
  const char* md5() const override { return "113cd09c9771ac278544b7e90efe7df2"; }

  virtual RomSettings* clone() const;

  virtual bool isMinimal(const Action& a) const;

  virtual void step(const stella::System& system);

  virtual void saveState(stella::Serializer& ser);

  virtual void loadState(stella::Deserializer& ser);

  virtual ModeVect getAvailableModes();

  virtual void setMode(game_mode_t m, stella::System& system,
                       std::unique_ptr<StellaEnvironmentWrapper> environment);

  virtual DifficultyVect getAvailableDifficulties();

 private:
  bool m_terminal;
  reward_t m_reward;
  int m_score;
  int m_no_input;
};

}  // namespace ale

#endif  // __OTHELLO_HPP__
