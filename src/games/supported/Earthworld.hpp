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

#ifndef __EARTHWORLD_HPP__
#define __EARTHWORLD_HPP__

#include "games/RomSettings.hpp"

namespace ale {

class EarthworldSettings : public RomSettings {
 public:
  EarthworldSettings();

  void reset() override;

  bool isTerminal() const override;

  reward_t getReward() const override;

  const char* rom() const override { return "earthworld"; }

  // The md5 checksum of the ROM that this game supports
  const char* md5() const override { return "5aea9974b975a6a844e6df10d2b861c4"; }

  RomSettings* clone() const override;

  bool isMinimal(const Action& a) const override;

  void step(const System& system) override;

  void saveState(Serializer& ser) override;

  void loadState(Deserializer& ser) override;

 private:
  bool m_terminal;
  reward_t m_reward;
  int m_score;
};

}  // namespace ale

#endif  // __EARTHWORLD_HPP__
