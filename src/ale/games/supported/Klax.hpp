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

#ifndef __KLAX_HPP__
#define __KLAX_HPP__

#include "ale/games/RomSettings.hpp"

namespace ale {

class KlaxSettings : public RomSettings {
 public:
  KlaxSettings();

  void reset() override;

  bool isTerminal() const override;

  reward_t getReward() const override;

  const char* rom() const override { return "klax"; }

  // The md5 checksum of the ROM that this game supports
  const char* md5() const override { return "eed9eaf1a0b6a2b9bc4c8032cb43e3fb"; }

  RomSettings* clone() const override;

  bool isMinimal(const Action& a) const override;

  void step(const stella::System& system) override;

  void saveState(stella::Serializer& ser) override;

  void loadState(stella::Deserializer& ser) override;

  ModeVect getAvailableModes() override;

  void setMode(game_mode_t m, stella::System& system,
               std::unique_ptr<StellaEnvironmentWrapper> environment) override;

  ActionVect getStartingActions();

 private:
  bool m_terminal;
  reward_t m_reward;
  int m_score;

  int getKlaxScore(int lower_index, int middle_index, int higher_index,
                   const stella::System* system);
};

}  // namespace ale

#endif  // __KLAX_HPP__
