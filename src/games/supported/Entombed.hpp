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

#ifndef __ENTOMBED_HPP__
#define __ENTOMBED_HPP__

#include "../RomSettings.hpp"
#include "../RomSettings2P.hpp"

namespace ale {

class EntombedSettings : public RomSettings2P {
 public:
  EntombedSettings();

  void reset() override;

  bool isTerminal() const override;

  reward_t getReward() const override;
  reward_t getRewardP2() const override;

  const char* rom() const override { return "entombed"; }

  // The md5 checksum of the ROM that this game supports
  const char* md5() const override { return "6b683be69f92958abe0e2a9945157ad5"; }

  RomSettings* clone() const override;

  int lives() override;
  int livesP2() override;

  bool isMinimal(const Action& a) const override;

  void step(const System& system) override;

  void saveState(Serializer& ser) override;

  void loadState(Deserializer& ser) override;

  DifficultyVect getAvailableDifficulties() override;

  ActionVect getStartingActions() override;

  ModeVect getAvailableModes() override;
  ModeVect get2PlayerModes() override;

  void setMode(game_mode_t, System& system,
               std::unique_ptr<StellaEnvironmentWrapper> environment) override;


 private:
  int cur_depth;
  bool m_terminal;
  reward_t m_reward;
  int m_score;
  int lives_p1;
  int lives_p2;
  bool is_two_player;
  bool is_cooperative;
};

}  // namespace ale

#endif  // __ENTOMBED_HPP__
