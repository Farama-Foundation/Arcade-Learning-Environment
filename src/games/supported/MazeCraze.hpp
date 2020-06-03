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

#include "../RomSettings.hpp"
#include "../RomSettings2P.hpp"

namespace ale {

class MazeCrazeSettings : public RomSettings2P {
 public:
  MazeCrazeSettings();

  void reset();

  bool isTerminal() const;

  reward_t getReward() const override;
  reward_t getRewardP2() const override;

  int lives() override;
  int livesP2() override;

  const char* rom() const { return "maze_craze"; }

  // The md5 checksum of the ROM that this game supports
  const char* md5() const override { return "ed2218b3075d15eaa34e3356025ccca3"; }

  RomSettings* clone() const;

  bool isMinimal(const Action& a) const;

  void step(const System& system);

  void saveState(Serializer& ser);

  void loadState(Deserializer& ser);

  ModeVect getAvailableModes();
  ModeVect get2PlayerModes();

  void setMode(game_mode_t m, System& system,
                       std::unique_ptr<StellaEnvironmentWrapper> environment);

  DifficultyVect getAvailableDifficulties();

 private:
  bool m_terminal;
  bool p1_isalive;
  bool p2_isalive;
  int steps_p1_deactive;
  int steps_p2_deactive;
  reward_t m_reward_p1;
  reward_t m_reward_p2;
  int m_score;
};

}  // namespace ale

#endif  // __MAZECRAZE_HPP__
