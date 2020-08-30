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

#ifndef __VIDEO_CHECKERS_HPP__
#define __VIDEO_CHECKERS_HPP__

#include "../RomSettings.hpp"
#include "../RomSettings2P.hpp"

namespace ale {

class VideoCheckersSettings : public RomSettings2P {
 public:
  VideoCheckersSettings();

  void reset() override;

  bool isTerminal() const override;

  reward_t getReward() const override;
  reward_t getRewardP2() const override;

  const char* rom() const  override { return "video_checkers"; }

  // The md5 checksum of the ROM that this game supports
  const char* md5() const override { return "539d26b6e9df0da8e7465f0f5ad863b7"; }

  RomSettings* clone() const override;

  bool isMinimal(const Action& a) const override;

  virtual void modifyEnvironmentSettings(Settings& settings);

  void step(const System& system) override;

  void saveState(Serializer& ser) override;

  void loadState(Deserializer& ser) override;

  ModeVect getAvailableModes() override;
  ModeVect get2PlayerModes() override;

  void setMode(game_mode_t, System& system,
               std::unique_ptr<StellaEnvironmentWrapper> environment) override;

 private:
  bool m_terminal;
  bool m_is_white_turn;
  int turn_same_count;
  int max_turn_time;
  bool two_player_mode;
  reward_t m_reward_p1;
  reward_t m_reward_p2;
  bool m_reverse_checkers;
};

}  // namespace ale

#endif  // __VIDEO_CHECKERS_HPP__
