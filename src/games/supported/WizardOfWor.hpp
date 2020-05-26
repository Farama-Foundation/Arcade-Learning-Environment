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

#ifndef __WIZARDOFWOR_HPP__
#define __WIZARDOFWOR_HPP__

#include "../RomSettings.hpp"
#include "../RomSettings2P.hpp"

namespace ale {

/* RL wrapper for Wizard of Wor */
class WizardOfWorSettings : public RomSettings2P {
 public:
  WizardOfWorSettings();

  // reset
  void reset() override;

  // is end of game
  bool isTerminal() const override;

  // get the most recently observed reward
  reward_t getReward() const override;
  reward_t getRewardP2() const override;

  // the rom-name
  const char* rom() const override { return "wizard_of_wor"; }

  // The md5 checksum of the ROM that this game supports
  const char* md5() const override { return "7e8aa18bc9502eb57daaf5e7c1e94da7"; }

  // create a new instance of the rom
  RomSettings* clone() const override;

  // is an action part of the minimal set?
  bool isMinimal(const Action& a) const override;

  // process the latest information from ALE
  void step(const System& system) override;

  // saves the state of the rom settings
  void saveState(Serializer& ser) override;

  // loads the state of the rom settings
  void loadState(Deserializer& ser) override;

  int lives() override;
  int livesP2() override;


  ModeVect getAvailableModes() override;
  ModeVect get2PlayerModes() override;

  // returns a list of difficulties that the game can be played in
  // in this game, there are 2 available difficulties
  DifficultyVect getAvailableDifficulties() override;

  void setMode(game_mode_t, System& system,
               std::unique_ptr<StellaEnvironmentWrapper> environment) override;


 private:
  bool m_terminal;
  bool is_two_player;
  reward_t m_reward;
  reward_t m_reward_p2;
  reward_t m_score;
  reward_t m_score_p2;
  int m_lives;
  int m_lives_p2;
};

}  // namespace ale

#endif  // __WIZARDOFWOR_HPP__
