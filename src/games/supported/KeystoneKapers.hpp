/* *****************************************************************************
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

#ifndef __KEYSTONEKAPERS_HPP__
#define __KEYSTONEKAPERS_HPP__

#include "../RomSettings.hpp"

namespace ale {

/* RL wrapper for KeystoneKapers */
class KeystoneKapersSettings : public RomSettings {
 public:
  KeystoneKapersSettings();

  // reset
  void reset();

  // is end of game
  bool isTerminal() const;

  // get the most recently observed reward
  reward_t getReward() const;

  // the rom-name
  // MD5 be929419902e21bd7830a7a7d746195d
  const char* rom() const { return "keystone_kapers"; }

  // create a new instance of the rom
  RomSettings* clone() const;

  // is an action part of the minimal set?
  bool isMinimal(const Action& a) const;

  // process the latest information from ALE
  void step(const System& system);

  // saves the state of the rom settings
  void saveState(Serializer& ser);

  // loads the state of the rom settings
  void loadState(Deserializer& ser);

  // Keystone Kapers requires the reset button to start the game
  ActionVect getStartingActions();

  virtual int lives() { return isTerminal() ? 0 : m_lives; }

 private:
  bool m_terminal;
  reward_t m_reward;
  reward_t m_score;
  int m_lives;
};

}  // namespace ale

#endif  // __KEYSTONEKAPERS_HPP__
