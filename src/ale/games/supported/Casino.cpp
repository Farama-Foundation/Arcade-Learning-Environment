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

#include "ale/games/supported/Casino.hpp"

#include "ale/games/RomUtils.hpp"
#include "ale/common/Constants.h"

namespace ale {
using namespace stella;

CasinoSettings::CasinoSettings() { reset(); }

RomSettings* CasinoSettings::clone() const {
  return new CasinoSettings(*this);
}

void CasinoSettings::step(const System& system) {
  int score = getDecimalScore(0x95, 0x8c, &system);
  int game_mode = readRam(&system, 0xd4);
  if (game_mode == 3) {
    // For Poker Solitaire the game terminates after all 25 cards have been
    // placed down and the final score has been calculated.
    bool finished_awarding = readRam(&system, 0x9e) == 0xaa;
    m_reward = score - m_score;
    m_terminal = score > 0 && finished_awarding;
  } else {
    // For Blackjack or Stud Poker the game terminates when the player is bust
    // or 'breaks the bank' with 10,000 chips. Unfortunately we cannot detect
    // a decimal value > 9999 so we need to check whether the player input is
    // disabled by checking the top-bit of RAM address 0xD3.
    bool input_disabled = readRam(&system, 0xd3) & 0x80;
    // Only update the reward if the player has control in the game because
    // the player is given 1000 chips back at the moment the game ends, ready
    // for the next game. This nullifies the reward value if included.
    if (!input_disabled) {
      m_reward = score - m_score;
    }
    // The game isn't considered to have started until there is a valid bet.
    int bet_value = getDecimalScore(0x9e, &system);
    m_terminal = score == 0 || (bet_value > 0 && input_disabled);
  }

  m_score = score;
}

bool CasinoSettings::isTerminal() const { return m_terminal; }

reward_t CasinoSettings::getReward() const { return m_reward; }

// Note that Casino uses the paddle controller but a slightly different
// mapping is required to reach the full range of betting values and to select
// all possible betting actions. Recommend setting paddle_max = 1290196.
bool CasinoSettings::isMinimal(const Action& a) const {
  switch (a) {
    case PLAYER_A_NOOP:
    case PLAYER_A_FIRE:
    case PLAYER_A_UP:
    case PLAYER_A_DOWN:
      return true;
    default:
      return false;
  }
}

void CasinoSettings::reset() {
  m_reward = 0;
  m_score = 0;
  m_terminal = false;
}

void CasinoSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
}

void CasinoSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
}

// According to https://atariage.com/manual_html_page.php?SoftwareID=909
// For Blackjack:
//  Left difficulty switch sets how frequently the computer shuffles the cards.
//  Right difficulty switch sets which set of casino rules are used.
// For Stud Poker:
//  Left difficulty sets whether the dealer's first card is dealt face up or
//  face down. Right difficulty sets whether the player's first card is dealt
//  face up or face down.
// Not used in Poker Solitaire.
DifficultyVect CasinoSettings::getAvailableDifficulties() {
  return {0, 1, 2, 3};
}

// According to https://atariage.com/manual_html_page.php?SoftwareID=909
// there are four game modes. Games 1 and 2 are Blackjack but game 2 merely
// increases the number of possible players, so we ignore this mode.
// Game 3 is Stud Poker and game 4 is Poker Solitaire.
ModeVect CasinoSettings::getAvailableModes() {
  return {0, 2, 3};
}

void CasinoSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {
  if (m < 4 && m != 1) {
    // Read the mode we are currently in.
    int mode = readRam(&system, 0xd4);

    // Press select until the correct mode is reached for single player only.
    while (mode != m) {
      environment->pressSelect(2);
      mode = readRam(&system, 0xd4);
    }

    // Reset the environment to apply changes.
    environment->softReset();
  } else {
    throw std::runtime_error("This game mode is not supported.");
  }
}

ActionVect CasinoSettings::getStartingActions() {
  // Need to wait for one second (60 frames) for the cards to be shuffled.
  ActionVect startingActions(60, PLAYER_A_NOOP);

  // Press fire for a couple of frames to enter player A into the game.
  startingActions.push_back(PLAYER_A_FIRE);
  startingActions.push_back(PLAYER_A_FIRE);

  return startingActions;
}

}  // namespace ale
