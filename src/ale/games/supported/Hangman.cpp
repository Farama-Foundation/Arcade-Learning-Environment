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

#include "ale/games/supported/Hangman.hpp"

#include "ale/games/RomUtils.hpp"

namespace ale {
using namespace stella;

HangmanSettings::HangmanSettings() { reset(); }

/* *****************************************************************************
 * Hangman findings.
 *
 * Character encoding:
 * A: 0x00 --> Z: 0x19, with 0x1a representing a null character.
 *
 * Maximum word length is 6 letters, minimum word length is 3 letters, with the
 * first letter of the current word stored at location ram_f3, and the last at
 * ram_f8 (0x1a is used to fill in the blanks where a word is less than 6
 * letters).
 *
 * e.g. CAT would be represented as |0x02|0x00|0x13|0x1a|0x1a|0x1a|
 *
 * Location ram_fd contains a bit pattern representing the status of the current
 * word.  Bits 5-0 each represent a letter, while bits 7-6 are unused.
 * A set bit represents the known state of a letter, so in our example above
 * ram_fd would be initialised to %00000111 when a new word is set.  Once the
 * word has been completely guessed, game would expect to find %00111111 (0x3f)
 * and terminate.
 *
 * Location ram_ef contains the current player selected letter (same encoding
 * as above).
 *
 * Location ram_eb contains player 1(the guesser)'s score.
 *
 * Location ram_ec contains player 2(the setter)'s score.
 *
 * Location ram_ee contains the currently selected game mode.
 *
 * Resetting the game generates a new word, but maintains the player scores
 * across resets.
 *
 * Location ram_f1 contains a game timer, and counts off 0xff seconds before
 * properly terminating a game.  When a game is terminated it goes into a colour
 * cycling mode, whilst maintaining the player scores.  A subsequent reset here
 * will reset the player scores.
 * The game timer is reset back to 0x00 every time the player guesses a letter.
 *
 * Location ram_fe contains a turn timer, which only has an effect in advanced
 * mode.  After 20 seconds (in advanced mode), the player loses a guess (the
 * next part of the hanging monkey is drawn).
 * The turn timer is reset back to 0x00 every time the player guesses a letter.
 *
 * *****************************************************************************
 */

/* create a new instance of the rom */
RomSettings* HangmanSettings::clone() const {
  return new HangmanSettings(*this);
}

/* process the latest information from ALE */
void HangmanSettings::step(const System& system) {
  // update the reward
  int computerScore = getDecimalScore(0xEc, &system);
  int playerScore = getDecimalScore(0xEb, &system);

  // Because we can't force a soft reset and generate a new word, we'll register
  // the reward here and use the change in score as a terminating condition.
  // For this reason there's no need to test for the scores wrapping around to
  // zero.
  // +1 for guessing a word correctly, -1 for failing.
  m_reward = (playerScore - m_playerScore) - (computerScore - m_computerScore);
  m_computerScore = computerScore;
  m_playerScore = playerScore;

  // The game timer wrapping around to 0x00 triggers an end game condition.
  // Because the timer gets reset back to 0x00 at the start of a new game, or
  // when a player makes a guess, we need to check that a value of 0x00 was
  // preceeded by 0xff as an indicator that the timer overflowed.

  // Shift the previous timer along, and grab the latest timer value.
  m_timerArray[1] = m_timerArray[0];
  m_timerArray[0] = readRam(&system, 0xf1);

  // If the sequence is 0x00 followed by 0xff then let's
  // use that as a time out termination condition.
  bool timedOut = m_timerArray[0] == 0 && m_timerArray[1] == 255;

  // update terminal status
  m_terminal = m_reward != 0 || timedOut;
}

/* is end of game */
bool HangmanSettings::isTerminal() const { return m_terminal; }

/* get the most recently observed reward */
reward_t HangmanSettings::getReward() const {
  if (m_reward != 0) {
    // Ideally we should soft reset here, so we can force the game to generate
    // the next word.
  }
  return m_reward;
}

/* is an action part of the minimal set? */
bool HangmanSettings::isMinimal(const Action& a) const {
  switch (a) {
    case PLAYER_A_NOOP:
    case PLAYER_A_FIRE:
    case PLAYER_A_UP:
    case PLAYER_A_RIGHT:
    case PLAYER_A_LEFT:
    case PLAYER_A_DOWN:
    case PLAYER_A_UPRIGHT:
    case PLAYER_A_UPLEFT:
    case PLAYER_A_DOWNRIGHT:
    case PLAYER_A_DOWNLEFT:
    case PLAYER_A_UPFIRE:
    case PLAYER_A_RIGHTFIRE:
    case PLAYER_A_LEFTFIRE:
    case PLAYER_A_DOWNFIRE:
    case PLAYER_A_UPRIGHTFIRE:
    case PLAYER_A_UPLEFTFIRE:
    case PLAYER_A_DOWNRIGHTFIRE:
    case PLAYER_A_DOWNLEFTFIRE:
      return true;
    default:
      return false;
  }
}

/* reset the state of the game */
void HangmanSettings::reset() {
  m_reward = 0;
  m_computerScore = 0;
  m_playerScore = 0;
  m_terminal = false;
  m_timerArray[0] = -1;
  m_timerArray[1] = -1;
}

/* saves the state of the rom settings */
void HangmanSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putInt(m_computerScore);
  ser.putInt(m_playerScore);
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void HangmanSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_computerScore = ser.getInt();
  m_playerScore = ser.getInt();
  m_terminal = ser.getBool();
}

// According to https://atariage.com/manual_page.php?SystemID=2600&SoftwareLabelID=230&currentPage=4
// the left difficulty switch adds a handicap when set to A, which gives the
// player only 20 seconds to make their move.
DifficultyVect HangmanSettings::getAvailableDifficulties() {
  return {0, 1};
}

// According to https://atariage.com/manual_page.php?SystemID=2600&SoftwareLabelID=230&currentPage=4
// there are 9 game modes. 1-4 are one-player games, 5-8 are two-player games,
// while 9 is a challenge game where player two sets a word for player 1.
// We only support game modes 1-4 currently.
// Game modes set the vocabularly according to school grade level:
// 1 = 1st->3rd grade, 2 = 1st->6th grade, 3 = 1st->9th, 4 = 1st->high school.
ModeVect HangmanSettings::getAvailableModes() {
  return {0, 1, 2, 3};
}

void HangmanSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {
  if (m < 4) {
    // Read the mode we are currently in.
    unsigned char mode = readRam(&system, 0xee);

    // Press select until the correct mode is reached for single player only.
    while (mode != m) {
      environment->pressSelect(2);
      mode = readRam(&system, 0xee);
    }

    // Reset the environment to apply changes.
    environment->softReset();
  } else {
    throw std::runtime_error("This game mode is not supported.");
  }
}

}  // namespace ale
