/* *****************************************************************************
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

#include "games/supported/Adventure.hpp"

#include "games/RomUtils.hpp"

namespace ale {

AdventureSettings::AdventureSettings() { reset(); }

/* create a new instance of the rom */
RomSettings* AdventureSettings::clone() const {
  return new AdventureSettings(*this);
}

/* process the latest information from ALE */
void AdventureSettings::step(const System& system) {
  int chalice_status = readRam(&system, 0xB9);
  bool chalice_in_yellow_castle = chalice_status == 0x12;

  if (chalice_in_yellow_castle) {
    m_reward = 1;
  }

  int player_status = readRam(&system, 0xE0);
  bool player_eaten = player_status == 2;

  m_terminal = player_eaten || chalice_in_yellow_castle;
}

/* is end of game */
bool AdventureSettings::isTerminal() const { return m_terminal; }

/* get the most recently observed reward */
reward_t AdventureSettings::getReward() const { return m_reward; }

/* is an action part of the minimal set? */
bool AdventureSettings::isMinimal(const Action& a) const {
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
void AdventureSettings::reset() {
  m_reward = 0;
  m_terminal = false;
}

/* saves the state of the rom settings */
void AdventureSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward);
  ser.putBool(m_terminal);
}

// loads the state of the rom settings
void AdventureSettings::loadState(Deserializer& ser) {
  m_reward = ser.getInt();
  m_terminal = ser.getBool();
}

// Returns the supported modes for the game.
ModeVect AdventureSettings::getAvailableModes() {
  return {0, 1, 2};
}

// Set the game mode.
// The given mode must be one returned by the previous function.
// According to Wikipedia (https://en.wikipedia.org/wiki/Adventure_(Atari_2600))
// Adventure has 3 game modes:
// Level 1 is the easiest, as it uses a simplified room layout and doesn't
//   include the White Castle, bat, Rhindle the red dragon, nor invisible mazes.
// Level 2 is the full version of the game, with the various objects appearing
//   in set positions at the start of the game.
// Level 3 is similar to Level 2, but the location of the objects is
//   randomized to provide a more challenging game. The randomiser uses the
//   low byte of its internal frame counter at RAM address 0xE5 to seed the rng.
//   We press select for a random amount of time to enable different layouts to
//   be generated.
void AdventureSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {
  if (m < 3) {
    Random& rng = environment->getSystemRng();
    // Read the mode we are currently in.
    unsigned char mode = (readRam(&system, 0xDD) >> 1) & 0x03;

    // Press select until the correct mode is reached.
    while (mode != m) {
      // Press select for a random amount of time as the randomiser for level 3
      // uses the 1 byte elapsed frame counter at RAM address 0xE5 as a seed for
      // its internal rng used to configure the object spawning in rooms.
      environment->pressSelect(2 + rng.next() % 256);
      // Adventure uses a debouncer so need to wait before the select takes
      // effect.
      environment->act(PLAYER_A_NOOP, PLAYER_B_NOOP);
      mode = (readRam(&system, 0xDD) >> 1) & 0x03;
    }
    // Reset the environment to apply changes.
    environment->softReset();
  } else {
    throw std::runtime_error("This game mode is not supported.");
  }
}

// Return the supported difficulty settings for the game.
// According to Wikipedia (https://en.wikipedia.org/wiki/Adventure_(Atari_2600))
// one difficulty switch controls controls the dragons' bite speed, and one
// causes them to flee when the player is wielding the sword.
DifficultyVect AdventureSettings::getAvailableDifficulties() {
  return {0, 1, 2, 3};
}

}  // namespace ale
