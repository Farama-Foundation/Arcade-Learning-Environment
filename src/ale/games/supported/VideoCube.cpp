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

#include "ale/games/supported/VideoCube.hpp"

#include <cmath>
#include <stdexcept>

#include "ale/games/RomUtils.hpp"

namespace ale {
using namespace stella;

VideoCubeSettings::VideoCubeSettings() { reset(); }

/* *****************************************************************************
 * VideoCube findings.
 *
 * Face colours - stored in rom at $fee6
 * White(Mauve):0x7f, Green:0xd6, Blue:0x88, Purple:0x66, Red:0x44 , Orange:0x38
 *
 * Face colour index mappings.
 * 0x00:White(Mauve), 0x01:Green, 0x02:Blue, 0x03:Purple, 0x04:Red, 0x05:Orange
 *
 * Location ram_f7 contains the current player colour (see above for mappings).
 * Location ram_f3 contains the player's column (0: 0x00, 1: 0x06, 2: 0x0c).
 * Location ram_f4 contains the player's row (0: 0x00, 1: 0x06, 2: 0x0c).
 *
 * Location ram_fb contains the currently selected game mode.
 *
 * Location ram_9f contains the currently selected game cube.
 *
 * Locations ram_df, ram_e0, ram_e1 contain six digits of time/score data stored
 * in BCD format.
 *
 * Location ram_db contains a game timer, which increments approximately every 5
 * seconds.  Once the timer ticks over from 0xff to 0x00 the computer takes over
 * control from the player and finishes solving the cube, effectively ending the
 * game for the player.
 * The game timer is reset back to 0x00 every time the player successfully moves
 * the character on screen.
 * We're using this as a terminating condition.
 *
 * Locations ram_a0 --> ram_d5 (inclusive) contain the face data for all 6 faces
 * of the cube, so each block of 9 bytes maps to 3x3 blocks on a face.  Each
 * byte contains one of the face colour indices as listed above.
 *
 * Locations ram_81 --> ram_89 appear to the hold the colour information of the
 * face currently on screen.
 *
 * *****************************************************************************
 */

const std::uint8_t kFaceStartAddress = 0xa0;
const int kMaxBlocksPerFace = 9;
const int kMaxFaces = 6;
const int kMaxMode = 3;

RomSettings* VideoCubeSettings::clone() const {
  return new VideoCubeSettings(*this);
}

void VideoCubeSettings::step(const System& system) {
  // We originally planned to use this as part of the reward, but it doesn't
  // feel quite right that agents should be punished for exploring.  However
  // we're still capturing this data should it be required during a later
  // rethink of the reward function.
  int turnsTaken = getDecimalScore(0xdf, 0xe0, 0xe1, &system);

  // We need to go through each face to see if the colour blocks match, and
  // count the total numbers of matched faces.
  int completeFaceCount = 0;
  std::uint8_t blockAddress = kFaceStartAddress;
  for (int c0 = 0; c0 < kMaxFaces; c0++) {
    // Grab the first colour block of the current face.
    std::uint8_t firstColourBlock = readRam(&system, blockAddress);

    // Analyse the remaining 8 blocks of the face to see if they match the first
    // colour block.
    bool allMatch = true;
    for (int c1 = 1; c1 < kMaxBlocksPerFace; c1++) {
      allMatch = readRam(&system, blockAddress + c1) == firstColourBlock;
      if (!allMatch) {break;}
    }

    // Count the face.
    if (allMatch) {completeFaceCount++;}

    // Check the next face (next nine bytes).
    blockAddress += kMaxBlocksPerFace;
  }

  // The game timer wrapping around to 0x00 triggers an end game condition,
  // which manifests itself as the computer taking over control and solving the
  // cube.
  // Because the timer gets reset back to 0x00 at the start of a new game, or
  // when a player makes a guess, we need to check that a value of 0x00 was
  // preceeded by 0xff as an indicator that the timer overflowed.`

  // Shift the previous timer along, and grab the latest timer value.
  m_timerArray[1] = m_timerArray[0];
  m_timerArray[0] = readRam(&system, 0xdb);

  // If the sequence is 0x00 followed by 0xff then let's
  // use that as a time out termination condition.
  bool timedOut = m_timerArray[0] == 0 && m_timerArray[1] == 255;

  // Set the reward: -1 if we time out, or the current face count delta.
  m_reward = timedOut ? -1 : completeFaceCount - m_faceCount;
  m_faceCount = completeFaceCount;

  // update terminal status
  m_terminal = timedOut || completeFaceCount == 6;
}

bool VideoCubeSettings::isTerminal() const { return m_terminal; }

reward_t VideoCubeSettings::getReward() const { return m_reward; }

bool VideoCubeSettings::isMinimal(const Action& a) const {
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

void VideoCubeSettings::reset() {
  m_cubeNumber = 1;
  m_reward = 0;
  m_faceCount = 0;
  m_terminal = false;
}

void VideoCubeSettings::saveState(Serializer& ser) {
  ser.putInt(m_cubeNumber);
  ser.putInt(m_faceCount);
  ser.putInt(m_reward);
  ser.putBool(m_terminal);
}

void VideoCubeSettings::loadState(Deserializer& ser) {
  m_cubeNumber = ser.getInt();
  m_faceCount = ser.getInt();
  m_reward = ser.getInt();
  m_terminal = ser.getBool();
}

DifficultyVect VideoCubeSettings::getAvailableDifficulties() {
  return {0, 1};
}

ActionVect VideoCubeSettings::getStartingActions() {
  // This game requires selection of a cube type at the start of play, with the
  // fire button being pressed to confirm the selection and start the game.
  // We do the cube selection in setMode, but force a "fire button" press after
  // approximately a second of waiting about.
  ActionVect startingActions(61, PLAYER_A_NOOP);

  // Press fire to start and wait a couple of frames.
  startingActions.push_back(PLAYER_A_FIRE);
  startingActions.push_back(PLAYER_A_NOOP);
  startingActions.push_back(PLAYER_A_NOOP);

  return startingActions;
}

ModeVect VideoCubeSettings::getAvailableModes() {
  ModeVect availableModes;

  // For each mode we want to create 50 combinations representing the 50
  // selectable cubes i.e. mode to pass = game mode + (cube number * 100).
  for (int c0 = 0; c0 <= 50; c0++) {
    for (int c1 = 0; c1 < kMaxMode; c1++) {
      availableModes.push_back( c0 * 100 + c1);
    }
  }

  return availableModes;
}

// According to https://atariage.com/manual_html_page.php?SoftwareLabelID=974
// there are 18 game modes/variations.
// We're only exposing 3 game modes at this stage, as the Speed, and Scoring by
// Time variations are irrelent (we're scoring by completion), and the Computer
// Play options seems to only ever play the default Cube and has no user input.
// The game variations we're exposing, as per the game select matrix in the link
// above, are 1 (normal speed), 3 (normal speed, blacked out), and 9 (normal
// speed, restricted movement).  These map externally to modes 0,1, and 2
// respectively.
void VideoCubeSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {
  const int modeMap[kMaxMode] = {1, 3, 9};

    // Has a cube number been encoded into the mode number?
  if (m >= 100) {
    // Yes, so let's extract the cube number...
    m_cubeNumber = std::floor(m/100);

    // ...and make sure it's in range.
    if (m_cubeNumber > 50) {
      throw std::runtime_error("The cube number is out of range.");
    }
  }

  // Filter the mode index.
  int modeIndex = m%100;
  if (modeIndex < kMaxMode) {
    // Make a note of the actual required game mode.
    int requiredMode = modeMap[modeIndex] - 1;

    // Read the current game mode...the game uses the upper 3 bits for
    // "something", so we need to mask those out here.
    unsigned char currentMode = (readRam(&system, 0xfb) & 0x1f);

    // Press select until the correct mode is reached for single player only.
    while (currentMode != requiredMode) {
      environment->pressSelect(2);
      currentMode = (readRam(&system, 0xfb) & 0x1f);
    }

    // Reset the environment to apply changes.
    environment->softReset();

    // Press "right" until the correct cube is selected.
    unsigned char cube = getDecimalScore(0x9f, &system);
    while (cube != m_cubeNumber) {
      environment->act(PLAYER_A_RIGHT, PLAYER_B_NOOP);
      cube = getDecimalScore(0x9f, &system);
    }
  } else {
    throw std::runtime_error("This game mode is not supported.");
  }
}

}  // namespace ale
