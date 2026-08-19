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

#include "ale/games/supported/MazeCraze.hpp"

#include "ale/games/RomUtils.hpp"

namespace ale {
using namespace stella;

MazeCrazeSettings::MazeCrazeSettings() { reset(); }

RomSettings* MazeCrazeSettings::clone() const {
  return new MazeCrazeSettings(*this);
}

void MazeCrazeSettings::step(const System& system) {
  int player1_score = readRam(&system, 0xEC) == 0xff ? 1 : 0;
  int player2_score = readRam(&system, 0xED) == 0xff ? 1 : 0;

  m_reward_p1 = 0;
  m_reward_p2 = 0;
  // a player is killed if the bit saying it cannot move
  // is set for a couple seconds.
  if (p1_isalive && readRam(&system, 0xEA) & 0x40) {
    p1_isalive = false;
    m_reward_p1 = -1;
  }
  if (p2_isalive && readRam(&system, 0xEB) & 0x40) {
    p2_isalive = false;
    m_reward_p2 = -1;
  }
  int completion_score = player1_score - player2_score;
  if (completion_score != 0) {
    if (!p1_isalive) {
      m_reward_p2 = 1;
    } else if (!p2_isalive) {
      m_reward_p1 = 1;
    } else {
      m_reward_p1 = completion_score;
      m_reward_p2 = -completion_score;
    }
  }

  // game is over when some player wins, i.e. reward is not zero,
  // or both players are dead
  m_terminal = completion_score != 0 || (!p1_isalive && !p2_isalive);
}

bool MazeCrazeSettings::isTerminal() const { return m_terminal; }

reward_t MazeCrazeSettings::getReward() const { return m_reward_p1; }
reward_t MazeCrazeSettings::getRewardP2() const { return m_reward_p2; }

int MazeCrazeSettings::lives() { return p1_isalive ? 0 : -1; }
int MazeCrazeSettings::livesP2() { return p2_isalive ? 0 : -1; }

bool MazeCrazeSettings::isMinimal(const Action& a) const {
  switch (a) {
    // the joystick usually doesn't need to fire. Only used for the
    // "player peek" functionality
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

void MazeCrazeSettings::reset() {
  m_reward_p1 = 0;
  m_reward_p2 = 0;
  m_score = 0;
  p1_isalive = true;
  p2_isalive = true;
  m_terminal = false;
}

void MazeCrazeSettings::saveState(Serializer& ser) {
  ser.putInt(m_reward_p1);
  ser.putInt(m_reward_p2);
  ser.putInt(m_score);
  ser.putBool(m_terminal);
  ser.putBool(p1_isalive);
  ser.putBool(p2_isalive);
}

void MazeCrazeSettings::loadState(Deserializer& ser) {
  m_reward_p1 = ser.getInt();
  m_reward_p2 = ser.getInt();
  m_score = ser.getInt();
  m_terminal = ser.getBool();
  p1_isalive = ser.getBool();
  p2_isalive = ser.getBool();
}

DifficultyVect MazeCrazeSettings::getAvailableDifficulties() {
  // According to https://atariage.com/manual_html_page.php?SoftwareLabelID=931
  // the left difficulty switch controls player a speed and the
  // right difficulty switch controls player b speed
  return {0};
}

// Maze Craze has no single-player variants: every game variant needs two
// players.
ModeVect MazeCrazeSettings::getAvailableModes() {
  return {};
}

// The 64 modes combine the 16 game variants with the 4 speed settings
// (RAM 0xBD holds variant * 4 + speed).
ModeVect MazeCrazeSettings::get2PlayerModes() {
  ModeVect modes;
  for (int variant = 0; variant < 16; variant++) {
    for (int speed = 0; speed < 4; speed++) {
      modes.push_back(variant * 4 + speed);
    }
  }
  return modes;
}

void MazeCrazeSettings::setMode(
    game_mode_t m, System& system,
    std::unique_ptr<StellaEnvironmentWrapper> environment) {
  // Press select until the correct mode is reached.
  while (readRam(&system, 0xbd) != m) {
    environment->pressSelect(2);
  }

  // Reset the environment to apply changes.
  environment->softReset();
}

}  // namespace ale
