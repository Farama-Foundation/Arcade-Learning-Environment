/* *****************************************************************************
 *
 * This wrapper was authored by Stig Petersen, March 2014
 *
 * Xitari
 *
 * Copyright 2014 Google Inc.
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
#include "VideoChess.hpp"

#include "../RomUtils.hpp"


using namespace ale;


VideoChessSettings::VideoChessSettings() {

    reset();
}


/* create a new instance of the rom */
RomSettings* VideoChessSettings::clone() const {

    RomSettings* rval = new VideoChessSettings();
    *rval = *this;
    return rval;
}


/* Calculate the material value based on pieces present on the board */
int VideoChessSettings::CalculateMaterialValue(const System& system) const {
//    const int KING_VALUE    = 10000;
    const int QUEEN_VALUE   = 9;
    const int ROOK_VALUE    = 5;
    const int KNIGHT_VALUE  = 3;
    const int BISHOP_VALUE  = 3;
    const int PAWN_VALUE    = 1;

//    const int BLACK_KING_ID     = 0x01;
    const int BLACK_QUEEN_ID    = 0x02;
    const int BLACK_BISHOP_ID   = 0x03;
    const int BLACK_KNIGHT_ID   = 0x04;
    const int BLACK_ROOK_ID     = 0x05;
    const int BLACK_PAWN_ID     = 0x06;
//    const int WHITE_KING_ID     = 0x09;
    const int WHITE_QUEEN_ID    = 0x0A;
    const int WHITE_BISHOP_ID   = 0x0B;
    const int WHITE_KNIGHT_ID   = 0x0C;
    const int WHITE_ROOK_ID     = 0x0D;
    const int WHITE_PAWN_ID     = 0x0E;

    int totalMaterialValue = 0;

    // Loop through board squares and sum up material values
    for (int address = 0x80; address < 0xC0; ++address) {
        int squareState = (readRam(&system, address) & 0x0F);
        switch (squareState) {
        case BLACK_QUEEN_ID:
            totalMaterialValue -= QUEEN_VALUE;
            break;
        case BLACK_BISHOP_ID:
            totalMaterialValue -= BISHOP_VALUE;
            break;
        case BLACK_KNIGHT_ID:
            totalMaterialValue -= KNIGHT_VALUE;
            break;
        case BLACK_ROOK_ID:
            totalMaterialValue -= ROOK_VALUE;
            break;
        case BLACK_PAWN_ID:
            totalMaterialValue -= PAWN_VALUE;
            break;
        case WHITE_QUEEN_ID:
            totalMaterialValue += QUEEN_VALUE;
            break;
        case WHITE_BISHOP_ID:
            totalMaterialValue += BISHOP_VALUE;
            break;
        case WHITE_KNIGHT_ID:
            totalMaterialValue += KNIGHT_VALUE;
            break;
        case WHITE_ROOK_ID:
            totalMaterialValue += ROOK_VALUE;
            break;
        case WHITE_PAWN_ID:
            totalMaterialValue += PAWN_VALUE;
            break;
        }
    }

    return totalMaterialValue;
}


/* process the latest information from ALE */
void VideoChessSettings::step(const System& system) {

    // TURN_BLACK = 0x0;
    const int TURN_WHITE = 0x82;
    int currentPlayer = readRam(&system, 0xE1);

    if (currentPlayer == TURN_WHITE) {

        int materialValue = CalculateMaterialValue(system);
        m_reward = materialValue - m_materialValue;
        m_materialValue = materialValue;

        // 0xEE == 0: check mate black
        // 0xEE == 1: check mate white
        // 0xEE == 3: game ongoing
        const int CHECK_MATE_BLACK = 0x00;
        const int CHECK_MATE_WHITE = 0x01;
        int checkMateByte = readRam(&system, 0xEE);

        const int CHECK_MATE_REWARD = 10000;

        if (checkMateByte == CHECK_MATE_BLACK) {
            m_reward += CHECK_MATE_REWARD;
            m_terminal = true;
        }
        else if (checkMateByte == CHECK_MATE_WHITE) {
            m_reward -= CHECK_MATE_REWARD;
            m_terminal = true;
        }
    }
    else // Atari AI simulates moves while it searches the tree, so we want to ignore those
    {
        m_reward = 0;
    }

    // 0xE4: Number of moves by black

    // Notes on addresses that change when players make a move
    // E0*, E1=82/0 , E2*, E3*, E4*, E5=FF/80/0C, E6*, E7*, E8*, E9*, EA*, EB=40/98, EC*, ED*, *EE, *EF
    // *change wildly during colour flashes
}


/* is end of game */
bool VideoChessSettings::isTerminal() const {

    return m_terminal;
};


/* get the most recently observed reward */
reward_t VideoChessSettings::getReward() const {

    return m_reward;
}


/* is an action part of the minimal set? */
bool VideoChessSettings::isMinimal(const Action &a) const {

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
            return true;
        default:
            return false;
    }
}


/* reset the state of the game */
void VideoChessSettings::reset() {

    m_reward   = 0;
    m_terminal = false;
    m_lives    = 1;
    m_materialValue = 0;
}


/* saves the state of the rom settings */
void VideoChessSettings::saveState(Serializer & ser) {
  ser.putInt(m_reward);
  ser.putBool(m_terminal);
  ser.putInt(m_lives);
  ser.putInt(m_materialValue);
}

// loads the state of the rom settings
void VideoChessSettings::loadState(Deserializer & ser) {
  m_reward = ser.getInt();
  m_terminal = ser.getBool();
  m_lives = ser.getInt();
  m_materialValue = ser.getInt();
}

