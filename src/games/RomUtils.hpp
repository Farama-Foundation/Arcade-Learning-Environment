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
 *
 * RomUtils.hpp
 *
 * Additional utilities to operate on RAM.
 * *****************************************************************************
 */

#ifndef __ROMUTILS_HPP__
#define __ROMUTILS_HPP__

namespace ale {

namespace stella{
class System;
}

// reads a byte at a RAM location between 0 and 0x7f also mapped to [0x80, 0xff]
extern int readRam(const stella::System* system, int offset);

// reads a byte from anywhere in the memory map between 0x0000 and 0xffff.
extern int readMappedRam(const stella::System* system, int offset);

// extracts a decimal value from 1, 2, and 3 bytes respectively
extern int getDecimalScore(int idx, const stella::System* system);
extern int getDecimalScore(int lo, int hi, const stella::System* system);
extern int getDecimalScore(int lo, int mid, int hi, const stella::System* system);

}  // namespace ale

#endif  // __ROMUTILS_HPP__
