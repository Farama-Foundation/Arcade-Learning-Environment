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
 *  ale_ram.hpp
 *
 *  A class that encapsulates the Atari 2600 RAM. Code is provided inline for
 *   efficiency reasonss.
 *
 **************************************************************************** */

#ifndef __ALE_RAM_HPP__
#define __ALE_RAM_HPP__

#include <cstddef>
#include <cstring>

namespace ale {

using byte_t = unsigned char;

/** A simple wrapper around the Atari RAM. */
class ALERAM {
  static constexpr std::size_t kRamSize = 128;

 public:
  /** Byte accessors: x is wrapped to [0, 128). */
  byte_t get(unsigned int x) const;
  byte_t* byte(unsigned int x);

  /** Returns a pointer to the first element of the entire
      array (equivalent to &byte[0]). */
  const byte_t* array() const { return m_ram; }

  std::size_t size() const { return sizeof(m_ram); }

  /** Returns whether two copies of the RAM are equal */
  bool equals(const ALERAM& rhs) const {
    return std::memcmp(m_ram, rhs.m_ram, size()) == 0;
  }

 protected:
  byte_t m_ram[kRamSize];
};

// Byte accessors
inline byte_t ALERAM::get(unsigned int x) const {
  // Wrap RAM around the first 128 bytes
  return m_ram[x & 0x7F];
}

inline byte_t* ALERAM::byte(unsigned int x) {
  // Wrap RAM around the first 128 bytes
  return &m_ram[x & 0x7F];
}

}  // namespace ale

#endif  // __ALE_RAM_HPP__
