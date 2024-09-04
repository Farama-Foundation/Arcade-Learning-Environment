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

#ifndef __ROMS_HPP__
#define __ROMS_HPP__

#include <filesystem>
#include <string>

#include "ale/games/RomSettings.hpp"

namespace fs = std::filesystem;

namespace ale {

// looks for the RL wrapper corresponding to a particular rom title
RomSettings* buildRomRLWrapper(const fs::path& rom, const std::string md5);

}  // namespace ale

#endif  // __ROMS_HPP__
