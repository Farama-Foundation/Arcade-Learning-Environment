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
 *  DynamicLoad.hpp
 *
 *  Helper functions to manage dynamic loading libraries.
 **************************************************************************** */
#ifndef DYNAMIC_LOAD_HPP
#define DYNAMIC_LOAD_HPP

namespace ale {

/*
 * Links function `source` from `library` to function pointer `fn`.
 */
bool DynamicLinkFunction(void** fn, const char* source, const char* library);

} // namespace ale
#endif // DYNAMIC_LOAD_HPP
