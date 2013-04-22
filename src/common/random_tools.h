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
 *  random_tools.h
 *  
 *  Implementation of a set of tools for random number generation
 *
 **************************************************************************** */
#ifndef __RANDOM_TOOLS_H__
#define __RANDOM_TOOLS_H__

#include <vector>
#include <cstdlib> 
#include "../emucore/m6502/src/bspf/src/bspf.hxx"

/* *********************************************************************
    Returns a random integer within the [lowest, highest] range.
    Code taken from here: http://www.daniweb.com/forums/thread1769.html
 ******************************************************************** */
inline int rand_range(int lowest, int highest) {
    int range = highest - lowest + 1;
    return (rand() % range) + lowest; 
}

/* *********************************************************************
    Returns a random element of the given vector
 ******************************************************************** */
template <class T> 
inline T choice(const vector<T>* p_vec) {
    assert(p_vec->size() > 0);
    unsigned int index = rand_range(0, p_vec->size() - 1);
    assert(index >= 0);
    assert(index < p_vec->size());
    return (*p_vec)[index];
}

#endif // __RANDOM_TOOLS_H__
