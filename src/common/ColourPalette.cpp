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
 *  ColourPalette.cpp 
 *
 *  Enables conversion from NTSC/SECAM/PAL to RGB via the OSystem's palette.
 **************************************************************************** */

#include "ColourPalette.hpp"
#include <cassert>
#include <string.h>
#include <stdio.h>

ColourPalette::ColourPalette():
    m_palette(NULL) {
}


void ColourPalette::getRGB(int val, int &r, int &g, int &b) const {

    assert (m_palette != NULL);
    assert(val < 256);
    
    // Set the RGB components accordingly
    r = (m_palette[val] >> 16) & 0xFF;
    g = (m_palette[val] >>  8) & 0xFF;
    b = (m_palette[val] >>  0) & 0xFF;
}


uInt32 ColourPalette::getRGB(int val) const {

    return m_palette[val];
}

const uInt32 *ColourPalette::getPalette()
{
    return m_palette;
}

void ColourPalette::setPalette(const uInt32 *palette) {

    m_palette = palette;
}


