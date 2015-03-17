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
 *  ColourPalette.hpp 
 *
 *  Enables conversion from NTSC/SECAM/PAL to RGB via the OSystem's palette.
 **************************************************************************** */

#ifndef __COLOUR_PALETTE_HPP__
#define __COLOUR_PALETTE_HPP__ 

// Include obscure header file for uInt32 definition
#include "../emucore/m6502/src/bspf/src/bspf.hxx"

class ColourPalette {

    public:

        ColourPalette();

        /** Converts a given palette value in range [0, 255] into its RGB components. */ 
        void getRGB(int val, int &r, int &g, int &b) const; 

        /** Converts a given palette value into packed RGB (format 0x00RRGGBB). */
        uInt32 getRGB(int val) const;

    private:

        friend class Console; 

        /** Sets the palette (provided by Console). */
        void setPalette(const uInt32 *palette);

        /** We don't own this array; it is owned by OSystem. */
        const uInt32 *m_palette;
};

#endif // __COLOUR_PALETTE_HPP__ 


