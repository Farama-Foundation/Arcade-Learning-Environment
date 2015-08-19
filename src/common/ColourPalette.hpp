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

#include <string>
// Include obscure header file for uInt32 definition
#include "../emucore/m6502/src/bspf/src/bspf.hxx"

class ColourPalette {

    public:

        ColourPalette();

        /** Converts a given palette value in range [0, 255] into its RGB components. */ 
        void getRGB(int val, int &r, int &g, int &b) const; 

        /** Converts a given palette value into packed RGB (format 0x00RRGGBB). */
        uInt32 getRGB(int val) const;

        /** returns a pointer to the palette array (256 elements) */
        const uInt32 *getPalette();

        /**
          Loads all defined palettes with PAL color-loss data depending
          on 'state'.
          Sets the palette according to the given palette name.

          @param type  The palette type = {standard, z26, user}
          @param displayFormat The display format = { NTSC, PAL, SECAM }
        */
        void setPalette(const std::string& type,
                        const std::string& displayFormat);

        /**
            Loads a user-defined palette file (from OSystem::paletteFile), filling the
            appropriate user-defined palette arrays.
        */
        void loadUserPalette(const std::string& paletteFile);

private:

        /**
         *  Calculates grayscale values for all palettes
         */
        void calculateGrayscaleValues();

        uInt32 *m_palette;

        bool myUserPaletteDefined;

        // Table of RGB values for NTSC, PAL and SECAM
        static uInt32 NTSCPalette[256];
        static uInt32 PALPalette[256];
        static uInt32 SECAMPalette[256];

        // Table of RGB values for NTSC, PAL and SECAM - Z26 version
        static uInt32 NTSCPaletteZ26[256];
        static uInt32 PALPaletteZ26[256];
        static uInt32 SECAMPaletteZ26[256];

        // Table of RGB values for NTSC, PAL and SECAM - user-defined
        static uInt32 UserNTSCPalette[256];
        static uInt32 UserPALPalette[256];
        static uInt32 UserSECAMPalette[256];

        static uInt32* availablePalettes[3][3];
};

#endif // __COLOUR_PALETTE_HPP__ 


