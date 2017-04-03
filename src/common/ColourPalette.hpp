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

#include <vector>
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

        /** Returns the byte-sized grayscale value for this palette index. */ 
        uInt8 getGrayscale(int val) const; 

        /**
            Applies the current RGB palette to the src_buffer and returns the results in dst_buffer
            For each byte in src_buffer, three bytes are returned in dst_buffer
            8 bits => 24 bits
         */
        void applyPaletteRGB(uInt8* dst_buffer, uInt8 *src_buffer, size_t src_size);
        void applyPaletteRGB(std::vector<unsigned char>& dst_buffer, uInt8 *src_buffer, size_t src_size);

        /**
            Applies the current grayscale palette to the src_buffer and returns the results in dst_buffer
            For each byte in src_buffer, a single byte is returned in dst_buffer
            8 bits => 8 bits
         */
        void applyPaletteGrayscale(uInt8* dst_buffer, uInt8 *src_buffer, size_t src_size);
        void applyPaletteGrayscale(std::vector<unsigned char>& dst_buffer, uInt8 *src_buffer, size_t src_size);

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
        uInt32 *m_palette;

        bool myUserPaletteDefined;

        // Table of RGB values for NTSC, PAL and SECAM - user-defined
        uInt32 m_userNTSCPalette[256];
        uInt32 m_userPALPalette[256];
        uInt32 m_userSECAMPalette[256];
};

#endif // __COLOUR_PALETTE_HPP__ 


