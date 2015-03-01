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
 *  ScreenExporter.hpp 
 *
 *  A class for exporting Atari 2600 frames as PNGs.
 *
 **************************************************************************** */

#ifndef __SCREEN_EXPORTER_HPP__
#define __SCREEN_EXPORTER_HPP__ 

#include <string>
#include "display_screen.h"
#include "ale_screen.hpp"

class ScreenExporter {

    public:

        /** Creates a new ScreenExporter which can be used to save screens using save(filename). */ 
        ScreenExporter(ColourPalette &palette);

        /** Creates a new ScreenExporter which will save frames successively as basename + ######, where
            ###### is an integer between 000000 and 999999. */
        ScreenExporter(ColourPalette &palette, const std::string &basename);

        /** Save the given screen to the given filename. No paths are created. */
        void save(const ALEScreen &screen, const std::string &filename) const;

        /** Save the given screen according to our own internal numbering. */
        void saveNext(const ALEScreen &screen);

    private:

        ColourPalette &m_palette;

        /** The next frame number. */
        int m_frame_number;

        /** The width of the frame number when constructing filenames (set to 6). */
        int m_frame_field_width;

        /** The base path, to which we append the frame number. */
        std::string m_basename;
};

#endif // __SCREEN_EXPORTER_HPP__ 



