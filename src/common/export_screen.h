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
 *  export_screen.h
 *
 *  The implementation of the ExportScreen class, which is responsible for 
 *  saving the screen matrix to an image file. 
 * 
 *  Note: Most of the code here is taken from Stella's Snapshot.hxx/cxx
 **************************************************************************** */

#ifndef EXPORT_SCREEN_H
#define EXPORT_SCREEN_H

#include <vector>
#include "Constants.h"
#include "ale_screen.hpp"

class OSystem;

class ExportScreen {
    /* *************************************************************************
        This class is responsible for saving the screen matrix to an image file. 

        Instance Variables:
            - pi_palette        An array containing the palette
            - p_props           Pointer to a Properties object
            - p_osystem         pointer to the Osystem object
            - i_screen_width    Width of the screen
            - i_screen_height   Height of the screen
            - v_custom_palette  Holds the rgb values for custom colors used
                                for drawing external info on the screen
    ************************************************************************* */
    public:
        ExportScreen();
         virtual ~ExportScreen() {}

        /* *********************************************************************
            Sets the default palette. This needs to be called before any
            export methods can be called.
         ******************************************************************** */
        virtual void set_palette(const uInt32* palette) {
            pi_palette = palette;
        }

        /* *********************************************************************
            Saves the given screen as a PNG file
         ******************************************************************** */
        void save_png(const ALEScreen& screen, const string& filename);

        /* *********************************************************************
            Gets the RGB values for a given screen value from the current palette
         ******************************************************************** */
        void get_rgb_from_palette(int val, int& r, int& g, int& b) const;

    protected:
        /* *********************************************************************
            Initializes the custom palette
         ******************************************************************** */    
        void init_custom_palette(void);
        void writePNGChunk(ofstream& out, const char* type, uInt8* data, int size) const;
        const uInt32* pi_palette;
        vector< vector <int> > v_custom_palette;
};

#endif // __EXPORT_SCREEN_H__
