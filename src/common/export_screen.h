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
#include "../environment/ale_screen.hpp"

class OSystem;

class ExportScreen {
    /* *************************************************************************
        This class is responsible for saving the screen matrix to an image file. 

        Instance Variables:
            - pi_palette        An array containing the palette
    ************************************************************************* */
    public:
        ExportScreen();

        /* *********************************************************************
            Saves the given screen as a PNG file
         ******************************************************************** */
        void save_png(const ALEScreen& screen, const string& filename);

    protected:
        /* *********************************************************************
            Initializes the custom palette
         ******************************************************************** */    
        void writePNGChunk(ofstream& out, const char* type, uInt8* data, int size) const;
};

#endif // __EXPORT_SCREEN_H__
