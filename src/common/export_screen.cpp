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
 *  export_screen.cpp
 *
 *  The implementation of the ExportScreen class, which is responsible for 
 *  saving the screen matrix to an image file. 
 * 
 *  Note: Most of the code here is taken from Stella's Snapshot.hxx/cxx
 **************************************************************************** */
#include <zlib.h>
#include <fstream>
#include <cstring>
#include <sstream>
#include "export_screen.h"
#include "random_tools.h"

#include <algorithm>

ExportScreen::ExportScreen() {

}

void ExportScreen::save_png(const ALEScreen& screen, const string& filename) {
    uInt8* buffer  = (uInt8*) NULL;
    uInt8* compmem = (uInt8*) NULL;
    ofstream out;

    try {
        pixel_t *screenArray = screen.getArray();
        int i_screen_height = screen.height();
        int i_screen_width = screen.width();

        // Get actual image dimensions. which are not always the same
        // as the framebuffer dimensions
        out.open(filename.c_str(), ios_base::binary);
        if(!out)
            throw "Couldn't open PNG file";

        // PNG file header
        uInt8 header[8] = { 137, 80, 78, 71, 13, 10, 26, 10 };
        out.write((const char*)header, 8);

        // PNG IHDR
        uInt8 ihdr[13];
        ihdr[0]  = i_screen_width >> 24;   // i_screen_width
        ihdr[1]  = i_screen_width >> 16;
        ihdr[2]  = i_screen_width >> 8;
        ihdr[3]  = i_screen_width & 0xFF;
        ihdr[4]  = i_screen_height >> 24;  // i_screen_height
        ihdr[5]  = i_screen_height >> 16;
        ihdr[6]  = i_screen_height >> 8;
        ihdr[7]  = i_screen_height & 0xFF;
        ihdr[8]  = 8;  // 8 bits per sample (24 bits per pixel)
        ihdr[9]  = 2;  // PNG_COLOR_TYPE_RGB
        ihdr[10] = 0;  // PNG_COMPRESSION_TYPE_DEFAULT
        ihdr[11] = 0;  // PNG_FILTER_TYPE_DEFAULT
        ihdr[12] = 0;  // PNG_INTERLACE_NONE
        writePNGChunk(out, "IHDR", ihdr, 13);

        // Fill the buffer with scanline data
        int rowbytes = i_screen_width * 3;
        buffer = new uInt8[(rowbytes + 1) * i_screen_height];
        uInt8* buf_ptr = buffer;
        for(int i = 0; i < i_screen_height; i++) {
            *buf_ptr++ = 0;                  // first byte of row is filter type
            for(int j = 0; j < i_screen_width; j++) {
                int r, g, b;
                get_rgb_from_palette(screenArray[i*i_screen_width+j], r, g, b);
                buf_ptr[j * 3 + 0] = r;
                buf_ptr[j * 3 + 1] = g;
                buf_ptr[j * 3 + 2] = b;
            }
            buf_ptr += rowbytes;                 // add pitch
        }

        // Compress the data with zlib
        uLongf compmemsize = (uLongf)((i_screen_height * (i_screen_width + 1)
                                        * 3 * 1.001 + 1) + 12);
        compmem = new uInt8[compmemsize];
        if(compmem == NULL ||
           (compress(compmem, &compmemsize, buffer, i_screen_height *
                                            (i_screen_width * 3 + 1)) != Z_OK))
            throw "Error: Couldn't compress PNG";

        // Write the compressed framebuffer data
        writePNGChunk(out, "IDAT", compmem, compmemsize);

        // Finish up
        writePNGChunk(out, "IEND", 0, 0);

        // Clean up
        if(buffer)  delete[] buffer;
        if(compmem) delete[] compmem;
        out.close();

    }
    catch(const char *msg)
    {
        if(buffer)  delete[] buffer;
        if(compmem) delete[] compmem;
        out.close();
        cerr << msg << endl;
    }
}

/* *********************************************************************
    Gets the RGB values for a given screen value from the current palette
 ******************************************************************** */
void ExportScreen::get_rgb_from_palette(int val, int& r, int& g, int& b) const {
    assert (pi_palette);
    assert(val < 256);
    
    if (val < 256) {
        // Regular palette
        r = (pi_palette[val] >> 16) & 0xff;
        g = (pi_palette[val] >> 8) & 0xff;
        b = pi_palette[val] & 0xff;
    } 
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void ExportScreen::writePNGChunk(ofstream& out, const char* type, uInt8* data,
                                int size) const {
    // Stuff the length/type into the buffer
    uInt8 temp[8];
    temp[0] = size >> 24;
    temp[1] = size >> 16;
    temp[2] = size >> 8;
    temp[3] = size;
    temp[4] = type[0];
    temp[5] = type[1];
    temp[6] = type[2];
    temp[7] = type[3];

    // Write the header
    out.write((const char*)temp, 8);

    // Append the actual data
    uInt32 crc = crc32(0, temp + 4, 4);
    if(size > 0)
    {
        out.write((const char*)data, size);
        crc = crc32(crc, data, size);
    }

    // Write the CRC
    temp[0] = crc >> 24;
    temp[1] = crc >> 16;
    temp[2] = crc >> 8;
    temp[3] = crc;
    out.write((const char*)temp, 4);
}
