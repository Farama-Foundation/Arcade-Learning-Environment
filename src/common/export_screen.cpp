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
#include "OSystem.hxx"
#include "export_screen.h"
#include "random_tools.h"

#include <algorithm>

ExportScreen::ExportScreen(OSystem* osystem) {
    p_osystem = osystem;
    pi_palette = NULL;
    MediaSource& mediasrc = p_osystem->console().mediaSource();
    p_props = &p_osystem->console().properties();
    i_screen_width  = mediasrc.width();
    i_screen_height = mediasrc.height();
    init_custom_palette();
}

        
/* *********************************************************************
    Saves the given screen matrix as a PNG file
 ******************************************************************** */        
void ExportScreen::save_png(const IntMatrix* screen_matrix, const string& filename) {
    uInt8* buffer  = (uInt8*) NULL;
    uInt8* compmem = (uInt8*) NULL;
    ofstream out;
    
    try {
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
                get_rgb_from_palette((*screen_matrix)[i][j], r, g, b);
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
        
        // Add some info about this snapshot
        writePNGText(out, "ROM Name", p_props->get(Cartridge_Name));
        writePNGText(out, "ROM MD5", p_props->get(Cartridge_MD5));
        writePNGText(out, "Display Format", p_props->get(Display_Format));
        
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
    Saves a matrix (e.g. the screen matrix)  as a PNG file
 ******************************************************************** */        
void ExportScreen::export_any_matrix (const IntMatrix* pm_matrix, 
                                     const string& filename) const {
    uInt8* buffer  = (uInt8*) NULL;
    uInt8* compmem = (uInt8*) NULL;
    ofstream out;
    int height = pm_matrix->size();
    int width = (*pm_matrix)[0].size();
    
    try {
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
        ihdr[0]  = width >> 24;   // i_screen_width
        ihdr[1]  = width >> 16;
        ihdr[2]  = width >> 8;
        ihdr[3]  = width & 0xFF;
        ihdr[4]  = height >> 24;  // i_screen_height
        ihdr[5]  = height >> 16;
        ihdr[6]  = height >> 8;
        ihdr[7]  = height & 0xFF;
        ihdr[8]  = 8;  // 8 bits per sample (24 bits per pixel)
        ihdr[9]  = 2;  // PNG_COLOR_TYPE_RGB
        ihdr[10] = 0;  // PNG_COMPRESSION_TYPE_DEFAULT
        ihdr[11] = 0;  // PNG_FILTER_TYPE_DEFAULT
        ihdr[12] = 0;  // PNG_INTERLACE_NONE
        writePNGChunk(out, "IHDR", ihdr, 13);
        
        // Fill the buffer with scanline data
        int rowbytes = width * 3;
        buffer = new uInt8[(rowbytes + 1) * height];
        uInt8* buf_ptr = buffer;
        for(int i = 0; i < height; i++) {
            *buf_ptr++ = 0;                  // first byte of row is filter type
            for(int j = 0; j < width; j++) {
                int r, g, b;
                get_rgb_from_palette((*pm_matrix)[i][j], r, g, b);
                buf_ptr[j * 3 + 0] = r;
                buf_ptr[j * 3 + 1] = g;
                buf_ptr[j * 3 + 2] = b;
            }
            buf_ptr += rowbytes;                 // add pitch
        }
        
        // Compress the data with zlib
        uLongf compmemsize = (uLongf)((height * (width + 1) 
                                        * 3 * 1.001 + 1) + 12);
        compmem = new uInt8[compmemsize];
        if(compmem == NULL ||
           (compress(compmem, &compmemsize, buffer, height * 
                                            (width * 3 + 1)) != Z_OK))
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
    if (val < 256) {
        // Regular palette
        r = (pi_palette[val] >> 16) & 0xff;
        g = (pi_palette[val] >> 8) & 0xff;
        b = pi_palette[val] & 0xff;
    } else {
        // Custom palette 
        val = val - 256;
        assert (val <= v_custom_palette.size());
        r = v_custom_palette[val][0];
        g = v_custom_palette[val][1];
        b = v_custom_palette[val][2];
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

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void ExportScreen::writePNGText(ofstream& out, const string& key, 
                                const string& text) const
{
    int length = key.length() + 1 + text.length() + 1;
    uInt8* data = new uInt8[length];
    
    strcpy((char*)data, key.c_str());
    strcpy((char*)data + key.length() + 1, text.c_str());
    
    writePNGChunk(out, "tEXt", data, length-1);
    
    delete[] data;
}

/* *********************************************************************
    Initializes the custom palette
 ******************************************************************** */    
void ExportScreen::init_custom_palette(void) {
    // add the 216 'web-safe' standard colors
    int shades[] = {0, 51, 102, 153, 204, 255};
    int r, g, b;
    for (int i = 5; i >= 0; i--) {
        for (int j = 5; j >= 0; j--) {
            for (int k = 0; k < 6; k++) {
                r = shades[i];
                g = shades[j];
                b = shades[k];
                if (r == 0 && g == 0 && b == 0) {
                    continue; // we'll add black later
                }
                vector<int> rand_color;
                rand_color.push_back(r);
                rand_color.push_back(g);
                rand_color.push_back(b);
                v_custom_palette.push_back(rand_color);
            }
        }
    }
    std::random_shuffle(v_custom_palette.begin(), v_custom_palette.end() );
    // add CUSTOM_PALLETE_SIZE random colors   
    for (int i = 0; i < CUSTOM_PALETTE_SIZE; i++) {
        r = rand_range(0, 256);
        g = rand_range(0, 256);
        b = rand_range(0, 256);
        vector<int> rand_color;
        rand_color.push_back(r);
        rand_color.push_back(g);
        rand_color.push_back(b);
        v_custom_palette.push_back(rand_color);
    }
    int black[] = {0, 0, 0};
    v_custom_palette[BLACK_COLOR_IND] = vector<int>(black, black + 3);
    int red[] = {255, 0, 0};
    v_custom_palette[RED_COLOR_IND] = vector<int>(red, red + 3);
    int secam_0[] = {0, 0, 0};
    int white[] = {255, 255, 255};
    v_custom_palette[WHITE_COLOR_IND] = vector<int>(white, white + 3);


    v_custom_palette[SECAM_COLOR_IND + 0] = vector<int>(secam_0, secam_0 + 3);
    int secam_1[] = {33, 33, 255};
    v_custom_palette[SECAM_COLOR_IND + 1] = vector<int>(secam_1, secam_1 + 3);
    int secam_2[] = {240, 60, 121};
    v_custom_palette[SECAM_COLOR_IND + 2] = vector<int>(secam_2, secam_2 + 3);
    int secam_3[] = {255, 80, 255};
    v_custom_palette[SECAM_COLOR_IND + 3] = vector<int>(secam_3, secam_3 + 3);
    int secam_4[] = {127, 255, 0};
    v_custom_palette[SECAM_COLOR_IND + 4] = vector<int>(secam_4, secam_4 + 3);
    int secam_5[] = {127, 255, 255};
    v_custom_palette[SECAM_COLOR_IND + 5] = vector<int>(secam_5, secam_5 + 3);
    int secam_6[] = {255, 255, 63};
    v_custom_palette[SECAM_COLOR_IND + 6] = vector<int>(secam_6, secam_6 + 3);
    int secam_7[] = {255, 255, 255};
    v_custom_palette[SECAM_COLOR_IND + 7] = vector<int>(secam_7, secam_7 + 3);
}
