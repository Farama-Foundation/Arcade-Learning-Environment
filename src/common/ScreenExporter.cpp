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

#include "ScreenExporter.hpp"
#include <zlib.h>
#include <sstream>
#include <fstream>
#include "Log.hpp"

// MGB: These methods originally belonged to ExportScreen. Possibly these should be returned to 
// their own class, rather than be static methods. They are here to avoid exposing the gritty 
// details of PNG generation. 
static void writePNGChunk(std::ofstream& out, const char* type, uInt8* data, int size) {

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


static void writePNGHeader(std::ofstream& out, const ALEScreen &screen, bool doubleWidth = true) {

        int width = doubleWidth ? screen.width() * 2: screen.width();
        int height = screen.height();
        // PNG file header
        uInt8 header[8] = { 137, 80, 78, 71, 13, 10, 26, 10 };
        out.write((const char*)header, sizeof(header));

        // PNG IHDR
        uInt8 ihdr[13];
        ihdr[0]  = (width >> 24) & 0xFF;   // width
        ihdr[1]  = (width >> 16) & 0xFF;
        ihdr[2]  = (width >>  8) & 0xFF;
        ihdr[3]  = (width >>  0) & 0xFF;
        ihdr[4]  = (height >> 24) & 0xFF;  // height
        ihdr[5]  = (height >> 16) & 0xFF;
        ihdr[6]  = (height >>  8) & 0xFF;
        ihdr[7]  = (height >>  0) & 0xFF;
        ihdr[8]  = 8;  // 8 bits per sample (24 bits per pixel)
        ihdr[9]  = 2;  // PNG_COLOR_TYPE_RGB
        ihdr[10] = 0;  // PNG_COMPRESSION_TYPE_DEFAULT
        ihdr[11] = 0;  // PNG_FILTER_TYPE_DEFAULT
        ihdr[12] = 0;  // PNG_INTERLACE_NONE
        writePNGChunk(out, "IHDR", ihdr, sizeof(ihdr));
}


static void writePNGData(std::ofstream &out, const ALEScreen &screen, const ColourPalette &palette, bool doubleWidth = true) {

    int dataWidth = screen.width(); 
    int width = doubleWidth ? dataWidth * 2 : dataWidth; 
    int height = screen.height();
   
    // If so desired, double the width

    // Fill the buffer with scanline data
    int rowbytes = width * 3;

    std::vector<uInt8> buffer((rowbytes + 1) * height, 0);
    uInt8* buf_ptr = &buffer[0];

    for(int i = 0; i < height; i++) {
        *buf_ptr++ = 0;                  // first byte of row is filter type
        for(int j = 0; j < dataWidth; j++) {
            int r, g, b;

            palette.getRGB(screen.getArray()[i * dataWidth + j], r, g, b);
            // Double the pixel width, if so desired
            int jj = doubleWidth ? 2 * j : j;

            buf_ptr[jj * 3 + 0] = r;
            buf_ptr[jj * 3 + 1] = g;
            buf_ptr[jj * 3 + 2] = b;
            
            if (doubleWidth) {
                
                jj = jj + 1;

                buf_ptr[jj * 3 + 0] = r;
                buf_ptr[jj * 3 + 1] = g;
                buf_ptr[jj * 3 + 2] = b;
            }
        }
        buf_ptr += rowbytes;                 // add pitch
    }

    // Compress the data with zlib
    uLongf compmemsize = (uLongf)((height * (width + 1) * 3 + 1) + 12);
    std::vector<uInt8> compmem(compmemsize, 0);
    
    if((compress(&compmem[0], &compmemsize, &buffer[0], height * (width * 3 + 1)) != Z_OK)) {

        // @todo -- throw a proper exception
        ale::Logger::Error << "Error: Couldn't compress PNG" << std::endl;
        return;
    }

    // Write the compressed framebuffer data
    writePNGChunk(out, "IDAT", &compmem[0], compmemsize);
}


static void writePNGEnd(std::ofstream &out) {

    // Finish up
    writePNGChunk(out, "IEND", 0, 0);
}

ScreenExporter::ScreenExporter(ColourPalette &palette):
    m_palette(palette),
    m_frame_number(0),
    m_frame_field_width(6) {
}


ScreenExporter::ScreenExporter(ColourPalette &palette, const std::string &path):
    m_palette(palette),
    m_frame_number(0),
    m_frame_field_width(6),
    m_path(path) {
}


void ScreenExporter::save(const ALEScreen &screen, const std::string &filename) const {

    // Open file for writing 
    std::ofstream out(filename.c_str(), std::ios_base::binary);
    if (!out.good()) {
        
        // @todo exception
        ale::Logger::Error << "Could not open " << filename << " for writing" << std::endl;
        return;
    }

    // Now write the PNG proper
    writePNGHeader(out, screen, true);
    writePNGData(out, screen, m_palette, true);
    writePNGEnd(out);

    out.close();
}

void ScreenExporter::saveNext(const ALEScreen &screen) {

    // Must have specified a directory. 
    assert(m_path.size() > 0);

    // MGB: It would be nice here to automagically create paths, but the only way I know of 
    // doing this cleanly is via boost, which we don't include.

    // Construct the filename from basepath & current frame number
    std::ostringstream oss;
    oss << m_path << "/" << 
        std::setw(m_frame_field_width) << std::setfill('0') << m_frame_number << ".png";

    // Save the png
    save(screen, oss.str());

    m_frame_number++;
}


