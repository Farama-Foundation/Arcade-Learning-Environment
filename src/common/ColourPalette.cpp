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
#include <math.h>
#include <fstream>
#include "Palettes.hpp"

using namespace std;

inline uInt32 packRGB(uInt8 r, uInt8 g, uInt8 b)
{
    return ((uInt32)r << 16) + ((uInt32)g << 8) + (uInt32)b;
}

inline uInt32 convertGrayscale(uInt32 packedRGBValue)
{
    double r = (packedRGBValue >> 16) & 0xff;
    double g = (packedRGBValue >> 8)  & 0xff;
    double b = (packedRGBValue >> 0)  & 0xff;

    uInt8 lum = (uInt8) round(r * 0.2989 + g * 0.5870 + b * 0.1140);

    return packRGB(lum, lum, lum);
}

ColourPalette::ColourPalette(): m_palette(NULL) {
}


void ColourPalette::getRGB(int val, int &r, int &g, int &b) const
{
    assert(m_palette != NULL);
    assert(val >= 0 && val <= 0xFF);
    // Make sure we are reading from RGB, not grayscale.
    assert((val & 0x01) == 0);
    
    // Set the RGB components accordingly
    r = (m_palette[val] >> 16) & 0xFF;
    g = (m_palette[val] >>  8) & 0xFF;
    b = (m_palette[val] >>  0) & 0xFF;
}

uInt8 ColourPalette::getGrayscale(int val) const
{
    assert(m_palette != NULL);
    assert(val >= 0 && val < 0xFF);
    assert((val & 0x01) == 1);

    // Set the RGB components accordingly
    return (m_palette[val+1] >> 0) & 0xFF;
}

uInt32 ColourPalette::getRGB(int val) const
{
    return m_palette[val];
}

void ColourPalette::applyPaletteRGB(uInt8* dst_buffer, uInt8 *src_buffer, size_t src_size)
{
    uInt8 *p = src_buffer;
    uInt8 *q = dst_buffer;

    for(size_t i = 0; i < src_size; i++, p++){
        int rgb = m_palette[*p];
        *q = (unsigned char) ((rgb >> 16));  q++;    // r
        *q = (unsigned char) ((rgb >>  8));  q++;    // g
        *q = (unsigned char) ((rgb >>  0));  q++;    // b
    }
}

void ColourPalette::applyPaletteRGB(std::vector<unsigned char>& dst_buffer, uInt8 *src_buffer, size_t src_size)
{
    dst_buffer.resize(3 * src_size);
    assert(dst_buffer.size() == 3 * src_size);

    uInt8 *p = src_buffer;

    for(size_t i = 0; i < src_size * 3; i += 3, p++){
        int rgb = m_palette[*p];
        dst_buffer[i+0] = (unsigned char) ((rgb >> 16));    // r
        dst_buffer[i+1] = (unsigned char) ((rgb >>  8));    // g
        dst_buffer[i+2] = (unsigned char) ((rgb >>  0));    // b
    }
}

void ColourPalette::applyPaletteGrayscale(uInt8* dst_buffer, uInt8 *src_buffer, size_t src_size)
{
    uInt8 *p = src_buffer;
    uInt8 *q = dst_buffer;

    for(size_t i = 0; i < src_size; i++, p++, q++){
        *q = (unsigned char) (m_palette[*p+1] & 0xFF);
    }
}

void ColourPalette::applyPaletteGrayscale(std::vector<unsigned char>& dst_buffer, uInt8 *src_buffer, size_t src_size)
{
    dst_buffer.resize(src_size);
    assert(dst_buffer.size() == src_size);

    uInt8 *p = src_buffer;

    for(size_t i = 0; i < src_size; i++, p++){
        dst_buffer[i] = (unsigned char) (m_palette[*p+1] & 0xFF);
    }
}

void ColourPalette::setPalette(const string& type,
                               const string& displayFormat)
{
    // See which format we should be using
    int paletteNum = 0;
    if(type == "standard")
        paletteNum = 0;
    else if(type == "z26")
        paletteNum = 1;
    else if(type == "user" && myUserPaletteDefined)
        paletteNum = 2;

    int paletteFormat = 0;
    if (displayFormat.compare(0, 3, "PAL") == 0)
        paletteFormat = 1;
    else if (displayFormat.compare(0, 5, "SECAM") == 0)
        paletteFormat = 2;

    uInt32* paletteMapping[3][3] = {
        {NTSCPalette,       PALPalette,     SECAMPalette},
        {NTSCPaletteZ26,    PALPaletteZ26,  SECAMPaletteZ26},
        {m_userNTSCPalette, m_userPALPalette, m_userSECAMPalette}
    };

    m_palette  = paletteMapping[paletteNum][paletteFormat];
}

void ColourPalette::loadUserPalette(const string& paletteFile)
{
    const int bytesPerColor = 3;
    const int NTSCPaletteSize = 128;
    const int PALPaletteSize = 128;
    const int SECAMPaletteSize = 8;

    int expectedFileSize =  NTSCPaletteSize * bytesPerColor +
                            PALPaletteSize * bytesPerColor +
                            SECAMPaletteSize * bytesPerColor;

    ifstream paletteStream(paletteFile.c_str(), ios::binary);
    if(!paletteStream)
        return;

    // Make sure the contains enough data for the NTSC, PAL and SECAM palettes
    // This means 128 colours each for NTSC and PAL, at 3 bytes per pixel
    // and 8 colours for SECAM at 3 bytes per pixel
    paletteStream.seekg(0, ios::end);
    streampos length = paletteStream.tellg();
    paletteStream.seekg(0, ios::beg);

    if(length < expectedFileSize)
    {
        paletteStream.close();
        cerr << "ERROR: invalid palette file " << paletteFile << endl;
        return;
    }

    // Now that we have valid data, create the user-defined palettes
    uInt8 pixbuf[bytesPerColor];  // Temporary buffer for one 24-bit pixel

    for(int i = 0; i < NTSCPaletteSize; i++)  // NTSC palette
    {
        paletteStream.read((char*)pixbuf, bytesPerColor);
        m_userNTSCPalette[(i<<1)] = packRGB(pixbuf[0], pixbuf[1], pixbuf[2]);
        m_userNTSCPalette[(i<<1)+1] = convertGrayscale(m_userNTSCPalette[(i<<1)]);
    }
    for(int i = 0; i < PALPaletteSize; i++)  // PAL palette
    {
        paletteStream.read((char*)pixbuf, bytesPerColor);
        m_userPALPalette[(i<<1)] = packRGB(pixbuf[0], pixbuf[1], pixbuf[2]);
        m_userPALPalette[(i<<1)+1] = convertGrayscale(m_userPALPalette[(i<<1)]);
    }

    uInt32 tmpSecam[SECAMPaletteSize*2];         // All 8 24-bit pixels, plus 8 colorloss pixels
    for(int i = 0; i < SECAMPaletteSize; i++)    // SECAM palette
    {
        paletteStream.read((char*)pixbuf, bytesPerColor);
        tmpSecam[(i<<1)]   = packRGB(pixbuf[0], pixbuf[1], pixbuf[2]);
        tmpSecam[(i<<1)+1] = convertGrayscale(tmpSecam[(i<<1)]);
    }

    uInt32*tmpSECAMPalettePtr = m_userSECAMPalette;
    for(int i = 0; i < 16; ++i)
    {
        memcpy(tmpSECAMPalettePtr, tmpSecam, SECAMPaletteSize*2);
        tmpSECAMPalettePtr += SECAMPaletteSize*2;
    }

    paletteStream.close();

    myUserPaletteDefined = true;
}
