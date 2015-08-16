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

using namespace std;

inline uInt32 packRGB(uInt8 r, uInt8 g, uInt8 b)
{
    return ((uInt32)r << 16) + ((uInt32)g << 8) + (uInt32)b;
}

ColourPalette::ColourPalette():
    m_palette(NULL) {
    calculateGrayscaleValues();
}


void ColourPalette::getRGB(int val, int &r, int &g, int &b) const {

    assert (m_palette != NULL);
    assert(val < 256);
    
    // Set the RGB components accordingly
    r = (m_palette[val] >> 16) & 0xFF;
    g = (m_palette[val] >>  8) & 0xFF;
    b = (m_palette[val] >>  0) & 0xFF;
}


uInt32 ColourPalette::getRGB(int val) const {

    return m_palette[val];
}

const uInt32 *ColourPalette::getPalette()
{
    return m_palette;
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

    m_palette  = availablePalettes[paletteNum][paletteFormat];
}

void ColourPalette::calculateGrayscaleValues()
{
    for (int type = 0; type < 3; type++)
    {
        for (int format = 0; format < 3; format++)
        {
            uInt32 *currentPalette = availablePalettes[type][format];

            // fill the odd numbered palette entries with gray values
            // (calculated using the standard RGB -> grayscale conversion formula)
            for(int i = 0; i < 128; ++i)
            {
                uInt32 pixel = currentPalette[(i <<1)];

                double r = (pixel >> 16) & 0xff;
                double g = (pixel >> 8)  & 0xff;
                double b = (pixel >> 0)  & 0xff;
                uInt8 sum = (uInt8) round(r * 0.2989 + g * 0.5870 + b * 0.1140 );
                pixel = (sum << 16) + (sum << 8) + sum;

                currentPalette[(i <<1)+1] = pixel;
            }
        }
    }
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
        UserNTSCPalette[(i<<1)] = packRGB(pixbuf[0], pixbuf[1], pixbuf[2]);
    }
    for(int i = 0; i < PALPaletteSize; i++)  // PAL palette
    {
        paletteStream.read((char*)pixbuf, bytesPerColor);
        UserPALPalette[(i<<1)] = packRGB(pixbuf[0], pixbuf[1], pixbuf[2]);
    }

    uInt32 tmpSecam[SECAMPaletteSize*2];        // All 8 24-bit pixels, plus 8 colorloss pixels
    for(int i = 0; i < SECAMPaletteSize; i++)    // SECAM palette
    {
        paletteStream.read((char*)pixbuf, bytesPerColor);
        tmpSecam[(i<<1)]   = packRGB(pixbuf[0], pixbuf[1], pixbuf[2]);
        tmpSecam[(i<<1)+1] = 0;
    }

    uInt32*tmpSECAMPalettePtr = UserSECAMPalette;
    for(int i = 0; i < 16; ++i)
    {
        memcpy(tmpSECAMPalettePtr, tmpSecam, SECAMPaletteSize*2);
        tmpSECAMPalettePtr += SECAMPaletteSize*2;
    }

    paletteStream.close();

    myUserPaletteDefined = true;

    // Palettes have been updated, recalculate
    calculateGrayscaleValues();
}

uInt32* ColourPalette::availablePalettes[3][3] = {
    {NTSCPalette,       PALPalette,     SECAMPalette},
    {NTSCPaletteZ26,    PALPaletteZ26,  SECAMPaletteZ26},
    {UserNTSCPalette,   UserPALPalette, UserSECAMPalette}
};

uInt32 ColourPalette::NTSCPalette[256] = {
    0x000000, 0, 0x4a4a4a, 0, 0x6f6f6f, 0, 0x8e8e8e, 0,
    0xaaaaaa, 0, 0xc0c0c0, 0, 0xd6d6d6, 0, 0xececec, 0,
    0x484800, 0, 0x69690f, 0, 0x86861d, 0, 0xa2a22a, 0,
    0xbbbb35, 0, 0xd2d240, 0, 0xe8e84a, 0, 0xfcfc54, 0,
    0x7c2c00, 0, 0x904811, 0, 0xa26221, 0, 0xb47a30, 0,
    0xc3903d, 0, 0xd2a44a, 0, 0xdfb755, 0, 0xecc860, 0,
    0x901c00, 0, 0xa33915, 0, 0xb55328, 0, 0xc66c3a, 0,
    0xd5824a, 0, 0xe39759, 0, 0xf0aa67, 0, 0xfcbc74, 0,
    0x940000, 0, 0xa71a1a, 0, 0xb83232, 0, 0xc84848, 0,
    0xd65c5c, 0, 0xe46f6f, 0, 0xf08080, 0, 0xfc9090, 0,
    0x840064, 0, 0x97197a, 0, 0xa8308f, 0, 0xb846a2, 0,
    0xc659b3, 0, 0xd46cc3, 0, 0xe07cd2, 0, 0xec8ce0, 0,
    0x500084, 0, 0x68199a, 0, 0x7d30ad, 0, 0x9246c0, 0,
    0xa459d0, 0, 0xb56ce0, 0, 0xc57cee, 0, 0xd48cfc, 0,
    0x140090, 0, 0x331aa3, 0, 0x4e32b5, 0, 0x6848c6, 0,
    0x7f5cd5, 0, 0x956fe3, 0, 0xa980f0, 0, 0xbc90fc, 0,
    0x000094, 0, 0x181aa7, 0, 0x2d32b8, 0, 0x4248c8, 0,
    0x545cd6, 0, 0x656fe4, 0, 0x7580f0, 0, 0x8490fc, 0,
    0x001c88, 0, 0x183b9d, 0, 0x2d57b0, 0, 0x4272c2, 0,
    0x548ad2, 0, 0x65a0e1, 0, 0x75b5ef, 0, 0x84c8fc, 0,
    0x003064, 0, 0x185080, 0, 0x2d6d98, 0, 0x4288b0, 0,
    0x54a0c5, 0, 0x65b7d9, 0, 0x75cceb, 0, 0x84e0fc, 0,
    0x004030, 0, 0x18624e, 0, 0x2d8169, 0, 0x429e82, 0,
    0x54b899, 0, 0x65d1ae, 0, 0x75e7c2, 0, 0x84fcd4, 0,
    0x004400, 0, 0x1a661a, 0, 0x328432, 0, 0x48a048, 0,
    0x5cba5c, 0, 0x6fd26f, 0, 0x80e880, 0, 0x90fc90, 0,
    0x143c00, 0, 0x355f18, 0, 0x527e2d, 0, 0x6e9c42, 0,
    0x87b754, 0, 0x9ed065, 0, 0xb4e775, 0, 0xc8fc84, 0,
    0x303800, 0, 0x505916, 0, 0x6d762b, 0, 0x88923e, 0,
    0xa0ab4f, 0, 0xb7c25f, 0, 0xccd86e, 0, 0xe0ec7c, 0,
    0x482c00, 0, 0x694d14, 0, 0x866a26, 0, 0xa28638, 0,
    0xbb9f47, 0, 0xd2b656, 0, 0xe8cc63, 0, 0xfce070, 0
};

uInt32 ColourPalette::PALPalette[256] = {
    0x000000, 0, 0x2b2b2b, 0, 0x525252, 0, 0x767676, 0,
    0x979797, 0, 0xb6b6b6, 0, 0xd2d2d2, 0, 0xececec, 0,
    0x000000, 0, 0x2b2b2b, 0, 0x525252, 0, 0x767676, 0,
    0x979797, 0, 0xb6b6b6, 0, 0xd2d2d2, 0, 0xececec, 0,
    0x805800, 0, 0x96711a, 0, 0xab8732, 0, 0xbe9c48, 0,
    0xcfaf5c, 0, 0xdfc06f, 0, 0xeed180, 0, 0xfce090, 0,
    0x445c00, 0, 0x5e791a, 0, 0x769332, 0, 0x8cac48, 0,
    0xa0c25c, 0, 0xb3d76f, 0, 0xc4ea80, 0, 0xd4fc90, 0,
    0x703400, 0, 0x89511a, 0, 0xa06b32, 0, 0xb68448, 0,
    0xc99a5c, 0, 0xdcaf6f, 0, 0xecc280, 0, 0xfcd490, 0,
    0x006414, 0, 0x1a8035, 0, 0x329852, 0, 0x48b06e, 0,
    0x5cc587, 0, 0x6fd99e, 0, 0x80ebb4, 0, 0x90fcc8, 0,
    0x700014, 0, 0x891a35, 0, 0xa03252, 0, 0xb6486e, 0,
    0xc95c87, 0, 0xdc6f9e, 0, 0xec80b4, 0, 0xfc90c8, 0,
    0x005c5c, 0, 0x1a7676, 0, 0x328e8e, 0, 0x48a4a4, 0,
    0x5cb8b8, 0, 0x6fcbcb, 0, 0x80dcdc, 0, 0x90ecec, 0,
    0x70005c, 0, 0x841a74, 0, 0x963289, 0, 0xa8489e, 0,
    0xb75cb0, 0, 0xc66fc1, 0, 0xd380d1, 0, 0xe090e0, 0,
    0x003c70, 0, 0x195a89, 0, 0x2f75a0, 0, 0x448eb6, 0,
    0x57a5c9, 0, 0x68badc, 0, 0x79ceec, 0, 0x88e0fc, 0,
    0x580070, 0, 0x6e1a89, 0, 0x8332a0, 0, 0x9648b6, 0,
    0xa75cc9, 0, 0xb76fdc, 0, 0xc680ec, 0, 0xd490fc, 0,
    0x002070, 0, 0x193f89, 0, 0x2f5aa0, 0, 0x4474b6, 0,
    0x578bc9, 0, 0x68a1dc, 0, 0x79b5ec, 0, 0x88c8fc, 0,
    0x340080, 0, 0x4a1a96, 0, 0x5f32ab, 0, 0x7248be, 0,
    0x835ccf, 0, 0x936fdf, 0, 0xa280ee, 0, 0xb090fc, 0,
    0x000088, 0, 0x1a1a9d, 0, 0x3232b0, 0, 0x4848c2, 0,
    0x5c5cd2, 0, 0x6f6fe1, 0, 0x8080ef, 0, 0x9090fc, 0,
    0x000000, 0, 0x2b2b2b, 0, 0x525252, 0, 0x767676, 0,
    0x979797, 0, 0xb6b6b6, 0, 0xd2d2d2, 0, 0xececec, 0,
    0x000000, 0, 0x2b2b2b, 0, 0x525252, 0, 0x767676, 0,
    0x979797, 0, 0xb6b6b6, 0, 0xd2d2d2, 0, 0xececec, 0
};

uInt32 ColourPalette::SECAMPalette[256] = {
    0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff50ff, 0,
    0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
    0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff50ff, 0,
    0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
    0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff50ff, 0,
    0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
    0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff50ff, 0,
    0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
    0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff50ff, 0,
    0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
    0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff50ff, 0,
    0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
    0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff50ff, 0,
    0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
    0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff50ff, 0,
    0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
    0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff50ff, 0,
    0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
    0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff50ff, 0,
    0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
    0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff50ff, 0,
    0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
    0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff50ff, 0,
    0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
    0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff50ff, 0,
    0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
    0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff50ff, 0,
    0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
    0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff50ff, 0,
    0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
    0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff50ff, 0,
    0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0
};

uInt32 ColourPalette::NTSCPaletteZ26[256] = {
    0x000000, 0, 0x505050, 0, 0x646464, 0, 0x787878, 0,
    0x8c8c8c, 0, 0xa0a0a0, 0, 0xb4b4b4, 0, 0xc8c8c8, 0,
    0x445400, 0, 0x586800, 0, 0x6c7c00, 0, 0x809000, 0,
    0x94a414, 0, 0xa8b828, 0, 0xbccc3c, 0, 0xd0e050, 0,
    0x673900, 0, 0x7b4d00, 0, 0x8f6100, 0, 0xa37513, 0,
    0xb78927, 0, 0xcb9d3b, 0, 0xdfb14f, 0, 0xf3c563, 0,
    0x7b2504, 0, 0x8f3918, 0, 0xa34d2c, 0, 0xb76140, 0,
    0xcb7554, 0, 0xdf8968, 0, 0xf39d7c, 0, 0xffb190, 0,
    0x7d122c, 0, 0x912640, 0, 0xa53a54, 0, 0xb94e68, 0,
    0xcd627c, 0, 0xe17690, 0, 0xf58aa4, 0, 0xff9eb8, 0,
    0x730871, 0, 0x871c85, 0, 0x9b3099, 0, 0xaf44ad, 0,
    0xc358c1, 0, 0xd76cd5, 0, 0xeb80e9, 0, 0xff94fd, 0,
    0x5d0b92, 0, 0x711fa6, 0, 0x8533ba, 0, 0x9947ce, 0,
    0xad5be2, 0, 0xc16ff6, 0, 0xd583ff, 0, 0xe997ff, 0,
    0x401599, 0, 0x5429ad, 0, 0x683dc1, 0, 0x7c51d5, 0,
    0x9065e9, 0, 0xa479fd, 0, 0xb88dff, 0, 0xcca1ff, 0,
    0x252593, 0, 0x3939a7, 0, 0x4d4dbb, 0, 0x6161cf, 0,
    0x7575e3, 0, 0x8989f7, 0, 0x9d9dff, 0, 0xb1b1ff, 0,
    0x0f3480, 0, 0x234894, 0, 0x375ca8, 0, 0x4b70bc, 0,
    0x5f84d0, 0, 0x7398e4, 0, 0x87acf8, 0, 0x9bc0ff, 0,
    0x04425a, 0, 0x18566e, 0, 0x2c6a82, 0, 0x407e96, 0,
    0x5492aa, 0, 0x68a6be, 0, 0x7cbad2, 0, 0x90cee6, 0,
    0x044f30, 0, 0x186344, 0, 0x2c7758, 0, 0x408b6c, 0,
    0x549f80, 0, 0x68b394, 0, 0x7cc7a8, 0, 0x90dbbc, 0,
    0x0f550a, 0, 0x23691e, 0, 0x377d32, 0, 0x4b9146, 0,
    0x5fa55a, 0, 0x73b96e, 0, 0x87cd82, 0, 0x9be196, 0,
    0x1f5100, 0, 0x336505, 0, 0x477919, 0, 0x5b8d2d, 0,
    0x6fa141, 0, 0x83b555, 0, 0x97c969, 0, 0xabdd7d, 0,
    0x344600, 0, 0x485a00, 0, 0x5c6e14, 0, 0x708228, 0,
    0x84963c, 0, 0x98aa50, 0, 0xacbe64, 0, 0xc0d278, 0,
    0x463e00, 0, 0x5a5205, 0, 0x6e6619, 0, 0x827a2d, 0,
    0x968e41, 0, 0xaaa255, 0, 0xbeb669, 0, 0xd2ca7d, 0
};

uInt32 ColourPalette::PALPaletteZ26[256] = {
    0x000000, 0, 0x4c4c4c, 0, 0x606060, 0, 0x747474, 0,
    0x888888, 0, 0x9c9c9c, 0, 0xb0b0b0, 0, 0xc4c4c4, 0,
    0x000000, 0, 0x4c4c4c, 0, 0x606060, 0, 0x747474, 0,
    0x888888, 0, 0x9c9c9c, 0, 0xb0b0b0, 0, 0xc4c4c4, 0,
    0x533a00, 0, 0x674e00, 0, 0x7b6203, 0, 0x8f7617, 0,
    0xa38a2b, 0, 0xb79e3f, 0, 0xcbb253, 0, 0xdfc667, 0,
    0x1b5800, 0, 0x2f6c00, 0, 0x438001, 0, 0x579415, 0,
    0x6ba829, 0, 0x7fbc3d, 0, 0x93d051, 0, 0xa7e465, 0,
    0x6a2900, 0, 0x7e3d12, 0, 0x925126, 0, 0xa6653a, 0,
    0xba794e, 0, 0xce8d62, 0, 0xe2a176, 0, 0xf6b58a, 0,
    0x075b00, 0, 0x1b6f11, 0, 0x2f8325, 0, 0x439739, 0,
    0x57ab4d, 0, 0x6bbf61, 0, 0x7fd375, 0, 0x93e789, 0,
    0x741b2f, 0, 0x882f43, 0, 0x9c4357, 0, 0xb0576b, 0,
    0xc46b7f, 0, 0xd87f93, 0, 0xec93a7, 0, 0xffa7bb, 0,
    0x00572e, 0, 0x106b42, 0, 0x247f56, 0, 0x38936a, 0,
    0x4ca77e, 0, 0x60bb92, 0, 0x74cfa6, 0, 0x88e3ba, 0,
    0x6d165f, 0, 0x812a73, 0, 0x953e87, 0, 0xa9529b, 0,
    0xbd66af, 0, 0xd17ac3, 0, 0xe58ed7, 0, 0xf9a2eb, 0,
    0x014c5e, 0, 0x156072, 0, 0x297486, 0, 0x3d889a, 0,
    0x519cae, 0, 0x65b0c2, 0, 0x79c4d6, 0, 0x8dd8ea, 0,
    0x5f1588, 0, 0x73299c, 0, 0x873db0, 0, 0x9b51c4, 0,
    0xaf65d8, 0, 0xc379ec, 0, 0xd78dff, 0, 0xeba1ff, 0,
    0x123b87, 0, 0x264f9b, 0, 0x3a63af, 0, 0x4e77c3, 0,
    0x628bd7, 0, 0x769feb, 0, 0x8ab3ff, 0, 0x9ec7ff, 0,
    0x451e9d, 0, 0x5932b1, 0, 0x6d46c5, 0, 0x815ad9, 0,
    0x956eed, 0, 0xa982ff, 0, 0xbd96ff, 0, 0xd1aaff, 0,
    0x2a2b9e, 0, 0x3e3fb2, 0, 0x5253c6, 0, 0x6667da, 0,
    0x7a7bee, 0, 0x8e8fff, 0, 0xa2a3ff, 0, 0xb6b7ff, 0,
    0x000000, 0, 0x4c4c4c, 0, 0x606060, 0, 0x747474, 0,
    0x888888, 0, 0x9c9c9c, 0, 0xb0b0b0, 0, 0xc4c4c4, 0,
    0x000000, 0, 0x4c4c4c, 0, 0x606060, 0, 0x747474, 0,
    0x888888, 0, 0x9c9c9c, 0, 0xb0b0b0, 0, 0xc4c4c4, 0
};

uInt32 ColourPalette::SECAMPaletteZ26[256] = {
    0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff3cff, 0,
    0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
    0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff3cff, 0,
    0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
    0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff3cff, 0,
    0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
    0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff3cff, 0,
    0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
    0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff3cff, 0,
    0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
    0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff3cff, 0,
    0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
    0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff3cff, 0,
    0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
    0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff3cff, 0,
    0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
    0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff3cff, 0,
    0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
    0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff3cff, 0,
    0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
    0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff3cff, 0,
    0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
    0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff3cff, 0,
    0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
    0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff3cff, 0,
    0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
    0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff3cff, 0,
    0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
    0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff3cff, 0,
    0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0,
    0x000000, 0, 0x2121ff, 0, 0xf03c79, 0, 0xff3cff, 0,
    0x7fff00, 0, 0x7fffff, 0, 0xffff3f, 0, 0xffffff, 0
};

uInt32 ColourPalette::UserNTSCPalette[256]  = { 0 }; // filled from external file

uInt32 ColourPalette::UserPALPalette[256]   = { 0 }; // filled from external file

uInt32 ColourPalette::UserSECAMPalette[256] = { 0 }; // filled from external file
