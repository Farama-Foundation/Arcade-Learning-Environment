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

#include "ale/common/ColourPalette.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>

#include "ale/common/Palettes.hpp"

namespace ale {
namespace {

inline uint32_t packRGB(uint8_t r, uint8_t g, uint8_t b) {
  return ((uint32_t)r << 16) + ((uint32_t)g << 8) + (uint32_t)b;
}

inline uint32_t convertGrayscale(uint32_t packedRGBValue) {
  double r = (packedRGBValue >> 16) & 0xff;
  double g = (packedRGBValue >> 8) & 0xff;
  double b = (packedRGBValue >> 0) & 0xff;

  uint8_t lum = (uint8_t)round(r * 0.2989 + g * 0.5870 + b * 0.1140);

  return packRGB(lum, lum, lum);
}

}  // namespace

ColourPalette::ColourPalette() : m_palette(NULL) {}

void ColourPalette::getRGB(int val, int& r, int& g, int& b) const {
  assert(m_palette != NULL);
  assert(val >= 0 && val <= 0xFF);
  // Make sure we are reading from RGB, not grayscale.
  assert((val & 0x01) == 0);

  // Set the RGB components accordingly
  r = (m_palette[val] >> 16) & 0xFF;
  g = (m_palette[val] >> 8) & 0xFF;
  b = (m_palette[val] >> 0) & 0xFF;
}

uint8_t ColourPalette::getGrayscale(int val) const {
  assert(m_palette != NULL);
  assert(val >= 0 && val < 0xFF);
  assert((val & 0x01) == 1);

  // Set the RGB components accordingly
  return (m_palette[val + 1] >> 0) & 0xFF;
}

uint32_t ColourPalette::getRGB(int val) const { return m_palette[val]; }

void ColourPalette::applyPaletteRGB(uint8_t* dst_buffer, uint8_t* src_buffer,
                                    std::size_t src_size) {
  uint8_t* p = src_buffer;
  uint8_t* q = dst_buffer;

  for (std::size_t i = 0; i < src_size; i++, p++) {
    int rgb = m_palette[*p];
    *q = (unsigned char)((rgb >> 16)); q++;  // r
    *q = (unsigned char)((rgb >>  8)); q++;  // g
    *q = (unsigned char)((rgb >>  0)); q++;  // b
  }
}

void ColourPalette::applyPaletteRGB(std::vector<unsigned char>& dst_buffer,
                                    uint8_t* src_buffer, std::size_t src_size) {
  dst_buffer.resize(3 * src_size);
  assert(dst_buffer.size() == 3 * src_size);

  uint8_t* p = src_buffer;

  for (std::size_t i = 0; i < src_size * 3; i += 3, p++) {
    int rgb = m_palette[*p];
    dst_buffer[i + 0] = (unsigned char)((rgb >> 16));  // r
    dst_buffer[i + 1] = (unsigned char)((rgb >>  8));  // g
    dst_buffer[i + 2] = (unsigned char)((rgb >>  0));  // b
  }
}

void ColourPalette::applyPaletteGrayscale(uint8_t* dst_buffer, uint8_t* src_buffer,
                                          std::size_t src_size) {
  uint8_t* p = src_buffer;
  uint8_t* q = dst_buffer;

  for (std::size_t i = 0; i < src_size; i++, p++, q++) {
    *q = (unsigned char)(m_palette[*p + 1] & 0xFF);
  }
}

void ColourPalette::applyPaletteGrayscale(
    std::vector<unsigned char>& dst_buffer, uint8_t* src_buffer,
    std::size_t src_size) {
  dst_buffer.resize(src_size);
  assert(dst_buffer.size() == src_size);

  uint8_t* p = src_buffer;

  for (std::size_t i = 0; i < src_size; i++, p++) {
    dst_buffer[i] = (unsigned char)(m_palette[*p + 1] & 0xFF);
  }
}

void ColourPalette::setPalette(const std::string& type,
                               const std::string& displayFormat) {
  // See which format we should be using
  int paletteNum = 0;
  if (type == "standard")
    paletteNum = 0;
  else if (type == "z26")
    paletteNum = 1;
  else if (type == "user" && myUserPaletteDefined)
    paletteNum = 2;

  int paletteFormat = 0;
  if (displayFormat.compare(0, 3, "PAL") == 0)
    paletteFormat = 1;
  else if (displayFormat.compare(0, 5, "SECAM") == 0)
    paletteFormat = 2;

  uint32_t* paletteMapping[3][3] = {
      {NTSCPalette, PALPalette, SECAMPalette},
      {NTSCPaletteZ26, PALPaletteZ26, SECAMPaletteZ26},
      {m_userNTSCPalette, m_userPALPalette, m_userSECAMPalette}};

  m_palette = paletteMapping[paletteNum][paletteFormat];
}

void ColourPalette::loadUserPalette(const std::string& paletteFile) {
  const int bytesPerColor = 3;
  const int NTSCPaletteSize = 128;
  const int PALPaletteSize = 128;
  const int SECAMPaletteSize = 8;

  int expectedFileSize = NTSCPaletteSize * bytesPerColor +
                         PALPaletteSize * bytesPerColor +
                         SECAMPaletteSize * bytesPerColor;

  std::ifstream paletteStream(paletteFile.c_str(), std::ios::binary);
  if (!paletteStream)
    return;

  // Make sure the contains enough data for the NTSC, PAL and SECAM palettes
  // This means 128 colours each for NTSC and PAL, at 3 bytes per pixel
  // and 8 colours for SECAM at 3 bytes per pixel
  paletteStream.seekg(0, std::ios::end);
  std::streampos length = paletteStream.tellg();
  paletteStream.seekg(0, std::ios::beg);

  if (length < expectedFileSize) {
    paletteStream.close();
    std::cerr << "ERROR: invalid palette file " << paletteFile << "\n";
    return;
  }

  // Now that we have valid data, create the user-defined palettes
  uint8_t pixbuf[bytesPerColor]; // Temporary buffer for one 24-bit pixel

  for (int i = 0; i < NTSCPaletteSize; i++) // NTSC palette
  {
    paletteStream.read((char*)pixbuf, bytesPerColor);
    m_userNTSCPalette[(i << 1)] = packRGB(pixbuf[0], pixbuf[1], pixbuf[2]);
    m_userNTSCPalette[(i << 1) + 1] =
        convertGrayscale(m_userNTSCPalette[(i << 1)]);
  }
  for (int i = 0; i < PALPaletteSize; i++) // PAL palette
  {
    paletteStream.read((char*)pixbuf, bytesPerColor);
    m_userPALPalette[(i << 1)] = packRGB(pixbuf[0], pixbuf[1], pixbuf[2]);
    m_userPALPalette[(i << 1) + 1] =
        convertGrayscale(m_userPALPalette[(i << 1)]);
  }

  uint32_t tmpSecam[SECAMPaletteSize *
                  2]; // All 8 24-bit pixels, plus 8 colorloss pixels
  for (int i = 0; i < SECAMPaletteSize; i++) // SECAM palette
  {
    paletteStream.read((char*)pixbuf, bytesPerColor);
    tmpSecam[(i << 1)] = packRGB(pixbuf[0], pixbuf[1], pixbuf[2]);
    tmpSecam[(i << 1) + 1] = convertGrayscale(tmpSecam[(i << 1)]);
  }

  uint32_t* tmpSECAMPalettePtr = m_userSECAMPalette;
  for (int i = 0; i < 16; ++i) {
    memcpy(tmpSECAMPalettePtr, tmpSecam, SECAMPaletteSize * 2);
    tmpSECAMPalettePtr += SECAMPaletteSize * 2;
  }

  paletteStream.close();

  myUserPaletteDefined = true;
}

}  // namespace ale
