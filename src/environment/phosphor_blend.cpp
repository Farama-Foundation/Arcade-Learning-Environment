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
 *  phosphor_blend.cpp
 *
 *  Methods for performing colour averaging over the screen.
 *  
 **************************************************************************** */

#include "phosphor_blend.hpp"
#include "../emucore/Console.hxx"

PhosphorBlend::PhosphorBlend(OSystem * osystem):
    m_osystem(osystem) {
  
  // Taken from default Stella settings
  m_phosphor_blend_ratio = 77;

  makeAveragePalette();
}

void PhosphorBlend::process(ALEScreen& screen) {
  Console& console = m_osystem->console();

  // Fetch current and previous frame buffers from the emulator
  uInt8 * current_buffer  = console.mediaSource().currentFrameBuffer();
  uInt8 * previous_buffer = console.mediaSource().previousFrameBuffer();

  // Process each pixel in turn
  for (size_t i = 0; i < screen.arraySize(); i++) { 
    int cv = current_buffer[i];
    int pv = previous_buffer[i];
    
    // Find out the corresponding rgb color 
    uInt32 rgb = m_avg_palette[cv][pv];

    // Set the corresponding pixel in the array
    screen.getArray()[i] = rgbToNTSC(rgb);
  }
}
void PhosphorBlend::makeAveragePalette() {
  
  ColourPalette &palette = m_osystem->colourPalette();

  // Precompute the average RGB values for phosphor-averaged colors c1 and c2.
  for (int c1 = 0; c1 < 256; c1 += 2) {
    for (int c2 = 0; c2 < 256; c2 += 2) {
      int r1, g1, b1;
      int r2, g2, b2;
      palette.getRGB(c1, r1, g1, b1);
      palette.getRGB(c2, r2, g2, b2);

      uInt8 r = getPhosphor(r1, r2);
      uInt8 g = getPhosphor(g1, g2);
      uInt8 b = getPhosphor(b1, b2);
      m_avg_palette[c1][c2] = makeRGB(r, g, b);
    }
  }
  
  // Also make a RGB to NTSC color map. We drop the lowest two bits to speed
  // the initialization a little. TODO(mgbellemare): Find a better solution.
  for (int r = 0; r < 256; r += 4) {
    for (int g = 0; g < 256; g += 4) {  
      for (int b = 0; b < 256; b += 4) {
        // For each RGB point, we find its closest NTSC match
        int minDist = 256 * 3 + 1;
        int minIndex = -1;

        // Look for the closest NTSC value matching (r,g,b). Odd palette
        // entries correspond to grayscale values and are ignored.
        for (int c1 = 0; c1 < 256; c1 += 2) {
          // Get the RGB corresponding to c1
          int r1, g1, b1;
          palette.getRGB(c1, r1, g1, b1);

          int dist = abs(r1 - r) + abs(g1 - g) + abs(b1 - b);
          if (dist < minDist) {
            minDist = dist;
            minIndex = c1;
          }
        }

        m_rgb_ntsc[r >> 2][g >> 2][b >> 2] = minIndex;
      }
    }
  }
}

uInt8 PhosphorBlend::getPhosphor(uInt8 v1, uInt8 v2) {
  if (v1 < v2) {
    int tmp = v1;
    v1 = v2;
    v2 = tmp;
  }

  uInt32 blendedValue = ((v1 - v2) * m_phosphor_blend_ratio) / 100 + v2;
  if (blendedValue > 255) return 255;
  else return (uInt8) blendedValue;
}

uInt32 PhosphorBlend::makeRGB(uInt8 r, uInt8 g, uInt8 b) {
  return (r << 16) | (g << 8) | b;
}

/** Converts a RGB value to an 8-bit format */
uInt8 PhosphorBlend::rgbToNTSC(uInt32 rgb) {
  int r = (rgb >> 16) & 0xFF;
  int g = (rgb >> 8) & 0xFF;
  int b = rgb & 0xFF;

  return m_rgb_ntsc[r >> 2][g >> 2][b >> 2];
}

