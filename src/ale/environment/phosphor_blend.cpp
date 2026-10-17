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

#include "ale/environment/phosphor_blend.hpp"

#include <array>
#include <cassert>
#include <map>
#include <mutex>

#include "ale/emucore/Console.hxx"

namespace ale {
using namespace stella;   // OSystem

namespace {

/** The full palette contents. Keying on these rather than on the display format
 *  keeps sharing exact for NTSC/PAL/SECAM and for user-supplied palettes alike. */
using PaletteKey = std::array<uint32_t, 256>;

/** One registry slot. Built at most once, immutable afterwards. */
struct TableEntry {
  // User-provided so that try_emplace() does not zero 512 KiB under the lock.
  TableEntry() {}

  std::once_flag built;
  PhosphorTables tables;
};

/** Shared tables, one entry per distinct palette (512 KiB each), instead of one
 *  per environment. std::map nodes never move, so a pointer into an entry stays
 *  valid while other entries are inserted. The mutex and map are deliberately
 *  leaked: environments may outlive static destruction, and nothing needs to run
 *  at exit. */
std::mutex& registryMutex() {
  static std::mutex* mutex = new std::mutex;
  return *mutex;
}

std::map<PaletteKey, TableEntry>& registry() {
  static auto* entries = new std::map<PaletteKey, TableEntry>;
  return *entries;
}

}  // namespace

PhosphorBlend::PhosphorBlend(OSystem* osystem) : m_osystem(osystem) {
  // The tables are only needed for colour averaging, which is off by default, so
  // most environments never pay for them.
  if (m_osystem->settings().getBool("color_averaging")) {
    m_tables = acquireTables(m_osystem->colourPalette());
  }
}

const PhosphorTables* PhosphorBlend::acquireTables(const ColourPalette& palette) {
  PaletteKey key;
  for (int i = 0; i < 256; ++i) {
    key[i] = palette.getRGB(i);
  }

  TableEntry* entry;
  {
    std::lock_guard<std::mutex> lock(registryMutex());
    entry = &registry().try_emplace(key).first->second;
  }

  // Build outside the registry lock: holding it across the build would serialise
  // construction of environments with different palettes.
  std::call_once(entry->built, [&] { buildTables(entry->tables, palette); });

  return &entry->tables;
}

void PhosphorBlend::process(ALEScreen& screen) {
  assert(m_tables != nullptr && "process() called with colour averaging off");
  Console& console = m_osystem->console();

  // Fetch current and previous frame buffers from the emulator
  uint8_t* current_buffer = console.mediaSource().currentFrameBuffer();
  uint8_t* previous_buffer = console.mediaSource().previousFrameBuffer();

  // Process each pixel in turn
  for (size_t i = 0; i < screen.arraySize(); i++) {
    int cv = current_buffer[i];
    int pv = previous_buffer[i];

    // Find out the corresponding rgb color
    uint32_t rgb = m_tables->avg_palette[cv][pv];

    // Set the corresponding pixel in the array
    screen.getArray()[i] = rgbToNTSC(rgb);
  }
}

void PhosphorBlend::buildTables(PhosphorTables& tables,
                                const ColourPalette& palette) {
  // Precompute the average RGB values for phosphor-averaged colors c1 and c2.
  // Every pair, odd indices included: the TIA greys out an unfinished frame
  // with odd (greyscale) indices, and those are averaged like any other.
  for (int c1 = 0; c1 < 256; c1++) {
    for (int c2 = 0; c2 < 256; c2++) {
      uint32_t rgb1 = palette.getRGB(c1);
      uint32_t rgb2 = palette.getRGB(c2);

      uint8_t r = getPhosphor((rgb1 >> 16) & 0xFF, (rgb2 >> 16) & 0xFF);
      uint8_t g = getPhosphor((rgb1 >> 8) & 0xFF, (rgb2 >> 8) & 0xFF);
      uint8_t b = getPhosphor(rgb1 & 0xFF, rgb2 & 0xFF);
      tables.avg_palette[c1][c2] = makeRGB(r, g, b);
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

        tables.rgb_ntsc[r >> 2][g >> 2][b >> 2] = minIndex;
      }
    }
  }
}

uint8_t PhosphorBlend::getPhosphor(uint8_t v1, uint8_t v2) {
  if (v1 < v2) {
    int tmp = v1;
    v1 = v2;
    v2 = tmp;
  }

  uint32_t blendedValue = ((v1 - v2) * kPhosphorBlendRatio) / 100 + v2;
  if (blendedValue > 255)
    return 255;
  else
    return (uint8_t)blendedValue;
}

uint32_t PhosphorBlend::makeRGB(uint8_t r, uint8_t g, uint8_t b) {
  return (r << 16) | (g << 8) | b;
}

/** Converts a RGB value to an 8-bit format */
uint8_t PhosphorBlend::rgbToNTSC(uint32_t rgb) {
  int r = (rgb >> 16) & 0xFF;
  int g = (rgb >> 8) & 0xFF;
  int b = rgb & 0xFF;

  return m_tables->rgb_ntsc[r >> 2][g >> 2][b >> 2];
}

}  // namespace ale
