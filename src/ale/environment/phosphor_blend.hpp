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
 *  phosphor_blend.hpp
 *
 *  Methods for performing colour averaging over the screen.
 *
 **************************************************************************** */

#ifndef __PHOSPHOR_BLEND_HPP__
#define __PHOSPHOR_BLEND_HPP__

#include <cstdint>

#include "ale/common/ColourPalette.hpp"
#include "ale/emucore/OSystem.hxx"
#include "ale/environment/ale_screen.hpp"

namespace ale {

/** The two lookup tables used for colour averaging, 512 KiB in total. They are a
 *  pure function of the colour palette (the blend ratio is a constant), so
 *  environments with the same palette share one immutable copy rather than
 *  building one each. */
struct PhosphorTables {
  uint8_t rgb_ntsc[64][64][64];
  uint32_t avg_palette[256][256];
};

class PhosphorBlend {
 public:
  /** Acquires the averaging tables only if the `color_averaging` setting is on;
   *  otherwise the blender is inert and process() must not be called. */
  PhosphorBlend(stella::OSystem*);

  /** Whether colour averaging is on, i.e. whether process() may be called. */
  bool enabled() const { return m_tables != nullptr; }

  void process(ALEScreen& screen);

 private:
  /** Returns the shared tables for this palette, building them on first use. The
   *  tables live until the process exits. */
  static const PhosphorTables* acquireTables(const ColourPalette& palette);

  static void buildTables(PhosphorTables& tables, const ColourPalette& palette);
  static uint8_t getPhosphor(uint8_t v1, uint8_t v2);
  static uint32_t makeRGB(uint8_t r, uint8_t g, uint8_t b);
  /** Converts a RGB value to an 8-bit format */
  uint8_t rgbToNTSC(uint32_t rgb);

  /** Taken from default Stella settings. */
  static constexpr uint8_t kPhosphorBlendRatio = 77;

 private:
  stella::OSystem* m_osystem;

  /** Shared, immutable; null when colour averaging is off. */
  const PhosphorTables* m_tables = nullptr;
};

}  // namespace ale

#endif  // __PHOSPHOR_BLEND_HPP__
